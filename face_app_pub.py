"""
顔認証 写真仕分けアプリ for vvaren NAGASAKI（公開版）
==================================================================
- 対象写真はブラウザからアップロード
- 仕分け結果は ZIP でダウンロード

エンジン: InsightFace (buffalo_l) / CPU
"""

from __future__ import annotations

import importlib.util
import io
import json
import re
import tempfile
import time
import unicodedata
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import streamlit as st

# ---------------------------------------------------------------- 定数

APP_DIR = Path(__file__).resolve().parent
REF_DIR = APP_DIR / "ref"          # リポジトリ同梱の参照画像
README_PATH = APP_DIR / "README.md"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
UPLOAD_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]

NAMING_RULE = (
    "**ファイル名（拡張子を除いた部分）がそのまま人物名になります。**"
    "選手は `MF05_山田太郎.jpg` のように「ポジション+背番号_氏名」、"
    "スタッフ・関係者は `コーチ_田中三郎.jpg` `MC_鈴木五郎.jpg` のように「役職_氏名」で"
    "保存してください（同じ人物名を追加すると上書きされます）。"
)
TOP_K = 8
OTHER_NAME = "その他"
OTHER_LABEL = "その他（一覧にない人）"
CHUNK_SECONDS = 1.5
MAX_FILES = 200            # 一度に扱える写真の枚数
MAX_SIDE = 2000            # 解析時に長辺をこのサイズへ縮小（サーバーのメモリ対策）
ZIP_WARN_MB = 200
ZIP_LIMIT_MB = 500

ROLE_DIRS = {"staff": "staff", "スタッフ": "staff", "etc": "etc", "その他": "etc"}
STAFF_GROUP = "staff"
POS_ORDER = ["GK", "DF", "MF", "FF", "FW", "監督", "SP", STAFF_GROUP, "etc", "その他"]
STAFF_ROLE_ORDER = {
    "監督": 0, "ヘッドコーチ": 1, "アシスタントコーチ": 2, "コーチ": 3,
    "GKコーチ": 4, "フィジカルコーチ": 5, "分析コーチ": 6,
    "チーフトレーナー": 10, "トレーナー": 11, "フィジオセラピスト": 12,
    "チーフマネージャー": 20, "マネージャー": 21, "キット": 22, "通訳": 23,
}

st.set_page_config(page_title="顔認証 写真仕分け", page_icon="🧑‍🤝‍🧑", layout="wide")


# ---------------------------------------------------------------- モデル


def check_deps() -> list[str]:
    """未インストールのパッケージ名を返す。"""
    missing = []
    for mod, pkg in [("insightface", "insightface"), ("onnxruntime", "onnxruntime"),
                     ("cv2", "opencv-python"), ("PIL", "Pillow"), ("pandas", "pandas")]:
        if importlib.util.find_spec(mod) is None:
            missing.append(pkg)
    return missing


@st.cache_resource(show_spinner="顔認識モデルを読み込み中…（初回はモデルのダウンロードに数分かかります）")
def load_model(det_size: int = 640, det_thresh: float = 0.50):
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_thresh=det_thresh, det_size=(det_size, det_size))
    return app


def get_model(det_size: int = 640, det_thresh: float = 0.50):
    """モデル取得。失敗時は分かりやすいエラーを出して停止する。"""
    try:
        return load_model(det_size, det_thresh)
    except ModuleNotFoundError as e:
        st.error(f"**必要なライブラリがありません: {e.name}**\n\n"
                 "`requirements.txt` を確認してください。")
        st.stop()
    except Exception as e:
        st.error(f"**モデルの読み込みに失敗しました**\n\n```\n{type(e).__name__}: {e}\n```")
        st.info("初回はモデル（約300MB）を自動ダウンロードします。"
                "時間をおいて再読み込みしてください。")
        st.stop()


def read_image(path: Path, max_side: int = MAX_SIDE) -> np.ndarray | None:
    """日本語パス対応の画像読み込み（BGR）。EXIF回転も反映。

    max_side を指定すると長辺をそのサイズまで縮小する（0 は原寸）。
    """
    import cv2
    from PIL import Image, ImageOps

    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            if max_side and max(im.size) > max_side:
                r = max_side / max(im.size)
                im = im.resize((int(im.width * r), int(im.height * r)), Image.LANCZOS)
            arr = np.asarray(im)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception:
        try:
            buf = np.fromfile(str(path), dtype=np.uint8)
            return cv2.imdecode(buf, cv2.IMREAD_COLOR)
        except Exception:
            return None


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


# ---------------------------------------------------------------- 人物名の解析


# 役職フォルダ由来の人物名 → グループ名（collect_refs が更新）
ROLE_MEMBERS: dict[str, str] = {}


def nfc(s: str) -> str:
    """macOS のファイル名は濁点などが分解された形（NFD）で返るため、比較・表示前に正規化する。"""
    return unicodedata.normalize("NFC", s)


def parse_person(name: str) -> tuple[str, str, str]:
    """人物名を (グループ, 番号または役職, 氏名) に分解する。

    選手 : 'MF05_山口蛍'      -> ('MF', '05', '山口蛍')
    監督 : '監督_高木琢也'     -> ('監督', '監督', '高木琢也')
    staff: 'フィジカルコーチ_櫻井朋泰' -> ('staff', 'フィジカルコーチ', '櫻井朋泰')
    etc  : 'MC_初村さん'       -> ('etc', 'MC', '初村さん')
    """
    # staff / etc フォルダの人物は「役職_氏名」として扱う
    if name in ROLE_MEMBERS:
        role, sep, person = nfc(name).partition("_")
        role, person = role.strip(), person.strip()
        if not sep:
            role, person = "スタッフ", nfc(name)
        group = "監督" if role == "監督" else ROLE_MEMBERS[name]
        return group, role, (person or nfc(name))

    m = re.match(r"^([A-Za-z]+|監督|コーチ|スタッフ)(\d*)[_\-\s]*(.*)$", nfc(name))
    if not m:
        return "その他", "", nfc(name)
    pos, num, person = m.group(1).upper(), m.group(2), m.group(3).strip()
    if m.group(1) in ("監督", "コーチ", "スタッフ"):
        pos = m.group(1)
    if pos not in POS_ORDER:
        pos = "その他"
    return pos, num, person or nfc(name)


def group_by_position(names: list[str]) -> dict[str, list[tuple[str, str, str]]]:
    """{グループ: [(番号または役職, 氏名, 元のファイル名), ...]}"""
    groups: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for n in names:
        pos, label, person = parse_person(n)
        groups[pos].append((label, person, n))
    for key, g in groups.items():
        if key in (STAFF_GROUP, "etc", "監督"):
            g.sort(key=lambda x: (STAFF_ROLE_ORDER.get(nfc(x[0]), 99), x[0], x[1]))
        else:
            g.sort(key=lambda x: (int(x[0]) if x[0].isdigit() else 999, x[1]))
    return {p: groups[p] for p in POS_ORDER if p in groups}


def short_label(name: str) -> str:
    """選択肢用の短い表示（拡張子なし）"""
    if name == OTHER_NAME:
        return OTHER_LABEL
    pos, label, person = parse_person(name)
    if name in ROLE_MEMBERS:
        return f"{label} {person}".strip()
    return f"{pos}{label} {person}".strip() if label else f"{pos} {person}".strip()


# ---------------------------------------------------------------- 背番号OCR


@st.cache_resource(show_spinner="背番号の読み取りモデルを準備中…（初回のみ）")
def _load_ocr():
    """背番号読み取り用の OCR を用意する。失敗時は例外を投げる。

    失敗を返り値にすると @st.cache_resource がその失敗を覚えてしまい、
    あとからインストールしても直らない。例外はキャッシュされないため投げる。
    """
    import importlib

    importlib.invalidate_caches()  # アプリ起動後に pip install した場合に備える
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        from rapidocr import RapidOCR  # 新しいパッケージ名（rapidocr 3.x）
    return RapidOCR()


def load_ocr():
    """(OCR, 失敗理由) を返す。理由は画面にそのまま出して切り分けに使う。"""
    try:
        return _load_ocr(), None
    except ImportError as e:
        return None, f"未インストール（{e}）"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def number_of(name: str) -> str | None:
    """人物名から背番号を取り出す（'MF05_山口蛍' -> '5'）。役職スタッフは None。"""
    if name in ROLE_MEMBERS:
        return None
    _, num, _ = parse_person(name)
    return str(int(num)) if num.isdigit() else None


def build_number_map(names: list[str]) -> dict[str, list[str]]:
    """背番号 -> 人物名（同じ番号が複数いる場合は候補が複数）"""
    m: dict[str, list[str]] = defaultdict(list)
    for n in names:
        num = number_of(n)
        if num:
            m[num].append(n)
    return dict(m)


OCR_SCALES = (1600, 0)   # 読み取りに使う長辺サイズ（0 = 原寸）


def _scan_numbers(ocr, img_bgr: np.ndarray, min_ratio: float, valid: set[str],
                  min_conf: float) -> list[tuple[str, float, float, tuple]]:
    """1枚の画像から数字を拾う。座標は画像サイズで正規化して返す。"""
    import cv2

    try:
        result, _ = ocr(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    except Exception:
        return []
    h, w = img_bgr.shape[:2]
    found = []
    for box, text, conf in (result or []):
        t = str(text).strip()
        if not re.fullmatch(r"\d{1,2}", t) or float(conf) < min_conf:
            continue
        num = str(int(t))
        if num not in valid:
            continue
        xs = [p[0] / w for p in box]
        ys = [p[1] / h for p in box]
        ratio = (max(ys) - min(ys)) * 100
        if ratio < min_ratio:
            continue
        found.append((num, float(conf), ratio, (min(xs), min(ys), max(xs), max(ys))))
    return found


def _contained(a: tuple, b: tuple) -> float:
    """重なり ÷ 小さい方の面積。片方がもう片方に含まれていれば 1 に近づく。"""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / small if small > 0 else 0.0


def read_numbers(ocr, img_path_or_img, min_ratio: float, valid: set[str],
                 min_conf: float = 0.50) -> list[tuple[str, float, float]]:
    """背番号らしい数字を読み取り (番号, 信頼度, 画像高さ比%) を返す。

    複数の解像度で読み取って統合する。背番号が大きく写りすぎていると
    原寸では「41」が「4」に割れることがあり、縮小すると正しく読める。
    逆に小さい背番号は原寸のほうが読めるため、両方を試して良いほうを採る。

    看板や観客席の数字を拾わないよう、文字の高さが画像高さの min_ratio(%) 未満、
    または登録メンバーにない番号は除外する。
    """
    if ocr is None:
        return []

    cands: list[tuple[str, float, float, tuple]] = []
    for side in OCR_SCALES:
        img = (read_image(img_path_or_img, max_side=side)
               if isinstance(img_path_or_img, Path) else img_path_or_img)
        if img is None:
            continue
        cands += _scan_numbers(ocr, img, min_ratio, valid, min_conf)
        if not isinstance(img_path_or_img, Path):
            break   # 画像を直接渡された場合は1回だけ

    # 同じ位置の読み取りは桁数の多いほうを採用（「41」を「4」と誤読するのを防ぐ）
    cands.sort(key=lambda x: (-len(x[0]), -x[1]))
    kept: list[tuple[str, float, float, tuple]] = []
    for c in cands:
        if any(_contained(c[3], k[3]) > 0.5 for k in kept):
            continue
        kept.append(c)

    best: dict[str, tuple[str, float, float]] = {}
    for num, conf, ratio, _ in kept:
        if num not in best or ratio > best[num][2]:
            best[num] = (num, conf, round(ratio, 1))
    return sorted(best.values(), key=lambda x: -x[2])


# ---------------------------------------------------------------- 参照画像


def collect_refs(ref_dir: Path) -> dict[str, list[Path]]:
    """参照画像を集める。

    - ref 直下の画像            : ファイル名（拡張子なし）= 人物名
    - staff / etc フォルダ      : 中の各画像が1人。ファイル名は「役職_氏名」
    - その他のサブフォルダ       : フォルダ名 = 人物名（同一人物の複数枚）

    これらのフォルダが無い場合は何も変わりません。
    """
    global ROLE_MEMBERS
    people: dict[str, list[Path]] = {}
    roles: dict[str, str] = {}
    if not ref_dir.is_dir():
        ROLE_MEMBERS = roles
        return people

    for entry in sorted(ref_dir.iterdir()):
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            group = ROLE_DIRS.get(entry.name.lower())
            if group:
                for p in sorted(entry.rglob("*")):
                    if p.suffix.lower() in IMG_EXTS and not p.name.startswith("."):
                        people.setdefault(p.stem, []).append(p)
                        roles[p.stem] = group
            else:
                imgs = [p for p in sorted(entry.rglob("*")) if p.suffix.lower() in IMG_EXTS]
                if imgs:
                    people.setdefault(entry.name, []).extend(imgs)
        elif entry.suffix.lower() in IMG_EXTS:
            people.setdefault(entry.stem, []).append(entry)

    ROLE_MEMBERS = roles
    return people


def collect_all_refs(base_dir: Path, extra_dir: Path) -> dict[str, list[Path]]:
    """同梱の参照画像と、このセッションで追加された参照画像をまとめる。"""
    global ROLE_MEMBERS
    base = collect_refs(base_dir)
    roles = dict(ROLE_MEMBERS)
    extra = collect_refs(extra_dir)
    roles.update(ROLE_MEMBERS)
    merged = {k: list(v) for k, v in base.items()}
    for name, files in extra.items():
        merged.setdefault(name, []).extend(files)
    ROLE_MEMBERS = roles
    return merged


def safe_name(name: str) -> str:
    """ZIP 内のフォルダ名として安全な文字列にする（Windows対応・NFC正規化）。"""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", nfc(name)).strip(" .")
    return cleaned or "unknown"


def estimate_zip_size(plan: dict, photos: list[dict], include_unmatched: bool) -> int:
    total = 0
    for i, names in plan.items():
        src = Path(photos[i]["path"])
        if not src.exists():
            continue
        copies = len(names) if names else (1 if include_unmatched else 0)
        total += src.stat().st_size * copies
    return total


def shrink_bytes(src: Path, max_side: int, quality: int = 88) -> bytes:
    from PIL import Image, ImageOps

    try:
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            if max(im.size) > max_side:
                r = max_side / max(im.size)
                im = im.resize((int(im.width * r), int(im.height * r)), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=quality, optimize=True)
            return buf.getvalue()
    except Exception:
        return src.read_bytes()


def build_zip(plan: dict, photos: list[dict], include_unmatched: bool,
              shrink_to: int | None = None, progress=None) -> bytes:
    """人物名フォルダごとに写真を格納した ZIP を作る。"""
    buf = io.BytesIO()
    items = list(plan.items())
    written: set[str] = set()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        for j, (i, names) in enumerate(items, 1):
            src = Path(photos[i]["path"])
            if not src.exists():
                continue
            dests = names if names else ([OTHER_NAME] if include_unmatched else [])
            if not dests:
                continue
            data = shrink_bytes(src, shrink_to) if shrink_to else src.read_bytes()
            fname = photos[i]["name"]
            if shrink_to:
                fname = f"{Path(fname).stem}.jpg"
            for n in dests:
                entry = f"{safe_name(n)}/{fname}"
                if entry in written:
                    continue
                zf.writestr(entry, data)
                written.add(entry)
            del data
            if progress:
                progress(j / len(items))
    return buf.getvalue()


def list_targets(image_dir: Path) -> list[Path]:
    if not image_dir.is_dir():
        return []
    return [
        p
        for p in sorted(image_dir.rglob("*"))
        if p.suffix.lower() in IMG_EXTS and not p.name.startswith(".")
    ]


LEARNED_FILE = ".learned.npz"   # 手動修正から覚えた顔ベクトル


def learned_path(ref_dir: Path) -> Path:
    return ref_dir / LEARNED_FILE


def load_learned(ref_dir: Path) -> tuple[list[str], np.ndarray, list[str]]:
    """(人物名, 顔ベクトル, 元の写真名) を返す。無ければ空。"""
    f = learned_path(ref_dir)
    if not f.is_file():
        return [], np.zeros((0, 512), dtype=np.float32), []
    try:
        with np.load(f, allow_pickle=False) as z:
            return [str(x) for x in z["owners"]], z["matrix"], [str(x) for x in z["srcs"]]
    except Exception:
        return [], np.zeros((0, 512), dtype=np.float32), []


def add_learned(ref_dir: Path, name: str, emb: np.ndarray, src: str) -> None:
    """修正された顔を、その人の参照データとして追加する。"""
    owners, mat, srcs = load_learned(ref_dir)
    owners.append(name)
    srcs.append(src)
    v = emb.astype(np.float32)[None, :]
    mat = np.vstack([mat, v]) if mat.shape[0] else v
    try:
        np.savez_compressed(learned_path(ref_dir), owners=np.array(owners),
                            matrix=mat, srcs=np.array(srcs))
    except Exception:
        pass


def embed_face(model, img_bgr: np.ndarray, bbox, kps) -> np.ndarray | None:
    """既に検出済みの顔（座標）から特徴量だけを計算する。"""
    try:
        from insightface.app.common import Face

        face = Face(bbox=np.array(bbox, dtype=np.float32),
                    kps=(np.array(kps, dtype=np.float32) if kps else None),
                    det_score=1.0)
        for taskname, m in model.models.items():
            if taskname == "detection":
                continue
            m.get(img_bgr, face)
        return normalize(face.normed_embedding)
    except Exception:
        return None


def ref_signature(ref_dir: Path) -> tuple:
    """参照フォルダの中身を表すキー。画像を追加・差し替えすると値が変わる。"""
    items = []
    try:
        for p in sorted(ref_dir.rglob("*")):
            if p.suffix.lower() in IMG_EXTS and not p.name.startswith("."):
                stt = p.stat()
                items.append((str(p.relative_to(ref_dir)), stt.st_size, int(stt.st_mtime)))
    except Exception:
        pass
    return tuple(items)


@st.cache_resource(show_spinner="参照画像の顔を登録中…（初回と、参照画像を追加したときだけ実行されます）")
def build_reference(sig: tuple, ref_dir_str: str, det_size: int, det_thresh: float):
    """(顔ベクトルの持ち主, 行列, 顔が取れなかった人物) を返す。

    sig が同じ間はキャッシュが使われるため、参照画像を追加・変更したときだけ再計算される。
    """
    model = load_model(det_size, det_thresh)
    people = collect_refs(Path(ref_dir_str))
    owners: list[str] = []
    vecs: list[np.ndarray] = []
    failed: list[str] = []
    for name, files in people.items():
        got = 0
        for f in files:
            img = read_image(f)
            if img is None:
                continue
            faces = model.get(img)
            if not faces:
                continue
            face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            vecs.append(normalize(face.normed_embedding))
            owners.append(name)
            got += 1
        if got == 0:
            failed.append(name)
    matrix = np.stack(vecs) if vecs else np.zeros((0, 512), dtype=np.float32)
    return owners, matrix, failed


# ---------------------------------------------------------------- 画像描画


@st.cache_data(show_spinner=False, max_entries=2000)
def thumbnail(path_str: str, mtime: float, box: int = 200) -> bytes:
    """一覧表示用の小さな JPEG を作る（パスと更新時刻でキャッシュ）。"""
    import io

    from PIL import Image, ImageOps

    with Image.open(path_str) as im:
        im.draft("RGB", (box * 2, box * 2))  # JPEG は縮小デコードで高速化
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((box, box), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=80)
    return buf.getvalue()


def thumb(path: Path, box: int = 200) -> bytes | None:
    try:
        return thumbnail(str(path), path.stat().st_mtime, box)
    except Exception:
        return None


def face_sharpness(img_bgr: np.ndarray, bbox) -> float:
    """顔部分のくっきり具合（ラプラシアン分散）。値が小さいほどボケている。

    顔の大きさに左右されないよう、大きい場合だけ縮小してから測る（拡大はしない）。
    絶対値は撮影条件で変わるため、しきい値はレビュー画面の表示値を見て決める。
    """
    import cv2

    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    ch, cw = g.shape
    if max(ch, cw) > 96:
        r = 96 / max(ch, cw)
        g = cv2.resize(g, (max(8, int(cw * r)), max(8, int(ch * r))),
                       interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


MIN_DETECT_PX = 20   # これ未満の顔は最初から扱わない
CROWD_KEEP = 3       # 顔が多すぎる写真でも、大きく写っているこの人数だけは照合する


def detect_faces(model, img_bgr: np.ndarray, cap: int, focus_ratio: float):
    """顔を検出し、ボケた顔を除いてから特徴量を計算する。

    観客席の顔は背景ボケで写るため、**その写真で最も鮮明な顔に対する比率**で判定する。
    絶対値は写真ごとに変わるが、比率なら「前景の被写体か背景か」を安定して分けられる。
    戻り値: (顔のリスト, 観客席とみなしたか)
    """
    try:
        bboxes, kpss = model.det_model.detect(img_bgr, max_num=0, metric="default")
    except Exception:  # 内部APIが使えない場合は従来どおり
        faces = model.get(img_bgr, max_num=(cap + 1) if cap else 0)
        for fc in faces:
            fc.sharp = face_sharpness(img_bgr, fc.bbox)
        return faces, bool(cap and len(faces) > cap)

    if bboxes is None or len(bboxes) == 0:
        return [], False

    sharps = [face_sharpness(img_bgr, b[:4]) for b in bboxes]
    top = max(sharps) if sharps else 0.0
    keep = []
    for i, b in enumerate(bboxes):
        if min(b[2] - b[0], b[3] - b[1]) < MIN_DETECT_PX:
            continue
        if focus_ratio and top > 0 and sharps[i] < top * focus_ratio:
            continue          # 背景のボケ顔（観客席など）
        keep.append(i)

    crowd = bool(cap) and len(keep) > cap
    if crowd:
        # 観客席のような写真でも、手前の主役だけは取りこぼさないよう
        # 大きく写っている顔だけを残して照合する（残りは無視）
        keep.sort(key=lambda i: -((bboxes[i][2] - bboxes[i][0])
                                  * (bboxes[i][3] - bboxes[i][1])))
        keep = keep[:CROWD_KEEP]

    from insightface.app.common import Face

    faces = []
    for i in keep:
        face = Face(bbox=bboxes[i, 0:4], kps=(kpss[i] if kpss is not None else None),
                    det_score=bboxes[i, 4])
        for taskname, m in model.models.items():
            if taskname == "detection":
                continue
            m.get(img_bgr, face)
        face.sharp = sharps[i]
        faces.append(face)
    faces.sort(key=lambda fc: fc.bbox[0])   # 左から順に並べる
    return faces, crowd


def crop_face(img_bgr: np.ndarray, bbox, margin: float = 0.45) -> np.ndarray:
    import cv2

    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * margin))
    y1 = max(0, int(y1 - bh * margin))
    x2 = min(w, int(x2 + bw * margin))
    y2 = min(h, int(y2 + bh * margin))
    return cv2.cvtColor(img_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)


def draw_boxes(img_bgr: np.ndarray, faces: list[dict], highlight: int | None = None) -> np.ndarray:
    import cv2

    img = img_bgr.copy()
    scale = max(1.0, img.shape[1] / 1200)
    for i, f in enumerate(faces):
        x1, y1, x2, y2 = [int(v) for v in f["bbox"]]
        color = (0, 0, 255) if highlight == i else (0, 200, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, max(2, int(3 * scale)))
        cv2.putText(
            img, str(i + 1), (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
            0.9 * scale, color, max(2, int(2 * scale)), cv2.LINE_AA,
        )
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------- 判定ロジック


def usable_face(f: dict, min_face_px: int, min_sharp: float) -> bool:
    """仕分けに使える顔か（小さすぎず、ボケていない）。"""
    return f["size"] >= min_face_px and (not min_sharp or f.get("sharp", 1e9) >= min_sharp)


def derive_assignment(photo: dict, threshold: float, min_face_px: int, top_only: bool,
                      use_numbers: bool = True, min_sharp: float = 0.0) -> list[str]:
    """保存済み候補と現在のパラメータから、この写真に割り当てる人物名を決める。
    手動確定済み(manual)があればそれを優先。
    顔で誰も判定できなかった場合のみ、背番号 OCR の結果を使う。"""
    manual = photo.get("manual")
    if manual is not None:
        return sorted({n for n in manual if n})

    names: dict[str, float] = {}
    for f in photo["faces"]:
        if not usable_face(f, min_face_px, min_sharp):
            continue
        cands = f["cands"]
        if not cands:
            continue
        picks = cands[:1] if top_only else cands
        for n, s in picks:
            if s >= threshold and s > names.get(n, -1):
                names[n] = s
    if names:
        return sorted(names, key=lambda n: -names[n])

    if use_numbers:
        return list(photo.get("num_names") or [])
    return []


def selections_have_person(selections: list[str | None]) -> bool:
    """顔のプルダウンで「その他」以外の人物が選ばれているか。"""
    return any(s and s != OTHER_NAME for s in selections)


STATE_FILE = ".face_sorting.json"   # 対象フォルダに置く作業状態
STATE_VERSION = 1


def state_path(image_dir: Path) -> Path:
    return image_dir / STATE_FILE


def save_state(image_dir: Path) -> None:
    """解析結果とレビュー内容を対象フォルダに保存する（再起動しても続きから使える）。"""
    photos = st.session_state.get("photos")
    if not photos or not image_dir.is_dir():
        return
    try:
        state_path(image_dir).write_text(
            json.dumps({"version": STATE_VERSION, "photos": photos},
                       ensure_ascii=False),
            encoding="utf-8")
    except Exception:
        pass       # 保存できなくても作業は続けられるので黙って諦める


def load_state(image_dir: Path) -> list[dict] | None:
    """保存済みの作業状態を読み込む。ファイルが消えている写真は除く。"""
    f = state_path(image_dir)
    if not f.is_file():
        return None
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return None
    if data.get("version") != STATE_VERSION:
        return None
    photos = [p for p in data.get("photos", []) if Path(p.get("path", "")).exists()]
    return photos or None


def merge_photos(new_photos: list[dict]) -> None:
    """解析結果を既存の結果に追記する（同じファイルは重複させない）。
    レビューで確定済みの内容は保持される。"""
    existing = st.session_state.get("photos") or []
    seen = {p["path"] for p in existing}
    merged = existing + [p for p in new_photos if p["path"] not in seen]
    merged.sort(key=lambda p: p["name"])
    st.session_state["photos"] = merged


# ---------------------------------------------------------------- マニュアル

README_PATH = APP_DIR / "README.md"

FALLBACK_MANUAL = """## マニュアル

README.md が見つかりませんでした。アプリと同じ場所に README.md を置いてください。

### 基本の流れ
1. **① 写真を追加** — 仕分けしたい写真をアップロード
2. **② 解析** — 「未解析を解析」を押す（参照画像の登録は自動）
3. **③ レビュー** — 自動判定を確認し、違うところだけ直す
4. **④ 出力** — ZIP をダウンロード
"""


@st.cache_data(show_spinner=False)
def load_manual(mtime: float | None) -> str:
    """README.md を読み込む（更新時刻をキーにキャッシュ）。"""
    if README_PATH.exists():
        try:
            return README_PATH.read_text(encoding="utf-8")
        except Exception:
            pass
    return FALLBACK_MANUAL


def manual_text() -> str:
    mt = README_PATH.stat().st_mtime if README_PATH.exists() else None
    return load_manual(mt)


def split_sections(md: str) -> dict[str, str]:
    """'## ' 見出しごとに分割。先頭の導入部は「概要」。"""
    sections: dict[str, str] = {}
    title, buf = "概要", []
    for line in md.splitlines():
        if line.startswith("## "):
            if buf:
                sections[title] = "\n".join(buf).strip()
            title, buf = line[3:].strip(), []
        else:
            buf.append(line)
    if buf:
        sections[title] = "\n".join(buf).strip()
    return {k: v for k, v in sections.items() if v}


def render_manual(container=st, key_prefix: str = "man") -> None:
    md = manual_text()
    sections = split_sections(md)
    names = list(sections.keys())
    mode = container.radio(
        "表示範囲", ["全文"] + names, horizontal=True,
        key=f"{key_prefix}_sec", label_visibility="collapsed",
    )
    container.divider()
    if mode == "全文":
        container.markdown(md)
    else:
        container.markdown(f"## {mode}\n\n{sections[mode]}")
    container.divider()
    container.download_button(
        "📄 README.md をダウンロード", md.encode("utf-8"),
        file_name="README.md", mime="text/markdown", key=f"{key_prefix}_dl",
    )


@st.dialog("📖 マニュアル", width="large")
def manual_dialog():
    render_manual(st, "dlg")


# ---------------------------------------------------------------- サイドバー

st.title("顔認証 写真仕分け for v-varen NAGASAKI")

ss = st.session_state
ss.setdefault("ref_failed", None)
ss.setdefault("photos", None)
ss.setdefault("review_idx", 0)
ss.setdefault("job", None)
ss.setdefault("open_pi", None)

_missing = check_deps()
if _missing:
    st.error("必要なパッケージが不足しています: " + ", ".join(_missing)
             + "　（requirements.txt を確認してください）")
    st.stop()

# セッションごとの作業場所（ブラウザを閉じると破棄されます）
for key, prefix in (("workdir", "photos_"), ("ref_extra_dir", "refadd_"),
                    ("learn_dir", "learn_")):
    if key not in ss:
        ss[key] = tempfile.mkdtemp(prefix=prefix)
WORKDIR = Path(ss["workdir"])          # アップロードされた写真
REF_EXTRA = Path(ss["ref_extra_dir"])  # このセッションで追加された参照画像
LEARN_DIR = Path(ss["learn_dir"])      # このセッションで覚えた顔

with st.sidebar:
    if st.button("📖 マニュアルを開く", use_container_width=True):
        manual_dialog()
    st.divider()

    st.header("認証パラメータ")
    threshold = st.slider(
        "類似度しきい値", 0.10, 0.90, 0.40, 0.01,
        help="コサイン類似度。高いほど厳しく（誤検出↓ / 見逃し↑）。解析後でも変更すれば即座に反映されます。",
    )
    min_face_px = st.slider("最小顔サイズ (px)", 0, 200, 40, 5, help="これより小さい顔は無視")
    det_size = st.select_slider(
        "検出解像度", [320, 480, 640, 800, 1024], value=640,
        help="大きいほど小さな顔を検出できるが遅くなる（変更後は再解析が必要）",
    )
    det_thresh = st.slider(
        "顔検出のしきい値", 0.10, 0.90, 0.50, 0.05,
        help="下げると顔として検出されにくいもの（横顔・マスコット・着ぐるみなど）も"
             "拾えますが、誤検出も増えます。変更後は再解析が必要です。",
    )
    focus_pct = st.slider(
        "ボケ顔とみなす比率 (%)", 0, 60, 15, 5,
        help="その写真で最も鮮明な顔に対して、この割合より不鮮明な顔は"
             "「背景のボケ顔」とみなして無視します（観客席の除外にも使われます）。"
             "0 で無効。変更後は再解析が必要です。",
    )
    focus_ratio = focus_pct / 100.0
    min_sharp = st.slider(
        "ボケ顔を除外（絶対値 / 0 = 使わない）", 0.0, 200.0, 0.0, 5.0,
        help="上の比率に加えて、くっきり具合の絶対値でも除外したい場合に使います。"
             "レビュー画面に顔ごとの実測値が出るので、それを見て決めてください。"
             "解析後に変更しても再解析は不要です。",
    )
    max_faces = st.number_input(
        "1枚あたりの最大顔数", 0, 200, 20, step=5,
        help="これを超える顔が写っている写真は観客席とみなし、照合をスキップします"
             f"（{OTHER_NAME}/ 行き）。0 で無制限。変更後は再解析が必要です。",
    )
    top_only = st.checkbox("1顔につき最上位の1人だけ採用", value=True)

    st.header("背番号での判定")
    use_numbers = st.checkbox(
        "背番号も読み取る", value=True,
        help="背中向きなど顔が写っていない選手を、ユニフォームの背番号から判定します。"
             "顔で誰も判定できなかった写真にだけ適用されます。変更後は再解析が必要です。",
    )
    num_min_ratio = st.slider(
        "背番号の最小の大きさ（画像高さ比 %）", 1.0, 15.0, 5.0, 0.5, disabled=not use_numbers,
        help="これより小さい数字は看板や観客席とみなして無視します。"
             "遠くの選手まで拾いたい場合は下げてください（誤検出は増えます）。",
    )

    st.header("追加学習")
    use_learning = st.checkbox(
        "修正した顔を覚える", value=True,
        help="レビューで人物を選び直した顔を、その人の参照データに追加します。"
             "横顔・斜め後ろなど、証明写真では判定できない角度に強くなります。"
             "自動判定のままの顔は学習しません（誤りを覚えないため）。",
    )
    _base_n = len(load_learned(REF_DIR)[0])
    _sess_n = len(load_learned(LEARN_DIR)[0])
    st.caption(f"同梱の学習データ {_base_n} 件 ＋ このセッション {_sess_n} 件。"
               "セッション分はブラウザを閉じると消えます"
               "（「👥 人物一覧」タブから書き出して次回持ち込めます）。")

    st.header("出力設定")
    save_unmatched = st.checkbox(f"該当なしの写真も {OTHER_NAME}/ に入れる", value=True,
                                 help="誰にも該当しなかった写真を、一覧にない人と同じフォルダにまとめます。")

    st.divider()
    st.caption("アップロードした写真はブラウザを閉じると自動的に破棄されます。")

# ---------------------------------------------------------------- 参照と対象

people = collect_all_refs(REF_DIR, REF_EXTRA)
extra_names = sorted({p.stem for p in REF_EXTRA.rglob("*")
                      if p.suffix.lower() in IMG_EXTS})
all_names = sorted(people.keys())
targets = sorted(p for p in WORKDIR.iterdir() if p.suffix.lower() in IMG_EXTS)

if not people:
    st.error("参照画像が見つかりません（ref/ が空です）。")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("登録人物数", len(people))
c2.metric("参照画像数", sum(len(v) for v in people.values()))
c3.metric("アップロード済み", len(targets))
c4.metric("解析済み", len(ss["photos"]) if ss["photos"] else 0)

(tab_upload, tab_run, tab_review, tab_apply, tab_result,
 tab_people, tab_manual) = st.tabs(
    ["① 写真を追加", "② 解析", "③ レビュー", "④ 出力", "⑤ 結果",
     "👥 人物一覧", "📖 マニュアル"]
)

# ---------------------------------------------------------------- ① 写真を追加

with tab_upload:
    st.info("**アップロードされた写真はサーバー上で一時的に処理され、"
            "セッション終了後に削除されます。**\n\n"
            "顔画像は個人情報にあたります。ご本人の同意が得られている写真のみをご利用ください。",
            icon="🔒")
    ups = st.file_uploader(
        f"仕分けする写真を選択（複数可 / 最大 {MAX_FILES} 枚）",
        type=UPLOAD_TYPES, accept_multiple_files=True, key="photo_uploader",
    )
    if ups:
        if len(ups) > MAX_FILES:
            st.warning(f"{MAX_FILES} 枚を超えた分は無視されます。")
            ups = ups[:MAX_FILES]
        saved = 0
        for uf in ups:
            dest = WORKDIR / Path(uf.name).name
            if not dest.exists() or dest.stat().st_size != uf.size:
                dest.write_bytes(uf.getbuffer())
                saved += 1
        st.success(f"{len(ups)} 枚を受け付けました（新規 {saved} 枚）。"
                   "「② 解析」へ進んでください。")

    if targets and st.button("アップロードした写真をすべて削除"):
        for f in targets:
            f.unlink(missing_ok=True)
        ss["photos"] = None
        ss["job"] = None
        ss["zip_bytes"] = None
        st.rerun()

# ---------------------------------------------------------------- 人物一覧タブ

with tab_people:
    with st.expander("➕ 参照画像を追加する（このセッション限り）"):
        st.markdown(NAMING_RULE)
        dest_kind = st.radio(
            "保存先",
            ["選手など", "スタッフ", "その他の関係者"],
            horizontal=True, key="ref_up_kind",
        )
        new_refs = st.file_uploader(
            "参照画像を選択（複数可）", type=UPLOAD_TYPES, accept_multiple_files=True,
            key="ref_uploader",
        )
        if new_refs:
            preview = [f"`{Path(f.name).stem}`" for f in new_refs[:8]]
            st.caption("登録される人物名: " + "、".join(preview)
                       + ("…" if len(new_refs) > 8 else ""))
            odd = [f.name for f in new_refs if "_" not in Path(f.name).stem]
            if odd:
                st.warning("「_」が含まれないファイル名があります（そのままの名前で登録されます）: "
                           + ", ".join(odd[:5]))

        if st.button("参照画像を追加", type="primary", disabled=not new_refs):
            sub = "staff" if "スタッフ" in dest_kind else ("etc" if "その他" in dest_kind else "")
            dest_dir = REF_EXTRA / sub if sub else REF_EXTRA
            dest_dir.mkdir(parents=True, exist_ok=True)
            added, replaced = [], []
            for uf in new_refs:
                dest = dest_dir / Path(uf.name).name
                (replaced if dest.exists() else added).append(dest.stem)
                dest.write_bytes(uf.getbuffer())
            msg = f"{len(added)} 件を追加しました。" if added else ""
            if replaced:
                msg += f"{len(replaced)} 件を上書きしました（{', '.join(replaced[:3])}…）。"
            st.success(msg + "次回の解析時に自動で登録されます（このセッション限り）。")
            st.rerun()

    # ---- 学習データ（同梱ぶん＋このセッションぶん）----
    b_owners, _bm, _bs = load_learned(REF_DIR)
    s_owners, s_matrix, s_srcs = load_learned(LEARN_DIR)
    with st.expander(f"🧠 覚えた顔: 同梱 {len(b_owners)} 件 ＋ このセッション {len(s_owners)} 件"):
        st.caption("レビューで人物を選び直した顔です。次の解析から照合に使われます。"
                   "**このセッションぶんはブラウザを閉じると消えます。**"
                   "書き出して保存しておけば、次回アップロードして引き継げます。")
        if s_owners:
            cnt: dict[str, int] = defaultdict(int)
            for n in s_owners:
                cnt[n] += 1
            st.markdown("\n".join(f"- {short_label(n)}: {c} 件" for n, c in
                                  sorted(cnt.items(), key=lambda x: -x[1])))
        c1, c2, c3 = st.columns(3)
        if s_owners and learned_path(LEARN_DIR).is_file():
            c1.download_button("💾 学習データを書き出す",
                               learned_path(LEARN_DIR).read_bytes(),
                               file_name=LEARNED_FILE.lstrip("."),
                               mime="application/octet-stream")
        if s_owners and c2.button("このセッションの学習を削除"):
            learned_path(LEARN_DIR).unlink(missing_ok=True)
            st.rerun()
        up_learn = c3.file_uploader("前回の学習データを読み込む", type=["npz"],
                                    key="learn_upload", label_visibility="collapsed")
        if up_learn is not None:
            learned_path(LEARN_DIR).write_bytes(up_learn.getbuffer())
            st.success("学習データを読み込みました。")
            st.rerun()
        st.caption("恒久的に反映するには、書き出したファイルを `ref/.learned.npz` として"
                   "リポジトリにコミットしてください（全利用者に反映されます）。")

    groups = group_by_position(all_names)
    ref_failed = ss.get("ref_failed")
    embedded = set(all_names) - set(ref_failed or [])

    view = st.radio("表示", ["リスト", "サムネイル"], horizontal=True, label_visibility="collapsed")

    if view == "リスト":
        cols = st.columns(len(groups) or 1)
        for col, (pos, members) in zip(cols, groups.items()):
            with col:
                st.markdown(f"##### {pos} ({len(members)})")
                lines = []
                for num, person, raw in members:
                    mark = "" if ref_failed is None else ("✅ " if raw in embedded else "⚠️ ")
                    lines.append(f"{mark}`{num or '—'}` {person}")
                st.markdown("\n".join(f"- {l}" for l in lines))
        if ref_failed is not None:
            st.caption("✅ 顔ベクトル登録済み / ⚠️ 参照画像から顔を検出できず")
    else:
        for pos, members in groups.items():
            st.markdown(f"##### {pos} ({len(members)})")
            per_row = 6
            for i in range(0, len(members), per_row):
                cols = st.columns(per_row)
                for col, (label, person, raw) in zip(cols, members[i : i + per_row]):
                    with col:
                        st.image(str(people[raw][0]), use_container_width=True)
                        # 1行目に背番号または役職、2行目に氏名
                        st.caption(f"**{label}**  \n{person}" if label else person)

# ---------------------------------------------------------------- 解析タブ

with tab_run:
    st.caption("この段階ではファイルは書き込みません。各写真の顔と候補人物を記録します。"
               "ヴィヴィくんは人では無いので顔認証で利用されません。"
               "何回かに分けて解析でき、結果は積み上がります。"
               "参照画像の登録は初回の解析時に自動で行われます"
               "（参照画像を追加・変更したときだけ自動で登録し直します）。")

    if not targets:
        st.info("先に「① 写真を追加」で写真をアップロードしてください。")
    analyzed = {p["path"] for p in (ss["photos"] or [])}
    pending = [p for p in targets if str(p) not in analyzed]

    s1, s2, s3 = st.columns(3)
    s1.metric("対象写真", len(targets))
    s2.metric("解析済み", len(analyzed))
    s3.metric("未解析", len(pending))

    batch = st.number_input(
        "1回に解析する枚数（0 = 未解析をすべて）", 0, 100000, 50, step=10,
        help="少しずつ解析して、途中でレビューや出力を挟めます。続きはいつでも再開できます。",
    )

    job = ss.get("job")
    status = job["status"] if job else "idle"
    active = status in ("running", "paused")

    bc = st.columns(4)
    if bc[0].button(f"未解析を解析（{len(pending)}枚）", type="primary",
                    disabled=active or not pending):
        todo = pending[: int(batch)] if batch else pending
        ss["job"] = {"status": "running", "todo": [str(p) for p in todo],
                     "cursor": 0, "photos": [], "elapsed": 0.0}
        st.rerun()

    if status == "running" and bc[1].button("⏸ 一時停止", use_container_width=True):
        job["status"] = "paused"
        st.rerun()
    if status == "paused" and bc[1].button("▶ 再開", type="primary", use_container_width=True):
        job["status"] = "running"
        st.rerun()
    if active and bc[2].button("⏹ 中止", use_container_width=True):
        job["status"] = "stopped"
        st.rerun()
    if status in ("stopped", "done") and bc[3].button("解析結果をすべて破棄",
                                                      use_container_width=True):
        ss["job"] = None
        ss["photos"] = None
        ss["review_idx"] = 0
        state_path(WORKDIR).unlink(missing_ok=True)
        st.rerun()

    # --- 進捗表示 ---
    if job:
        total = len(job["todo"])
        cur = job["cursor"]
        st.progress(cur / total if total else 1.0, f"この回: {cur} / {total} 枚")
        speed = (job["elapsed"] / cur) if cur else 0
        remain = speed * (total - cur)
        info = f"経過 {job['elapsed']:.0f}秒"
        if cur:
            info += f"（1枚あたり {speed:.1f}秒）"
        if status == "running" and cur:
            info += f" ／ この回の残り約 {remain:.0f}秒"
        st.caption(info)

        left_all = len(pending) - (cur if active else 0)
        if status == "paused":
            st.info("一時停止中です。「▶ 再開」でこの回の続きから処理します。")
        elif status == "stopped":
            st.warning(f"中止しました（この回 {cur} / {total} 枚）。解析できた分はそのまま使えます。"
                       + (f"未解析があと {len(pending)} 枚あります。"
                          "「未解析を解析」でいつでも続きから再開できます。" if pending else ""))
        elif status == "done":
            msg = f"この回の解析完了：{total} 枚。"
            msg += (f"未解析があと {len(pending)} 枚あります。"
                    if pending else "すべての写真を解析しました。")
            st.success(msg)

    if ss.get("ref_failed"):
        st.warning("参照画像から顔を検出できなかった人物: "
                   + "、".join(short_label(n) for n in ss["ref_failed"])
                   + "（この人は顔では判定されません）")

    # --- チャンク処理（一定時間ごとに画面へ戻り、停止操作を受け付ける） ---
    if status == "running":
        model = get_model(det_size, det_thresh)
        ref_owner, ref_matrix, ref_failed = build_reference(
            ref_signature(REF_DIR), str(REF_DIR), det_size, det_thresh)
        if extra_names:
            e_owner, e_matrix, e_failed = build_reference(
                ref_signature(REF_EXTRA), str(REF_EXTRA), det_size, det_thresh)
            if e_matrix.shape[0]:
                ref_owner = list(ref_owner) + list(e_owner)
                ref_matrix = np.concatenate([ref_matrix, e_matrix], axis=0)
            ref_failed = list(ref_failed) + list(e_failed)
        ss["ref_failed"] = ref_failed
        # 覚えた顔も参照に加える（リポジトリ同梱ぶん＋このセッションぶん）
        for _dir in (REF_DIR, LEARN_DIR):
            l_owners, l_matrix, _ = load_learned(_dir)
            if l_matrix.shape[0]:
                ref_owner = list(ref_owner) + l_owners
                ref_matrix = np.concatenate([ref_matrix, l_matrix], axis=0)
        if ref_matrix.shape[0] == 0:
            ss["job"] = None
            st.error("参照画像から顔を検出できませんでした。")
            st.stop()
        ocr, ocr_err = (load_ocr() if use_numbers else (None, None))
        num_map = build_number_map(all_names)
        if use_numbers and ocr is None and job["cursor"] == 0:
            st.warning(f"背番号の読み取りを使えませんでした（{ocr_err}）。"
                       "今回は顔認証のみで解析します。"
                       "`requirements.txt` に `rapidocr_onnxruntime` があるか確認してください。")

        t0 = time.time()
        while job["cursor"] < len(job["todo"]) and time.time() - t0 < CHUNK_SECONDS:
            path = Path(job["todo"][job["cursor"]])
            img = read_image(path)
            if img is None:
                job["photos"].append(
                    {"path": str(path), "name": path.name, "faces": [], "error": "読込失敗"})
                job["cursor"] += 1
                continue

            cap = int(max_faces)
            detected, is_crowd = detect_faces(model, img, cap, focus_ratio)

            faces_meta = []
            for f in detected:
                w = float(f.bbox[2] - f.bbox[0])
                h = float(f.bbox[3] - f.bbox[1])
                sims = ref_matrix @ normalize(f.normed_embedding)
                best: dict[str, float] = {}
                for owner, s in zip(ref_owner, sims):
                    if s > best.get(owner, -1):
                        best[owner] = float(s)
                cands = sorted(best.items(), key=lambda x: -x[1])[:TOP_K]
                faces_meta.append({
                    "bbox": [float(v) for v in f.bbox],
                    "kps": ([[float(x), float(y)] for x, y in f.kps]
                            if getattr(f, "kps", None) is not None else None),
                    "size": int(min(w, h)),
                    "sharp": round(getattr(f, "sharp", 0.0), 1),
                    "cands": [(n, round(s, 4)) for n, s in cands],
                })
            rec = {"path": str(path), "name": path.name, "faces": faces_meta}
            if is_crowd:
                rec["crowd"] = True

            # 背番号の読み取り。顔で誰も判定できなかった写真だけに行う（速度のため）
            matched = any(m["size"] >= min_face_px and m["cands"]
                          and m["cands"][0][1] >= threshold for m in faces_meta)
            if ocr is not None and not matched:
                nums = read_numbers(ocr, path, float(num_min_ratio), set(num_map))
                if nums:
                    rec["numbers"] = [(n, round(c, 3), r) for n, c, r in nums]
                    rec["num_names"] = sorted({p for n, _, _ in nums
                                               for p in num_map.get(n, [])})

            job["photos"].append(rec)
            job["cursor"] += 1

        job["elapsed"] += time.time() - t0
        merge_photos(job["photos"])  # 途中結果も随時反映（中断しても失われない）
        save_state(WORKDIR)
        if job["cursor"] >= len(job["todo"]):
            job["status"] = "done"
        st.rerun()

# ---------------------------------------------------------------- レビュータブ

with tab_review:
    if not ss["photos"]:
        st.info("先に「② 解析」を実行してください。")
    else:
        photos = ss["photos"]

        scope = st.radio(
            "レビュー対象",
            ["複数人の写真", "すべての写真", "要確認（該当なし・低確度）", "未確認のみ"],
            horizontal=True, key="rev_scope",
            help="取りこぼしを防ぐには「未確認のみ」を選び、空になるまで進めてください。",
        )

        def _needs_check(p: dict) -> bool:
            if not derive_assignment(p, threshold, min_face_px, top_only, min_sharp=min_sharp):
                return True
            best = max((c[1] for f in p["faces"] for c in f["cands"][:1]), default=0.0)
            return best < threshold + 0.05

        if scope == "複数人の写真":
            queue = [i for i, p in enumerate(photos) if len(p["faces"]) >= 2]
        elif scope == "すべての写真":
            queue = list(range(len(photos)))
        elif scope == "未確認のみ":
            queue = [i for i, p in enumerate(photos) if not p.get("seen")]
        else:
            queue = [i for i, p in enumerate(photos) if _needs_check(p)]

        n_fixed = sum(1 for p in photos if p.get("manual") is not None)
        n_seen = sum(1 for p in photos if p.get("seen") and p.get("manual") is None)
        n_todo = len(photos) - n_fixed - n_seen
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("解析済み", len(photos))
        m2.metric("修正済み", n_fixed, help="レビューで内容を変えた写真")
        m3.metric("確認済み", n_seen, help="レビュー画面で開いたが、自動判定のままにした写真")
        m4.metric("未確認", n_todo, help="まだレビュー画面で開いていない写真")
        st.caption(f"この条件の対象: {len(queue)} 枚。"
                   "自動判定が初期値に入っているので、誤りだけ直してください。")

        if not queue:
            st.success("この条件に該当する写真はありません。")
        else:
            ss["review_idx"] = min(ss["review_idx"], len(queue) - 1)
            def _mark(p: dict) -> str:
                if p.get("manual") is not None:
                    return "✔ "        # 修正済み
                return "・" if p.get("seen") else ""   # 確認済み / 未確認
            labels = [f"{_mark(photos[i])}{photos[i]['name']}"
                      f"（顔{len(photos[i]['faces'])}）" for i in queue]
            pick = st.selectbox("写真を選択", range(len(queue)),
                                format_func=lambda k: labels[k], index=ss["review_idx"])
            ss["review_idx"] = pick
            pi = queue[pick]
            photo = photos[pi]
            path = Path(photo["path"])

            # --- 別の写真へ移ったら、直前の写真の変更を自動保存する ---
            def widget_state(idx: int):
                """session_state に残っているその写真の選択内容を取り出す。"""
                if f"extra_{idx}" not in st.session_state:
                    return None
                sel: list[str | None] = []
                for k in range(len(photos[idx]["faces"])):
                    rk, uk = f"rev_{idx}_{k}", f"use_{idx}_{k}"
                    if rk not in st.session_state:
                        return None
                    sel.append(st.session_state[rk]
                               if st.session_state.get(uk) else None)
                return sel, list(st.session_state[f"extra_{idx}"])

            def learn_corrections(idx: int, sel: list) -> None:
                """人物を選び直した顔だけを覚える。自動判定のままの顔は覚えない。"""
                photo_ = photos[idx]
                todo = []
                for k, name in enumerate(sel):
                    if not name or name == OTHER_NAME or k >= len(photo_["faces"]):
                        continue
                    f = photo_["faces"][k]
                    auto = (f["cands"][0][0]
                            if f["cands"] and f["cands"][0][1] >= threshold else None)
                    if name == auto:
                        continue                      # 修正ではないので覚えない
                    if f.get("learned") == name:
                        continue                      # 覚え済み
                    todo.append((k, f, name))
                if not todo:
                    return
                img_ = read_image(Path(photo_["path"]))
                if img_ is None:
                    return
                model_ = get_model(det_size, det_thresh)
                done = []
                for k, f, name in todo:
                    emb = embed_face(model_, img_, f["bbox"], f.get("kps"))
                    if emb is None:
                        continue
                    add_learned(LEARN_DIR, name, emb, photo_["name"])
                    f["learned"] = name
                    done.append(short_label(name))
                if done:
                    save_state(WORKDIR)
                    st.toast("覚えました: " + "、".join(done))

            def autosave(idx: int) -> None:
                """自動判定と違う内容になっていれば保存する（押し忘れ防止）。"""
                state = widget_state(idx)
                if state is None:
                    return
                sel, ext = state
                picked = sorted({n for n in sel if n} | set(ext))
                auto = derive_assignment({"faces": photos[idx]["faces"]}, threshold,
                                         min_face_px, top_only, min_sharp=min_sharp)
                # 自動判定は類似度順、こちらは名前順なので集合で比べる
                if set(picked) != set(auto) or photos[idx].get("manual") is not None:
                    changed = photos[idx].get("manual") != picked
                    photos[idx]["manual_faces"] = sel
                    photos[idx]["manual_extra"] = ext
                    photos[idx]["manual"] = picked
                    if changed:
                        save_state(WORKDIR)
                    if use_learning:
                        learn_corrections(idx, sel)

            # 直前まで開いていた写真と、いま開いている写真の両方を保存する。
            # いま開いている写真も毎回保存しないと、レビュー後そのまま
            # 「③ 出力」タブへ移動したときに変更が反映されない。
            if ss.get("open_pi") is not None and ss["open_pi"] != pi:
                autosave(ss["open_pi"])
            autosave(pi)
            if not photo.get("seen"):
                photo["seen"] = True          # レビュー画面で開いた印
                save_state(WORKDIR)
            ss["open_pi"] = pi

            nav1, nav2, _ = st.columns([1, 1, 6])
            if nav1.button("← 前へ", disabled=pick == 0):
                ss["review_idx"] = pick - 1
                st.rerun()
            if nav2.button("次へ →", disabled=pick >= len(queue) - 1):
                ss["review_idx"] = pick + 1
                st.rerun()

            img = read_image(path)
            if photo.get("crowd"):
                st.warning(f"顔が {max_faces} 個より多いため、**大きく写っている {CROWD_KEEP} 人だけ**"
                           "照合しました（観客席とみなして残りは無視）。"
                           "足りない人は「名前で追加する」から選べます。")

            left, right = st.columns([3, 5])
            with left:
                if img is not None:
                    st.image(draw_boxes(img, photo["faces"]), caption=photo["name"],
                             use_container_width=True)

            with right:
                prev_faces = photo.get("manual_faces")

                # --- 顔ごとの初期値をあらかじめ決めておく（表示は下段だが値は先に必要） ---
                face_setup = []
                for k, f in enumerate(photo["faces"]):
                    score_map = {n: s for n, s in f["cands"]}
                    options = (list(score_map)
                               + [n for n in all_names if n not in score_map]
                               + [OTHER_NAME])
                    ok = usable_face(f, min_face_px, min_sharp)
                    confident = bool(ok and f["cands"] and f["cands"][0][1] >= threshold)
                    auto_name = f["cands"][0][0] if confident else OTHER_NAME
                    if prev_faces is not None and k < len(prev_faces):
                        default = prev_faces[k] or auto_name
                        default_on = bool(prev_faces[k])
                    else:
                        default = auto_name
                        default_on = confident   # 確度が高い顔は自動でチェック
                    face_setup.append((score_map, options, default, default_on, ok))

                # いま選ばれている状態（前回描画時の値）。画面上の並びを入れ替えるために使う
                selections: list[str | None] = []
                for k, (_, options, default, default_on, _) in enumerate(face_setup):
                    name = st.session_state.get(f"rev_{pi}_{k}", default)
                    if name not in options:
                        name = default
                    selections.append(
                        name if st.session_state.get(f"use_{pi}_{k}", default_on) else None)

                # --- 背番号 ---
                num_names = list(photo.get("num_names") or [])
                if photo.get("numbers"):
                    detail = "、".join(f"**{n}番**（信頼度 {c:.2f}）"
                                      for n, c, _ in photo["numbers"])
                    hit = "、".join(short_label(n) for n in num_names) or "該当する選手なし"
                    st.info(f"🔢 背番号を検出: {detail}　→　{hit}")

                # --- 名前で追加する ---
                extra_options = all_names + [OTHER_NAME]
                extra_default = [n for n in (photo.get("manual_extra") or [])
                                 if n in extra_options]
                if photo.get("manual_extra") is None and not selections_have_person(selections):
                    extra_default = [n for n in num_names if n in extra_options]

                extra = st.multiselect(
                    "名前で追加する（背中向き・顔が検出されなかった人など）",
                    extra_options, default=extra_default,
                    format_func=short_label, key=f"extra_{pi}",
                )

                # --- 仕分け先 ---
                assigned = sorted({n for n in selections if n} | set(extra))
                st.info("仕分け先: " + ("、".join(short_label(n) for n in assigned)
                                        or f"{OTHER_NAME}（該当なし）"))
                st.caption("変更は別の写真へ移ったときに自動保存されます"
                           "（「確定して次へ」を押さなくても反映されます）。")

                st.divider()

                # --- 検出された顔ごとの指定 ---
                st.markdown("##### 検出された顔と人物の割り当て")
                st.caption("顔ごとに自動認識した人物を表示しています。**仕分けたい顔にチェック**を入れてください。"
                           "人物が違うときは下のプルダウンで選び直せます。"
                           "変更内容は上の「仕分け先」に反映されます。")

                if photo["faces"]:
                    PER_ROW = 4
                    for row0 in range(0, len(photo["faces"]), PER_ROW):
                        cols = st.columns(PER_ROW)
                        for col, k in zip(cols, range(row0, min(row0 + PER_ROW,
                                                                len(photo["faces"])))):
                            f = photo["faces"][k]
                            score_map, options, default, default_on, ok = face_setup[k]
                            sel_key = f"rev_{pi}_{k}"
                            current = st.session_state.get(sel_key, default)
                            if current not in options:
                                current = default

                            with col:
                                if img is not None:
                                    st.image(crop_face(img, f["bbox"]),
                                             use_container_width=True)
                                st.checkbox(
                                    short_label(current)
                                    + (f"　{score_map[current]:.2f}" if current in score_map else ""),
                                    value=default_on, key=f"use_{pi}_{k}",
                                )
                                st.selectbox(
                                    f"顔 {k + 1}", options,
                                    index=options.index(current),
                                    key=sel_key, label_visibility="collapsed",
                                    format_func=lambda n: (
                                        f"{short_label(n)}　({score_map[n]:.3f})"
                                        if n in score_map else short_label(n)),
                                )
                                note = f"顔{k + 1} / {f['size']}px"
                                if "sharp" in f:
                                    note += f" / ボケ度 {f['sharp']:.0f}"
                                if not ok:
                                    note += "（対象外）"
                                st.caption(note)
                else:
                    st.caption("顔は検出されませんでした。")

                b1, b2, b3 = st.columns(3)
                if b1.button("確定して次へ", type="primary", key=f"ok_{pi}"):
                    photo["manual_faces"] = selections
                    photo["manual_extra"] = list(extra)
                    photo["manual"] = assigned
                    if pick < len(queue) - 1:
                        ss["review_idx"] = pick + 1
                    st.rerun()
                if b2.button("自動判定のまま確定", key=f"auto_{pi}"):
                    photo.pop("manual_faces", None)
                    photo.pop("manual_extra", None)
                    photo["manual"] = derive_assignment(
                        {"faces": photo["faces"]}, threshold, min_face_px, top_only,
                        min_sharp=min_sharp)
                    if pick < len(queue) - 1:
                        ss["review_idx"] = pick + 1
                    st.rerun()
                if b3.button("確定を取り消す", key=f"rst_{pi}",
                             disabled=photo.get("manual") is None):
                    for key in ("manual", "manual_faces", "manual_extra"):
                        photo.pop(key, None)
                    st.rerun()

# ---------------------------------------------------------------- ④ 出力（ZIP）

with tab_apply:
    if not ss["photos"]:
        st.info("先に「② 解析」を実行してください。")
    else:
        photos = ss["photos"]
        plan = {i: derive_assignment(p, threshold, min_face_px, top_only, min_sharp=min_sharp)
                for i, p in enumerate(photos)}
        counter: dict[str, int] = defaultdict(int)
        for names in plan.values():
            for n in names:
                counter[n] += 1
        n_unmatched = sum(1 for v in plan.values() if not v)

        m1, m2, m3 = st.columns(3)
        m1.metric("仕分けされる件数", sum(counter.values()))
        m2.metric("フォルダ数", len(counter))
        m3.metric("該当なしの写真", n_unmatched)
        st.caption("しきい値などを変えると、再解析なしでこの内容が更新されます。"
                   "レビューで確定した写真は手動の内容が優先されます。")

        est_mb = estimate_zip_size(plan, photos, save_unmatched) / 1024 / 1024
        size_mode = st.radio(
            "ZIP に入れる画像",
            [f"オリジナル画質（約 {est_mb:.0f} MB）", "長辺1600pxに縮小（軽量）"],
            index=0 if est_mb <= ZIP_WARN_MB else 1,
            help="複数人が写った写真は人数分コピーされるため、合計は元より大きくなります。",
        )
        shrink_to = None if size_mode.startswith("オリジナル") else 1600

        too_big = est_mb > ZIP_LIMIT_MB and shrink_to is None
        if too_big:
            st.error(f"ZIP が約 {est_mb:.0f} MB と大きすぎます（上限 {ZIP_LIMIT_MB} MB）。"
                     "「長辺1600pxに縮小」を選ぶか、写真を分けて処理してください。")
        elif est_mb > ZIP_WARN_MB and shrink_to is None:
            st.warning(f"ZIP が約 {est_mb:.0f} MB になります。作成に時間がかかります。")

        d1, d2 = st.columns(2)
        with d1:
            if st.button("ZIP を作成", type="primary", disabled=too_big):
                bar = st.progress(0.0, "ZIP を作成中…")
                ss["zip_bytes"] = build_zip(
                    plan, photos, save_unmatched, shrink_to,
                    progress=lambda r: bar.progress(min(r, 1.0), "ZIP を作成中…"))
                bar.empty()
            if ss.get("zip_bytes"):
                mb = len(ss["zip_bytes"]) / 1024 / 1024
                st.download_button(f"📦 仕分け済みZIPをダウンロード ({mb:.1f} MB)",
                                   ss["zip_bytes"], file_name="sorted_photos.zip",
                                   mime="application/zip", type="primary")

# ---------------------------------------------------------------- 結果タブ

with tab_result:
    if not ss["photos"]:
        st.info("先に「② 解析」を実行してください。")
    else:
        import pandas as pd

        photos = ss["photos"]
        plan = {i: derive_assignment(p, threshold, min_face_px, top_only, min_sharp=min_sharp)
                for i, p in enumerate(photos)}
        counter: dict[str, int] = defaultdict(int)
        for names in plan.values():
            for n in names:
                counter[n] += 1

        st.subheader("人物ごとの枚数")
        if counter:
            rows = []
            for name, cnt in counter.items():
                pos, num, person = parse_person(name)
                rows.append({"ポジション": pos, "番号": num, "選手": person, "枚数": cnt})
            df_c = pd.DataFrame(rows).sort_values("枚数", ascending=False)
            cc1, cc2 = st.columns([1, 1])
            cc1.dataframe(df_c, use_container_width=True, hide_index=True, height=400)
            cc2.bar_chart(df_c.set_index("選手")["枚数"])

        st.subheader("写真ごとの判定")
        rows = []
        for i, p in enumerate(photos):
            best = max((c[1] for f in p["faces"] for c in f["cands"][:1]), default=0.0)
            rows.append(
                {
                    "ファイル": p["name"],
                    "顔数": len(p["faces"]),
                    "確定": ("手動" if p.get("manual") is not None
                             else ("観客席（一部）" if p.get("crowd") else "自動")),
                    "判定人物": ", ".join(short_label(n) for n in plan[i]) or "—",
                    "最高類似度": round(best, 3),
                    "背番号": ", ".join(n for n, _, _ in (p.get("numbers") or [])) or "—",
                }
            )
        df = pd.DataFrame(rows)
        f1, f2 = st.columns(2)
        only_un = f1.checkbox("該当なしのみ表示")
        only_multi = f2.checkbox("複数顔のみ表示")
        view_df = df
        if only_un:
            view_df = view_df[view_df["判定人物"] == "—"]
        if only_multi:
            view_df = view_df[view_df["顔数"] >= 2]
        st.dataframe(view_df, use_container_width=True, hide_index=True, height=420)
        st.download_button(
            "結果をCSVでダウンロード",
            df.to_csv(index=False).encode("utf-8-sig"),
            file_name="face_sorting_result.csv",
            mime="text/csv",
        )

# ---------------------------------------------------------------- マニュアルタブ

with tab_manual:
    st.caption(f"参照元: {README_PATH}　（ファイルを編集すると自動で反映されます）")
    render_manual(st, "tab")
