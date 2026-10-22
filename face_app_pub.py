"""
顔認証による写真仕分け Streamlit アプリ
======================================
- ref/  : 参照画像（ファイル名 = 人物名）。サブフォルダ形式にも対応。
- image/: 仕分け対象の写真
- output/<人物名>/ に該当写真をコピー

フロー: ① 参照登録 → ② 全件解析 → ③ 複数顔写真をレビュー → ④ ファイル出力
エンジン: InsightFace (buffalo_l)
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st

# ---------------------------------------------------------------- 定数

APP_DIR = Path(__file__).resolve().parent
DEFAULT_REF = APP_DIR / "ref"
DEFAULT_IMAGE = APP_DIR / "image"
DEFAULT_OUTPUT = APP_DIR / "output"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
UPLOAD_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]

# 参照画像のファイル名の付け方（アップロード画面に表示）
NAMING_RULE = (
    "**ファイル名（拡張子を除いた部分）がそのまま人物名になります。**"
    "選手は `MF05_山口蛍.jpg` のように「ポジション+背番号_氏名」、"
    "スタッフ・関係者は `コーチ_伊藤彰.jpg` `MC_初村さん.jpg` のように「役職_氏名」で"
    "保存してください（同じ人物名のファイルを追加すると上書きされます）。"
)
TOP_K = 8  # 1顔あたり保存する候補数
# 一覧にない人物・誰にも該当しなかった写真の共通の出力先
OTHER_NAME = "その他"
OTHER_LABEL = "その他（一覧にない人）"
CHUNK_SECONDS = 1.5  # この秒数ごとに画面へ制御を戻す（一時停止・中止の応答用）

# 中の各ファイルを「役職_氏名」として1人ずつ登録するフォルダ（フォルダ名 → グループ名）
ROLE_DIRS = {"staff": "staff", "スタッフ": "staff", "etc": "etc", "その他": "etc"}
STAFF_GROUP = "staff"

# 一覧表示用のグループ並び順（SP = マスコットなど / etc = MC など）
POS_ORDER = ["GK", "DF", "MF", "FF", "FW", "監督", "SP", STAFF_GROUP, "etc", "その他"]

# staff グループ内での役職の並び順（未定義の役職は末尾に五十音順）
STAFF_ROLE_ORDER = {
    "監督": 0, "ヘッドコーチ": 1, "アシスタントコーチ": 2, "コーチ": 3,
    "GKコーチ": 4, "フィジカルコーチ": 5, "分析コーチ": 6,
    "チーフトレーナー": 10, "トレーナー": 11, "フィジオセラピスト": 12,
    "チーフマネージャー": 20, "マネージャー": 21, "キット": 22, "通訳": 23,
}

st.set_page_config(page_title="顔認証 写真仕分け", page_icon="🧑‍🤝‍🧑", layout="wide")


# ---------------------------------------------------------------- モデル


INSTALL_HINT = f"""
このアプリを動かしている Python に `insightface` が入っていません。
ターミナルで次を実行してください（このアプリを起動している Python 環境に入れる必要があります）：

```bash
"{Path(sys.executable)}" -m pip install insightface onnxruntime
```

インストール後、ブラウザのページを再読み込み（⌘R）してください。
"""


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
    except ModuleNotFoundError:
        st.error("**顔認識ライブラリが見つかりません**")
        st.markdown(INSTALL_HINT)
        st.stop()
    except Exception as e:
        st.error(f"**モデルの読み込みに失敗しました**\n\n```\n{type(e).__name__}: {e}\n```")
        st.markdown(
            "初回起動時は buffalo_l モデル（約 300MB）を GitHub からダウンロードします。"
            "ネットワークに接続しているか確認してください。\n\n"
            "手動で配置する場合は `~/.insightface/models/buffalo_l/` に "
            "`det_10g.onnx` などの .onnx ファイルを置いてください。"
        )
        st.stop()


def read_image(path: Path, max_side: int = 0) -> np.ndarray | None:
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


# ---------------------------------------------------------------- フォルダ選択


def normalize_path_input(text: str) -> str:
    """入力欄の値を整える。Finder からドラッグした file:// URL や引用符も受け付ける。"""
    s = (text or "").strip().strip('"').strip("'")
    if s.startswith("file://"):
        from urllib.parse import unquote, urlparse

        s = unquote(urlparse(s).path)
    return s.rstrip("/") or s


def pick_folder_dialog(initial: str = "") -> str | None:
    """OS のフォルダ選択画面を開く。別プロセスで tkinter を起動する。

    Streamlit のスクリプトはメインスレッドで動かないため、直接 tkinter を呼ぶと
    macOS では落ちることがある。そのため子プロセスに任せている。
    利用できない環境（tkinter 未導入・サーバー実行など）では None を返す。
    """
    code = (
        "import tkinter as tk\n"
        "from tkinter import filedialog\n"
        "r = tk.Tk(); r.withdraw()\n"
        "try:\n"
        "    r.call('wm', 'attributes', '.', '-topmost', True)\n"
        "except Exception:\n"
        "    pass\n"
        f"p = filedialog.askdirectory(initialdir={initial!r} or None, title='フォルダを選択')\n"
        "print(p or '')\n"
    )
    try:
        out = subprocess.run([sys.executable, "-c", code],
                             capture_output=True, text=True, timeout=300)
    except Exception:
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or ""


def subfolders(path: Path) -> list[Path]:
    try:
        return sorted((p for p in path.iterdir()
                       if p.is_dir() and not p.name.startswith(".")),
                      key=lambda p: p.name.lower())
    except Exception:
        return []


def count_images(path: Path) -> int:
    try:
        return sum(1 for p in path.iterdir() if p.suffix.lower() in IMG_EXTS)
    except Exception:
        return 0


@st.dialog("📂 フォルダを選択", width="large")
def folder_dialog(key: str, title: str):
    """OS のダイアログ、またはアプリ内のフォルダ一覧から選ぶ。"""
    st.markdown(f"**{title}**")
    current = Path(st.session_state.get(f"{key}_browse") or st.session_state[key])
    if not current.is_dir():
        current = APP_DIR

    if st.button("🖥 PC のフォルダ選択画面を開く", type="primary", use_container_width=True):
        picked = pick_folder_dialog(str(current))
        if picked is None:
            st.error("この環境では OS のフォルダ選択画面を開けませんでした。"
                     "下の一覧から選ぶか、パスを直接入力してください。")
        elif picked:
            st.session_state[key] = picked
            st.rerun()

    st.divider()
    st.caption(f"現在の場所: `{current}`　（画像 {count_images(current)} 枚）")

    nav = st.columns([1, 1, 3])
    if nav[0].button("⬆️ 上の階層へ", use_container_width=True,
                     disabled=current.parent == current):
        st.session_state[f"{key}_browse"] = str(current.parent)
        st.rerun()
    if nav[1].button("🏠 アプリの場所", use_container_width=True):
        st.session_state[f"{key}_browse"] = str(APP_DIR)
        st.rerun()
    if nav[2].button("✅ この場所に決定", type="primary", use_container_width=True):
        st.session_state[key] = str(current)
        st.rerun()

    subs = subfolders(current)
    if subs:
        st.caption("サブフォルダ（クリックで移動）")
        PER_ROW = 3
        for i in range(0, len(subs), PER_ROW):
            cols = st.columns(PER_ROW)
            for col, d in zip(cols, subs[i:i + PER_ROW]):
                n = count_images(d)
                if col.button(f"📁 {d.name}" + (f"（{n}枚）" if n else ""),
                              key=f"{key}_dir_{d}", use_container_width=True):
                    st.session_state[f"{key}_browse"] = str(d)
                    st.rerun()
    else:
        st.caption("このフォルダにサブフォルダはありません。")

    st.divider()
    typed = st.text_input("パスを直接入力（Finder からフォルダをドラッグ＆ドロップも可）",
                          str(current), key=f"{key}_typed")
    if st.button("入力したパスに決定", use_container_width=True):
        p = normalize_path_input(typed)
        if Path(p).is_dir():
            st.session_state[key] = p
            st.rerun()
        else:
            st.error(f"フォルダが見つかりません: {p}")


def folder_row(key: str, label: str) -> Path:
    """サイドバー用。現在のフォルダ名と「変更」ボタンを出す。"""
    path = Path(st.session_state[key])
    st.caption(f"**{label}**")
    c1, c2 = st.columns([3, 1])
    c1.code(path.name or str(path), language=None)
    if c2.button("変更", key=f"{key}_btn", use_container_width=True):
        st.session_state[f"{key}_browse"] = str(path)
        folder_dialog(key, label)
    if not path.is_dir():
        st.warning("フォルダが見つかりません")
    return path


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


def supporter_mask(faces: list[dict], min_face_px: int, sup: dict | None) -> list[bool]:
    """サポーター（観客）とみなして無視する顔に True を付ける。

    その写真で一番大きく写っている顔（＝主役）に比べて、**大きさも鮮明度も**
    一定の比率に満たない顔を無視する。どちらか一方だけなら残す
    （遠くの選手や、動いてブレた選手を巻き込まないため）。
    """
    mask = [False] * len(faces)
    if not sup or not (sup["rel_size"] or sup["rel_sharp"]):
        return mask
    main = sorted(((f["size"], f.get("sharp", 0.0)) for f in faces
                   if f["size"] >= min_face_px), reverse=True)[:3]
    if len(main) < 2:
        return mask           # 比べる相手がいない
    ref_size = main[0][0]
    ref_sharp = max(s for _, s in main)
    for k, f in enumerate(faces):
        small = not sup["rel_size"] or f["size"] < ref_size * sup["rel_size"]
        soft = not sup["rel_sharp"] or f.get("sharp", 1e9) < ref_sharp * sup["rel_sharp"]
        mask[k] = small and soft
    return mask


def face_threshold(f: dict, threshold: float, sup: dict | None) -> float:
    """小さい顔は特徴が乏しく誤認識しやすいので、しきい値を上乗せする。"""
    if sup and sup["small_bonus"] and f["size"] < sup["small_px"]:
        return threshold + sup["small_bonus"]
    return threshold


def is_ambiguous(f: dict, sup: dict | None) -> bool:
    """1位と2位の類似度が近すぎて、誰とも言い切れない顔か。"""
    c = f["cands"]
    return bool(sup and sup["margin"] and len(c) > 1 and c[0][1] - c[1][1] < sup["margin"])


def top_confident(f: dict, threshold: float, sup: dict | None) -> bool:
    """1位の候補を自動で採用してよい顔か（しきい値・小顔の上乗せ・曖昧さを考慮）。"""
    c = f["cands"]
    return bool(c and c[0][1] >= face_threshold(f, threshold, sup)
                and not is_ambiguous(f, sup))


def derive_assignment(photo: dict, threshold: float, min_face_px: int, top_only: bool,
                      use_numbers: bool = True, min_sharp: float = 0.0,
                      sup: dict | None = None) -> list[str]:
    """保存済み候補と現在のパラメータから、この写真に割り当てる人物名を決める。
    手動確定済み(manual)があればそれを優先。
    顔で誰も判定できなかった場合のみ、背番号 OCR の結果を使う。"""
    manual = photo.get("manual")
    if manual is not None:
        return sorted({n for n in manual if n})

    names: dict[str, float] = {}
    ignored = supporter_mask(photo["faces"], min_face_px, sup)
    for f, skip in zip(photo["faces"], ignored):
        if skip or not usable_face(f, min_face_px, min_sharp):
            continue
        cands = f["cands"]
        if not cands:
            continue
        if top_only:
            if top_confident(f, threshold, sup) and cands[0][1] > names.get(cands[0][0], -1):
                names[cands[0][0]] = cands[0][1]
            continue
        thr = face_threshold(f, threshold, sup)
        for n, s in cands:
            if s >= thr and s > names.get(n, -1):
                names[n] = s
    if names:
        return sorted(names, key=lambda n: -names[n])

    if use_numbers:
        return list(photo.get("num_names") or [])
    return []


def selections_have_person(selections: list[str | None]) -> bool:
    """顔のプルダウンで「その他」以外の人物が選ばれているか。"""
    return any(s and s != OTHER_NAME for s in selections)


# ---------------------------------------------------------------- シーン（連写）

SCENE_PREFIX = "シーン"
SCENE_OFF = "使わない"
SCENE_FILL = "未判定の写真だけ補完"
SCENE_EXPAND = "シーン内の全写真に人物を広げる"
SCENE_FOLDER = "シーンごとのフォルダにまとめる"


def is_scene_name(name: str) -> bool:
    return name.startswith(SCENE_PREFIX)


@st.cache_data(show_spinner=False)
def capture_time(path_str: str, mtime: float) -> float:
    """撮影時刻（EXIF・秒未満つき）。取れなければファイルの更新日時。"""
    from PIL import Image

    try:
        with Image.open(path_str) as im:
            ex = im.getexif()
            sub = ex.get_ifd(0x8769)
            t = datetime.strptime(str(sub.get(36867) or ex.get(306)),
                                  "%Y:%m:%d %H:%M:%S").timestamp()
            frac = str(sub.get(37521) or "").strip()
            return t + (float("0." + frac) if frac.isdigit() else 0.0)
    except Exception:
        return mtime


def detect_scenes(stamps: list[tuple[str, float]], burst_gap: float, link_gap: float,
                  min_bursts: int, bridge_gap: float) -> dict[str, int]:
    """撮影間隔から連写のまとまり（シーン）を見つける。{写真パス: シーン番号（時刻順に1から）}

    1. 隣り合う写真の間隔が link_gap 以下ならつなぐ（鎖）。
    2. 鎖の中に burst_gap 以下の間隔（連写）が min_bursts 個以上あればシーン。
       手で1枚ずつ撮った写真は間隔が長いので、つながってもシーンにならない。
    3. 連写が1つだけの短い鎖は、bridge_gap 以内にシーンがあれば、そのシーンに合流する。
    画像の見た目の類似度は使わない（ズームや動きで似なくなる連写があるため）。
    """
    order = sorted(stamps, key=lambda x: (x[1], x[0]))
    if len(order) < 2:
        return {}
    t = [x[1] for x in order]
    chains: list[list[int]] = [[0]]
    for i in range(1, len(order)):
        if t[i] - t[i - 1] <= link_gap:
            chains[-1].append(i)
        else:
            chains.append([i])

    def bursts(c: list[int]) -> int:
        return sum(1 for a, b in zip(c, c[1:]) if t[b] - t[a] <= burst_gap)

    core = [bursts(c) >= min_bursts for c in chains]
    group = {k: k for k, is_core in enumerate(core) if is_core}
    for k, c in enumerate(chains):
        if core[k] or bursts(c) < 1:
            continue
        near = []
        if k > 0 and core[k - 1]:
            near.append((t[c[0]] - t[chains[k - 1][-1]], k - 1))
        if k + 1 < len(chains) and core[k + 1]:
            near.append((t[chains[k + 1][0]] - t[c[-1]], k + 1))
        near = [x for x in near if x[0] <= bridge_gap]
        if near:
            group[k] = min(near)[1]
    heads = sorted(set(group.values()))
    number = {h: n for n, h in enumerate(heads, 1)}
    out: dict[str, int] = {}
    for k, c in enumerate(chains):
        if k in group:
            for i in c:
                out[order[i][0]] = number[group[k]]
    return out


SCENE_NAME_MAX = 3    # シーン名に付ける人物の最大数。これを超えると集合写真とみなして付けない


def scene_folder_name(no: int, members: list[dict], assigned: list[list[str]],
                      min_support: int) -> str:
    """シーンのフォルダ名。人物が選ばれていれば「シーン01-MF22翁長聖」、なければ「シーン01」。

    人物は、シーン内の min_support 枚以上に判定された人、またはレビューで手動選択された人。
    SCENE_NAME_MAX 人を超えるときは、名前を付けず「シーン01」のままにする。
    """
    label = f"{SCENE_PREFIX}{no:02d}"
    count: dict[str, int] = defaultdict(int)
    explicit: set[str] = set()
    for p, names in zip(members, assigned):
        for n in names:
            if n == OTHER_NAME or is_scene_name(n):
                continue
            count[n] += 1
            if p.get("manual") is not None:
                explicit.add(n)
    people = [n for n, c in sorted(count.items(), key=lambda x: -x[1])
              if c >= min_support or n in explicit]
    if not people or len(people) > SCENE_NAME_MAX:
        return label
    return label + "-" + "・".join(short_label(n).replace(" ", "").replace("　", "")
                                  for n in people)


MANUAL_SCENE_KEY = "手動:"   # 手動でまとめたシーンの鍵の接頭辞。後ろに作った順の通し番号が付く
MANUAL_SCENE_MAX = 60        # 手動でまとめられる枚数の上限（範囲の選び間違いで大量に巻き込まないため）


def manual_serial(key) -> int | None:
    """手動シーンの鍵（「手動:3」）から通し番号を取り出す。手動シーンの鍵でなければ None。"""
    if isinstance(key, str) and key.startswith(MANUAL_SCENE_KEY):
        tail = key[len(MANUAL_SCENE_KEY):]
        if tail.isdigit():
            return int(tail)
    return None


def next_manual_key(over: dict) -> str:
    """新しく手動でまとめるシーンの鍵。いま使われている通し番号の次を使う。"""
    used = [manual_serial(v) for v in over.values()]
    return f"{MANUAL_SCENE_KEY}{max([u for u in used if u], default=0) + 1}"


def resolve_scenes(stamps: list[tuple[str, float]], auto_of: dict[str, int],
                   over: dict) -> tuple[dict[str, int], dict[int, str]]:
    """自動で見つけたシーンに、手動の指定を重ねて最終的なシーンを決める。

    over: {写真名: 0（シーンに入れない） / 鍵の文字列（そのシーンに入れる） / 整数（旧形式: 自動シーンの番号）}
    シーンの鍵は、自動のシーンでは「最も早い写真の名前」、手動でまとめたシーンでは「手動:通し番号」。

    **既存のシーン番号は、手動で操作しても変わらない**ようにする。
    - 自動のシーンの番号は、自動検出で付いた番号のまま（手動で空になっても欠番にするだけで詰めない）
    - 手動でまとめたシーンは、自動の最後の番号の次から、作った順に付ける
    戻り値: ({写真パス: シーン番号}, {シーン番号: 鍵})
    """
    t = dict(stamps)
    earliest: dict[int, tuple[float, str]] = {}
    for path, n in auto_of.items():
        if path in t and (n not in earliest or (t[path], path) < earliest[n]):
            earliest[n] = (t[path], path)
    auto_key = {n: Path(path).name for n, (_, path) in earliest.items()}
    number_of: dict[str, int] = {k: n for n, k in auto_key.items()}
    last_auto = max(auto_of.values(), default=0)

    members: dict[str, list[str]] = defaultdict(list)
    for path, _ in stamps:
        v = over.get(Path(path).name)
        if v is None:
            key = auto_key.get(auto_of.get(path, -1))
        elif isinstance(v, str):
            key = v
        elif v == 0:
            key = None
        else:
            key = auto_key.get(int(v))
        if key:
            members[key].append(path)
    scene_of: dict[str, int] = {}
    keys: dict[int, str] = {}
    for key, paths in members.items():
        serial = manual_serial(key)
        if serial is not None:
            no = last_auto + serial
        elif key in number_of and len(paths) >= 2:
            no = number_of[key]       # 手動で一部を移した結果1枚だけ残ったものは、シーンとして扱わない
        else:
            continue
        keys[no] = key
        for path in paths:
            scene_of[path] = no
    return scene_of, keys


def build_plan(photos: list[dict], threshold: float, min_face_px: int, top_only: bool,
               min_sharp: float, sup: dict | None, scene_mode: str,
               scene_of: dict[str, int], min_support: int, scene_over: dict,
               ) -> tuple[dict[int, list[str]], dict[int, str]]:
    """出力・集計に使う振り分け計画を作る。

    戻り値: ({写真の番号: [フォルダ名]}, {写真の番号: シーン処理で変えた内容の注記})。

    「シーンごとのフォルダにまとめる」では、シーンに入る写真は人物フォルダへは入れず、
    シーンフォルダだけに入れる（レビューで人物を選んだ写真も同じ）。
    フォルダ名は、人物が選ばれていれば「シーン01-MF22翁長聖」、なければ「シーン01」。
    シーンに入れたくない写真は、レビューで「シーンに入れない」を選ぶ。

    「補完」「広げる」では、レビューで人物を選んで確定した写真は変更しない。
    人物なしで確定しただけ（manual が空）の写真は未確定と同じ扱いで、補完の対象になる。
    レビューでシーンを指定した写真（scene_over に入っている写真）は、人物の指定よりシーンを優先する。
    """
    base = {i: derive_assignment(p, threshold, min_face_px, top_only,
                                 min_sharp=min_sharp, sup=sup)
            for i, p in enumerate(photos)}
    plan = {i: list(n) for i, n in base.items()}
    notes: dict[int, str] = {}
    if scene_mode == SCENE_OFF:
        return plan, notes

    members: dict[int, list[int]] = defaultdict(list)
    for i, p in enumerate(photos):
        no = scene_of.get(p["path"], 0)
        if no:
            members[no].append(i)

    for no, idxs in members.items():
        label = f"{SCENE_PREFIX}{no:02d}"
        if scene_mode == SCENE_FOLDER:
            folder = scene_folder_name(no, [photos[i] for i in idxs],
                                       [base[i] for i in idxs], min_support)
            for i in idxs:
                plan[i] = [folder]
                notes[i] = folder
            continue
        # 人物を手動で選んだ写真は変えない
        free = [i for i in idxs
                if not (base[i] and photos[i].get("manual") is not None
                        and Path(photos[i]["path"]).name not in scene_over)]
        # シーン内で min_support 枚以上に確定している人物だけを「このシーンの人物」とする
        support: dict[str, int] = defaultdict(int)
        for i in idxs:
            for n in base[i]:
                support[n] += 1
        strong = [n for n, c in sorted(support.items(), key=lambda x: -x[1])
                  if c >= min_support]
        for i in free:
            if scene_mode == SCENE_FILL:
                # 人物が1人に絞れるシーンだけ。複数人が食い違うシーンは補完しない
                if not base[i] and len(strong) == 1:
                    plan[i] = list(strong)
                    notes[i] = f"{label}から補完"
            else:
                add = [n for n in strong if n not in base[i]]
                if add:
                    plan[i] = base[i] + add
                    notes[i] = f"{label}から追加"
    return plan, notes


STATE_FILE = ".face_sorting.json"   # 対象フォルダに置く作業状態
STATE_VERSION = 1


def state_path(image_dir: Path) -> Path:
    return image_dir / STATE_FILE


def save_state(image_dir: Path) -> None:
    """解析結果とレビュー内容を対象フォルダに保存する（再起動しても続きから使える）。"""
    photos = st.session_state.get("photos")
    over = st.session_state.get("scene_over") or {}
    if (not photos and not over) or not image_dir.is_dir():
        return
    try:
        state_path(image_dir).write_text(
            json.dumps({"version": STATE_VERSION, "photos": photos or [],
                        "scene_over": over},
                       ensure_ascii=False),
            encoding="utf-8")
    except Exception:
        pass       # 保存できなくても作業は続けられるので黙って諦める


def load_state(image_dir: Path) -> list[dict] | None:
    """保存済みの作業状態を読み込む。

    ファイルが消えている写真は除く。同名のまま中身が差し替えられた写真も、
    サイズ・更新日時が記録と食い違っていれば古い解析結果を捨てて未解析扱いに戻す
    （指紋情報が無い旧形式の記録はそのまま信用する）。
    """
    f = state_path(image_dir)
    if not f.is_file():
        return None
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return None
    if data.get("version") != STATE_VERSION:
        return None
    photos = []
    for p in data.get("photos", []):
        path = Path(p.get("path", ""))
        try:
            st = path.stat()
        except OSError:
            continue  # 写真が消えている
        rec_mtime, rec_size = p.get("mtime"), p.get("fsize")
        if rec_mtime is not None and rec_size is not None:
            if int(st.st_mtime) != rec_mtime or st.st_size != rec_size:
                continue  # 同名だが中身が差し替えられている → 未解析扱いに戻す
        photos.append(p)
    return photos or None


def load_scene_over(image_dir: Path) -> dict:
    """保存済みの、シーンの手動指定 {写真名: 0 または鍵}。旧形式（写真ごとの scene）も引き継ぐ。"""
    f = state_path(image_dir)
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if data.get("version") != STATE_VERSION:
        return {}
    over = dict(data.get("scene_over") or {})
    for p in data.get("photos", []):
        if p.get("scene") is not None and p.get("name"):
            over.setdefault(p["name"], p["scene"])
    # 旧形式の手動シーンの鍵（手動:開始の写真名）を、通し番号の鍵（手動:1）に移す
    old_keys = sorted({v for v in over.values() if isinstance(v, str)
                       and v.startswith(MANUAL_SCENE_KEY) and manual_serial(v) is None})
    if old_keys:
        base = max([manual_serial(v) or 0 for v in over.values()], default=0)
        moved = {k: f"{MANUAL_SCENE_KEY}{base + i}" for i, k in enumerate(old_keys, 1)}
        over = {n: moved.get(v, v) if isinstance(v, str) else v for n, v in over.items()}
    return over


def merge_photos(new_photos: list[dict]) -> None:
    """解析結果を既存の結果に追記する（同じファイルは重複させない）。
    レビューで確定済みの内容は保持される。"""
    existing = st.session_state.get("photos") or []
    seen = {p["path"] for p in existing}
    merged = existing + [p for p in new_photos if p["path"] not in seen]
    merged.sort(key=lambda p: p["name"])
    st.session_state["photos"] = merged


def put_file(src: Path, dest_dir: Path, move: bool = False) -> Path:
    """出力先へ配置する。同名ファイルが既にあれば上書きする。"""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.resolve() == src.resolve():
        return dest
    if move:
        if dest.exists():
            dest.unlink()
        shutil.move(str(src), str(dest))
    else:
        shutil.copy2(src, dest)  # 既存ファイルは上書きされる
    return dest


# ---------------------------------------------------------------- マニュアル

README_PATH = APP_DIR / "README.md"

FALLBACK_MANUAL = """## マニュアル

README.md が見つかりませんでした。アプリと同じフォルダに README.md を置いてください。

### 基本の流れ
1. **① 解析** タブ →「参照画像を解析」→「解析を実行」
2. **② レビュー** タブ → 複数顔の写真で各顔の人物を確定
3. **③ 出力** タブ → ドライランで確認してからファイル出力
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

st.title("🧑‍🤝‍🧑 顔認証による写真仕分け")

ss = st.session_state
ss.setdefault("ref_failed", None)
ss.setdefault("photos", None)
ss.setdefault("review_idx", 0)
ss.setdefault("job", None)
ss.setdefault("open_pi", None)
ss.setdefault("scene_over", {})
ss.setdefault("image_dir", str(DEFAULT_IMAGE))

_missing = check_deps()
if _missing:
    st.error("**必要なパッケージが不足しています: " + ", ".join(_missing) + "**")
    st.markdown(
        f"""
ターミナルで次を実行してください（**このアプリを起動している Python 環境**に入れる必要があります）：

```bash
"{Path(sys.executable)}" -m pip install {' '.join(_missing)}
```

インストール後、ブラウザのページを再読み込み（⌘R）してください。
"""
    )
    st.stop()

with st.sidebar:
    if st.button("📖 マニュアルを開く", use_container_width=True):
        manual_dialog()
    st.divider()

    st.header("フォルダ設定")
    ref_dir = DEFAULT_REF  # 参照フォルダは固定（追加は「👥 人物一覧」タブから）
    image_dir = folder_row("image_dir", "対象フォルダ（仕分けする写真）")
    # 出力先は対象フォルダと同じ階層の output/（無ければ出力時に作成）
    output_dir = image_dir.parent / "output"

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
    _lo, _lm, _ls = load_learned(DEFAULT_REF)
    st.caption(f"蓄積 {len(_lo)} 件（{len(set(_lo))} 名）／`{DEFAULT_REF.name}/` に保存。"
               "**対象フォルダを変えても引き継がれます**")

    st.header("サポーターの除外")
    use_sup = st.checkbox(
        "サポーター（観客）を無視する", value=True,
        help="その写真で最も大きく写っている顔（主役）に比べて、小さく不鮮明な顔を無視します。"
             "観客席・サイドラインの人が選手として認識されるのを防ぎます。"
             "解析後に変えても再解析は不要です。",
    )
    with st.expander("サポーター除外の詳細", expanded=False):
        sup_rel_size = st.slider(
            "主役に対する顔の大きさ (%)", 0, 100, 60, 5, disabled=not use_sup,
            help="主役の顔の大きさのこの割合より小さく、かつ下の鮮明度も満たさない顔を無視します。"
                 "0 でこの条件を使いません。")
        sup_rel_sharp = st.slider(
            "主役に対する鮮明度 (%)", 0, 100, 50, 5, disabled=not use_sup,
            help="主役の鮮明度のこの割合より不鮮明、かつ上の大きさも満たさない顔を無視します。"
                 "0 でこの条件を使いません。")
        sup_small_px = st.slider(
            "小さい顔とみなす大きさ (px)", 0, 200, 80, 5, disabled=not use_sup,
            help="これより小さい顔は特徴が乏しく誤認識しやすいため、しきい値を上乗せします。")
        sup_small_bonus = st.slider(
            "小さい顔のしきい値の上乗せ", 0.00, 0.30, 0.10, 0.01, disabled=not use_sup)
        sup_margin = st.slider(
            "1位と2位の最小の差", 0.00, 0.20, 0.05, 0.01, disabled=not use_sup,
            help="1位と2位の類似度の差がこれ未満の顔は、誰とも言い切れないので採用しません。"
                 "0 で無効。")
    sup = ({"rel_size": sup_rel_size / 100, "rel_sharp": sup_rel_sharp / 100,
            "small_px": sup_small_px, "small_bonus": sup_small_bonus,
            "margin": sup_margin} if use_sup else None)

    st.header("シーン（連写）")
    scene_mode = st.selectbox(
        "連写シーンの扱い", [SCENE_OFF, SCENE_FILL, SCENE_EXPAND, SCENE_FOLDER],
        help="撮影時刻から連写のまとまり（シーン）を見つけて、仕分けに使います。\n\n"
             f"- **{SCENE_FILL}**: 顔で判定できなかった写真に、同じシーンの人物を引き継ぎます"
             "（人物が1人に絞れるシーンのみ）\n"
             f"- **{SCENE_EXPAND}**: シーン内の全写真を、そのシーンの人物のフォルダにも入れます\n"
             f"- **{SCENE_FOLDER}**: 人物ではなく「{SCENE_PREFIX}01」などのフォルダだけに入れます"
             f"（人物フォルダには入れません。人物が選ばれていれば「{SCENE_PREFIX}01-MF22翁長聖」のように"
             "人物名が付きます。マスコットなど人物で分けられない連写向け）\n\n"
             "レビューで人物を選んで確定した写真は、上2つでは変更されません。"
             "シーンフォルダでは、シーンに入った写真はすべてシーンフォルダに入ります"
             "（入れたくない写真はレビューで「シーンに入れない」を選びます）。",
    )
    with st.expander("シーン判定の詳細", expanded=False):
        burst_gap = st.slider(
            "連写とみなす間隔（秒）", 0.1, 1.0, 0.5, 0.05,
            help="これ以下の間隔で撮った隣り合う写真を連写とみなします。")
        link_gap = st.slider(
            "つなぐ間隔（秒）", 0.5, 3.0, 1.0, 0.1,
            help="連写の途中で一瞬止まっても、この間隔以内なら同じシーンとしてつなぎます。")
        min_frames = st.number_input(
            "シーンの最小枚数", 3, 20, 3, step=1,
            help="これ未満の連写はシーンにしません（手で1枚ずつ撮った写真の誤判定を防ぎます）。")
        bridge_gap = st.slider(
            "短い連写を隣のシーンに合流する間隔（秒）", 0.0, 5.0, 3.0, 0.5,
            help="2枚だけの連写が、この間隔以内に別のシーンの直前・直後にあれば合流させます。")
        min_support = st.number_input(
            "シーンの人物とみなすのに必要な確定枚数", 1, 10, 2, step=1,
            help="シーン内でこの枚数以上に判定された人物だけを、そのシーンの人物とみなします。"
                 "補完・広げるでは引き継ぐ人物、シーンフォルダではフォルダ名に付ける人物になります"
                 "（レビューで手動選択した人物は1枚でも対象）。",
            disabled=scene_mode == SCENE_OFF)

    st.header("出力設定")
    op_mode = st.radio("ファイル操作", ["コピー", "移動"], index=0, horizontal=True)
    save_unmatched = st.checkbox(f"該当なしの写真も {OTHER_NAME}/ に入れる", value=True,
                                 help="誰にも該当しなかった写真を、一覧にない人と同じフォルダにまとめます。")

# ---------------------------------------------------------------- 概要

people = collect_refs(ref_dir)
targets = list_targets(image_dir)

# 対象フォルダを開いたとき、保存済みの作業状態があれば読み込む
if ss.get("loaded_dir") != str(image_dir):
    restored = load_state(image_dir)
    ss["scene_over"] = load_scene_over(image_dir)
    for _p in restored or []:
        _p.pop("scene", None)          # 旧形式。scene_over に移した
    ss["photos"] = restored
    ss["job"] = None
    ss["review_idx"] = 0
    ss["open_pi"] = None
    ss["loaded_dir"] = str(image_dir)
    if restored:
        st.toast(f"前回の作業状態を読み込みました（{len(restored)} 枚）")
all_names = sorted(people.keys())

# 連写のまとまり（シーン）。解析結果ではなく撮影時刻と手動の指定で決まるので、解析前でも求められる
scene_of: dict[str, int] = {}
scene_keys: dict[int, str] = {}       # シーン番号 → 鍵（手動で入れるときに使う）
scene_order: list[str] = []           # 撮影順の写真名
if scene_mode != SCENE_OFF:
    _stamps = []
    for _p in targets:
        try:
            _stamps.append((str(_p), capture_time(str(_p), _p.stat().st_mtime)))
        except OSError:
            pass
    _auto = detect_scenes(_stamps, burst_gap, link_gap,
                          max(1, int(min_frames) - 1), bridge_gap)
    scene_of, scene_keys = resolve_scenes(_stamps, _auto, ss["scene_over"])
    scene_order = [Path(_p).name for _p, _t in sorted(_stamps, key=lambda x: (x[1], x[0]))]

# シーン番号 → 写真名の一覧（レビュー画面でシーンを選ぶときの表示用）
scene_names: dict[int, list[str]] = defaultdict(list)
scene_no_by_name: dict[str, int] = {}
for _path, _no in scene_of.items():
    scene_names[_no].append(Path(_path).name)
    scene_no_by_name[Path(_path).name] = _no

c1, c2, c3, c4 = st.columns(4)
c1.metric("登録人物数", len(people))
c2.metric("参照画像数", sum(len(v) for v in people.values()))
c3.metric("対象写真数", len(targets))
c4.metric("解析済み", len(ss["photos"]) if ss["photos"] else 0)

if not people:
    st.error(f"参照画像が見つかりません: {ref_dir}")
    st.markdown(NAMING_RULE)
    first_refs = st.file_uploader("参照画像を選択（複数可）", type=UPLOAD_TYPES,
                                  accept_multiple_files=True, key="ref_uploader_first")
    if first_refs and st.button("参照画像を追加", type="primary"):
        ref_dir.mkdir(parents=True, exist_ok=True)
        for uf in first_refs:
            (ref_dir / Path(uf.name).name).write_bytes(uf.getbuffer())
        st.rerun()
    st.stop()

tab_people, tab_run, tab_review, tab_apply, tab_result, tab_manual = st.tabs(
    ["👥 人物一覧", "① 解析", "② レビュー", "③ 出力", "④ 結果", "📖 マニュアル"]
)

# ---------------------------------------------------------------- 人物一覧タブ

with tab_people:
    with st.expander("➕ 参照画像を追加する"):
        st.markdown(NAMING_RULE)
        dest_kind = st.radio(
            "保存先",
            [f"選手など（{ref_dir.name}/ 直下）", f"スタッフ（{ref_dir.name}/staff/）",
             f"その他の関係者（{ref_dir.name}/etc/）"],
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
            sub = "staff" if "スタッフ" in dest_kind else ("etc" if "/etc/" in dest_kind else "")
            dest_dir = ref_dir / sub if sub else ref_dir
            dest_dir.mkdir(parents=True, exist_ok=True)
            added, replaced = [], []
            for uf in new_refs:
                dest = dest_dir / Path(uf.name).name
                (replaced if dest.exists() else added).append(dest.stem)
                dest.write_bytes(uf.getbuffer())
            msg = f"{len(added)} 件を追加しました。" if added else ""
            if replaced:
                msg += f"{len(replaced)} 件を上書きしました（{', '.join(replaced[:3])}…）。"
            st.success(msg + "次回の解析時に自動で登録されます。")
            st.rerun()

    l_owners, l_matrix, l_srcs = load_learned(ref_dir)
    if l_owners:
        cnt: dict[str, int] = defaultdict(int)
        for n in l_owners:
            cnt[n] += 1
        with st.expander(f"🧠 修正から覚えた顔: {len(l_owners)} 件（{len(cnt)} 名）"):
            st.caption("レビューで人物を選び直した顔です。次回の解析から照合に使われます。")
            st.markdown("\n".join(
                f"- {short_label(n)}: {c} 件" for n, c in
                sorted(cnt.items(), key=lambda x: -x[1])))
            c1, c2 = st.columns(2)
            if c1.button("学習データをすべて削除"):
                learned_path(ref_dir).unlink(missing_ok=True)
                st.rerun()
            c2.download_button("元になった写真の一覧 (CSV)",
                               ("人物,写真\n" + "\n".join(
                                   f"{short_label(n)},{s2}" for n, s2 in zip(l_owners, l_srcs))
                                ).encode("utf-8-sig"),
                               file_name="learned_faces.csv", mime="text/csv")

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
               "何回かに分けて解析でき、結果は積み上がります。"
               "参照画像の登録は初回の解析時に自動で行われます"
               "（参照画像を追加・変更したときだけ自動で登録し直します）。")

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
        state_path(image_dir).unlink(missing_ok=True)
        save_state(image_dir)          # シーンの手動指定だけは残す
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
            ref_signature(ref_dir), str(ref_dir), det_size, det_thresh)
        ss["ref_failed"] = ref_failed
        # 手動修正から覚えた顔も参照に加える
        l_owners, l_matrix, _ = load_learned(ref_dir)
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
                       "今回は顔認証のみで解析します。")
            st.caption(
                "インストール済みなのにこの表示が出る場合は、**アプリを起動したまま "
                "pip install した**可能性があります。ターミナルで Streamlit を一度止めて "
                "（Ctrl+C）、起動し直してください。\n\n"
                f"インストール先の確認: `\"{Path(sys.executable)}\" "
                "-c \"import rapidocr_onnxruntime, sys; print(sys.executable)\"`"
            )

        t0 = time.time()
        while job["cursor"] < len(job["todo"]) and time.time() - t0 < CHUNK_SECONDS:
            path = Path(job["todo"][job["cursor"]])
            try:
                _st = path.stat()
                fp = {"mtime": int(_st.st_mtime), "fsize": _st.st_size}
            except OSError:
                fp = {}
            img = read_image(path)
            if img is None:
                job["photos"].append(
                    {"path": str(path), "name": path.name, "faces": [], "error": "読込失敗", **fp})
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
            rec = {"path": str(path), "name": path.name, "faces": faces_meta, **fp}
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
        save_state(image_dir)
        if job["cursor"] >= len(job["todo"]):
            job["status"] = "done"
        st.rerun()

# ---------------------------------------------------------------- レビュータブ

with tab_review:
    if not ss["photos"]:
        st.info("先に「① 解析」を実行してください。")
    else:
        photos = ss["photos"]

        scope = st.radio(
            "レビュー対象",
            ["複数人の写真", "すべての写真", "要確認（該当なし・低確度）", "未確認のみ"],
            horizontal=True, key="rev_scope",
            help="取りこぼしを防ぐには「未確認のみ」を選び、空になるまで進めてください。",
        )

        def _needs_check(p: dict) -> bool:
            if not derive_assignment(p, threshold, min_face_px, top_only,
                                     min_sharp=min_sharp, sup=sup):
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
                    add_learned(ref_dir, name, emb, photo_["name"])
                    f["learned"] = name
                    done.append(short_label(name))
                if done:
                    save_state(image_dir)
                    st.toast("覚えました: " + "、".join(done))

            def autosave(idx: int) -> None:
                """自動判定と違う内容になっていれば保存する（押し忘れ防止）。"""
                state = widget_state(idx)
                if state is None:
                    return
                sel, ext = state
                picked = sorted({n for n in sel if n} | set(ext))
                auto = derive_assignment({"faces": photos[idx]["faces"]}, threshold,
                                         min_face_px, top_only, min_sharp=min_sharp, sup=sup)
                # 自動判定は類似度順、こちらは名前順なので集合で比べる
                if set(picked) != set(auto) or photos[idx].get("manual") is not None:
                    changed = photos[idx].get("manual") != picked
                    photos[idx]["manual_faces"] = sel
                    photos[idx]["manual_extra"] = ext
                    photos[idx]["manual"] = picked
                    if changed:
                        save_state(image_dir)
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
                save_state(image_dir)
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

            if scene_mode != SCENE_OFF and photo["path"] in scene_of:
                no = scene_of[photo["path"]]
                names_ = sorted(scene_names[no])
                st.caption(f"🎞 {SCENE_PREFIX}{no:02d}（{names_[0]} 〜 {names_[-1]}・{len(names_)}枚）"
                           + ("　※手動で指定" if photo["name"] in ss["scene_over"] else ""))

            if scene_mode != SCENE_OFF and scene_order:
                with st.expander("✂ シーンを手動でまとめる（連写として検出されなかった写真）"):
                    st.caption("開始と終了の写真を選ぶと、その間（撮影順）の写真を1つのシーンにまとめます。"
                               "自動で見つかったシーンの一部を含めても、まとめた範囲が優先されます。")
                    i0 = scene_order.index(photo["name"]) if photo["name"] in scene_order else 0
                    r1, r2 = st.columns(2)
                    s_name = r1.selectbox("開始の写真", scene_order, index=i0, key=f"scs_{pi}")
                    e_name = r2.selectbox("終了の写真", scene_order, index=i0, key=f"sce_{pi}")
                    lo, hi = sorted((scene_order.index(s_name), scene_order.index(e_name)))
                    span = scene_order[lo:hi + 1]
                    in_scene = sum(1 for n in span if n in scene_no_by_name)
                    st.caption(f"{span[0]} 〜 {span[-1]}（{len(span)}枚）"
                               + (f"　うち {in_scene} 枚は現在シーンに入っています" if in_scene else "")
                               + (f"　※範囲が広すぎます（上限 {MANUAL_SCENE_MAX} 枚）。"
                                  "開始・終了の写真を確認してください"
                                  if len(span) > MANUAL_SCENE_MAX else ""))
                    g1, g2 = st.columns(2)
                    if g1.button("この範囲を1つのシーンにまとめる", type="primary",
                                 key=f"scg_{pi}",
                                 disabled=not 2 <= len(span) <= MANUAL_SCENE_MAX):
                        key_ = next_manual_key(ss["scene_over"])
                        for n in span:
                            ss["scene_over"][n] = key_
                        save_state(image_dir)
                        st.rerun()
                    if g2.button("この範囲の手動指定を自動に戻す", key=f"scr_{pi}"):
                        for n in span:
                            ss["scene_over"].pop(n, None)
                        save_state(image_dir)
                        st.rerun()

            left, right = st.columns([3, 5])
            with left:
                if img is not None:
                    st.image(draw_boxes(img, photo["faces"]), caption=photo["name"],
                             use_container_width=True)

            with right:
                prev_faces = photo.get("manual_faces")

                # --- 顔ごとの初期値をあらかじめ決めておく（表示は下段だが値は先に必要） ---
                face_setup = []
                is_supporter = supporter_mask(photo["faces"], min_face_px, sup)
                for k, f in enumerate(photo["faces"]):
                    score_map = {n: s for n, s in f["cands"]}
                    options = (list(score_map)
                               + [n for n in all_names if n not in score_map]
                               + [OTHER_NAME])
                    ok = usable_face(f, min_face_px, min_sharp) and not is_supporter[k]
                    confident = bool(ok and top_confident(f, threshold, sup))
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
                current_of: list[str] = []
                checked_of: list[bool] = []
                for k, (_, options, default, default_on, _) in enumerate(face_setup):
                    name = st.session_state.get(f"rev_{pi}_{k}", default)
                    if name not in options:
                        name = default
                    on = bool(st.session_state.get(f"use_{pi}_{k}", default_on))
                    current_of.append(name)
                    checked_of.append(on)
                    selections.append(name if on else None)

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

                # --- シーン（連写）への振り分け（取りこぼしを手動で入れる） ---
                if scene_mode != SCENE_OFF:
                    sc_opts = [None, 0] + sorted(scene_names)
                    ov = ss["scene_over"].get(photo["name"])
                    if ov is None:
                        cur_sc = None
                    elif isinstance(ov, str) or ov != 0:
                        cur_sc = scene_of.get(photo["path"])
                        if cur_sc not in sc_opts:
                            cur_sc = None
                    else:
                        cur_sc = 0
                    # 表示する値は、毎回、保存済みの指定から決め直す（番号が振り直されても合う）
                    st.session_state[f"scene_{pi}"] = cur_sc

                    def _sc_fmt(v) -> str:
                        if v is None:
                            return "自動"
                        if v == 0:
                            return "シーンに入れない"
                        ns = sorted(scene_names[v])
                        return f"{SCENE_PREFIX}{v:02d}（{ns[0]} 〜 {ns[-1]}・{len(ns)}枚）"

                    def _sc_changed(pi_: int, name_: str) -> None:
                        v = st.session_state.get(f"scene_{pi_}")
                        if v is None:
                            ss["scene_over"].pop(name_, None)
                        elif v == 0:
                            ss["scene_over"][name_] = 0
                        else:
                            ss["scene_over"][name_] = scene_keys[v]
                        save_state(image_dir)

                    st.selectbox(
                        "シーン（連写）への振り分け", sc_opts, format_func=_sc_fmt,
                        key=f"scene_{pi}", on_change=_sc_changed, args=(pi, photo["name"]),
                        help="連写として検出できなかった写真を、手動でシーンに入れられます。"
                             "「シーンに入れない」を選ぶと、この写真だけシーン処理の対象外になります。"
                             "人物の指定よりシーンの選択が優先されます。"
                             "複数の写真をまとめて新しいシーンにするときは、上の「シーンを手動でまとめる」を使います。",
                    )

                # --- 仕分け先 ---
                assigned = sorted({n for n in selections if n} | set(extra))
                dest = [short_label(n) for n in assigned]
                eff_no = scene_of.get(photo["path"], 0) if scene_mode != SCENE_OFF else 0
                if scene_mode == SCENE_FOLDER and eff_no:
                    # シーンの写真は人物フォルダへ入れない。人物名つきのフォルダ名を計画から取る
                    plan_all, _ = build_plan(photos, threshold, min_face_px, top_only,
                                             min_sharp, sup, scene_mode, scene_of,
                                             int(min_support), ss["scene_over"])
                    dest = plan_all[pi]
                st.info("仕分け先: " + ("、".join(dest) or f"{OTHER_NAME}（該当なし）"))
                st.caption("変更は別の写真へ移ったときに自動保存されます"
                           "（「確定して次へ」を押さなくても反映されます）。")

                st.divider()

                # --- 検出された顔ごとの指定 ---
                st.markdown("##### 検出された顔と人物の割り当て")
                st.caption("顔ごとに自動認識した人物を表示しています。**仕分けたい顔にチェック**を入れてください。"
                           "人物が違うときは下のプルダウンで選び直せます。"
                           "変更内容は上の「仕分け先」に反映されます。")

                # ボタンは顔の一覧より先に置く（「その他」の顔が多くても、下までスクロールしない）
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
                        min_sharp=min_sharp, sup=sup)
                    if pick < len(queue) - 1:
                        ss["review_idx"] = pick + 1
                    st.rerun()
                if b3.button("確定を取り消す", key=f"rst_{pi}",
                             disabled=photo.get("manual") is None):
                    for key in ("manual", "manual_faces", "manual_extra"):
                        photo.pop(key, None)
                    st.rerun()

                def face_card(k: int) -> None:
                    f = photo["faces"][k]
                    score_map, options, default, default_on, ok = face_setup[k]
                    sel_key = f"rev_{pi}_{k}"
                    current = st.session_state.get(sel_key, default)
                    if current not in options:
                        current = default
                    if img is not None:
                        st.image(crop_face(img, f["bbox"]), use_container_width=True)
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
                    if is_supporter[k]:
                        note += "（サポーターとみなして対象外）"
                    elif not ok:
                        note += "（対象外）"
                    st.caption(note)

                def face_grid(ks: list[int]) -> None:
                    PER_ROW = 4
                    for row0 in range(0, len(ks), PER_ROW):
                        cols = st.columns(PER_ROW)
                        for col, k in zip(cols, ks[row0:row0 + PER_ROW]):
                            with col:
                                face_card(k)

                if photo["faces"]:
                    # 仕分ける顔（チェックあり）→ 一覧の人でチェックなし → チェックなしの「その他」。
                    # 同じ区分の中では、いま選ばれている人物の類似度が高い順。
                    def _rank(k: int) -> tuple:
                        g = 0 if checked_of[k] else (2 if current_of[k] == OTHER_NAME else 1)
                        return (g, -face_setup[k][0].get(current_of[k], 0.0), k)

                    order = sorted(range(len(photo["faces"])), key=_rank)
                    on_ks = [k for k in order if _rank(k)[0] == 0]
                    cand_ks = [k for k in order if _rank(k)[0] == 1]
                    other_ks = [k for k in order if _rank(k)[0] == 2]

                    if on_ks:
                        st.markdown(f"**✅ 仕分ける顔（{len(on_ks)}）**")
                        face_grid(on_ks)
                    if cand_ks:
                        st.markdown(f"**候補あり・チェックなし（{len(cand_ks)}）**")
                        face_grid(cand_ks)
                    if other_ks:
                        with st.expander(
                                f"チェックなしの「{OTHER_NAME}」の顔（{len(other_ks)}）"
                                "　※観客席・サポーターなど", expanded=False):
                            face_grid(other_ks)
                else:
                    st.caption("顔は検出されませんでした。")

# ---------------------------------------------------------------- 出力タブ

with tab_apply:
    if not ss["photos"]:
        st.info("先に「① 解析」を実行してください。")
    else:
        photos = ss["photos"]
        plan, scene_notes = build_plan(photos, threshold, min_face_px, top_only, min_sharp,
                                       sup, scene_mode, scene_of, int(min_support),
                                       ss["scene_over"])
        counter: dict[str, int] = defaultdict(int)
        for names in plan.values():
            for n in names:
                counter[n] += 1
        unmatched = [i for i, n in plan.items() if not n]

        m1, m2, m3 = st.columns(3)
        m1.metric("仕分けされる件数", sum(counter.values()))
        m2.metric("作成フォルダ数", len(counter))
        m3.metric("該当なしの写真", len(unmatched))

        st.caption("しきい値・最小顔サイズを変更すると、再解析なしでこの内容が更新されます。"
                   "レビューで確定済みの写真は手動選択が優先されます。")
        if scene_mode != SCENE_OFF:
            n_scenes = len(set(scene_of.values()))
            st.caption(f"🎞 連写シーン: {n_scenes} 個（{len(scene_of)} 枚）／"
                       f"{scene_mode}: {len(scene_notes)} 枚に適用")

        n_done = sum(1 for p in photos if p.get("exported"))
        only_new = st.checkbox(
            f"まだ出力していない写真だけ（出力済み {n_done} 枚をスキップ）",
            value=bool(n_done), disabled=not n_done,
            help="追加で解析した分だけを書き出したいときに使います。"
                 "レビューで内容を変えた写真は、出力済みでも再出力されます。",
        )
        dry_run = st.checkbox("ドライラン（実際には書き込まない）", value=True)
        clean_first = st.checkbox(
            "出力前に人物フォルダを削除する（前回の結果を残さない）", value=False,
            help=f"{output_dir} の中のフォルダをすべて消してから書き出します。"
                 "レビューで外した人のフォルダに前回の写真が残るのを防げます。",
        )
        st.caption("出力先に同じファイル名が既にある場合は上書きされます。"
                   "設定を変えて出力し直すときは、上のチェックを入れると前回の結果が残りません。")
        if st.button("ファイルを出力", type="primary"):
            if clean_first and not dry_run and output_dir.is_dir():
                for d in output_dir.iterdir():
                    if d.is_dir():
                        shutil.rmtree(d, ignore_errors=True)
            bar = st.progress(0.0, "出力中…")
            written = skipped = 0
            for j, (i, names) in enumerate(plan.items(), 1):
                src = Path(photos[i]["path"])
                if not src.exists():
                    continue
                # 出力済みで、その後レビューで変えていない写真は飛ばす
                if only_new and photos[i].get("exported") == names:
                    skipped += 1
                    bar.progress(j / len(plan), f"{j}/{len(plan)}")
                    continue
                for n in names:
                    if not dry_run:
                        # 複数人に振り分ける写真は移動できないためコピー
                        move = (op_mode == "移動") and len(names) == 1
                        put_file(src, output_dir / n, move=move)
                    written += 1
                if not names and save_unmatched and not dry_run:
                    put_file(src, output_dir / OTHER_NAME, move=False)
                if not dry_run:
                    photos[i]["exported"] = list(names)
                bar.progress(j / len(plan), f"{j}/{len(plan)}")
            bar.empty()
            if not dry_run:
                save_state(image_dir)
            msg = (f"{written} 件を {len(counter)} フォルダに出力しました。"
                   if not dry_run
                   else f"ドライラン：{written} 件が出力対象です（書き込みなし）。")
            if skipped:
                msg += f"（出力済みの {skipped} 枚はスキップ）"
            st.success(msg)

# ---------------------------------------------------------------- 結果タブ

with tab_result:
    if not ss["photos"]:
        st.info("先に「① 解析」を実行してください。")
    else:
        import pandas as pd

        photos = ss["photos"]
        plan, scene_notes = build_plan(photos, threshold, min_face_px, top_only, min_sharp,
                                       sup, scene_mode, scene_of, int(min_support),
                                       ss["scene_over"])
        counter: dict[str, int] = defaultdict(int)
        for names in plan.values():
            for n in names:
                counter[n] += 1

        st.subheader("人物ごとの枚数")
        if counter:
            rows = []
            for name, cnt in counter.items():
                if is_scene_name(name):
                    pos, num, person = "シーン", "", name
                else:
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
                    "確定": (scene_notes.get(i)
                             or ("手動" if p.get("manual") is not None
                                 else ("観客席（一部）" if p.get("crowd") else "自動"))),
                    "シーン": (f"{SCENE_PREFIX}{scene_of[p['path']]:02d}"
                              if scene_mode != SCENE_OFF and p["path"] in scene_of
                              else "—"),
                    "判定人物": ", ".join(n if is_scene_name(n) else short_label(n)
                                      for n in plan[i]) or "—",
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
