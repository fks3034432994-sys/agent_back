import time
import os
import threading
from pathlib import Path
from dotenv import load_dotenv
import sys

PROJECT_ROOT = Path(__file__).parents[4]

load_dotenv()

# HF_HOME 可能是相对路径，按项目根解析成绝对路径；必须在导入 huggingface_hub 之前设置
_hf_home = os.getenv("HF_HOME")
if _hf_home and not Path(_hf_home).is_absolute():
    os.environ["HF_HOME"] = str((PROJECT_ROOT / _hf_home).resolve())

import cv2
import logging
from ultralytics import YOLO
from ultralytics.utils import LOGGER as _ultralytics_logger
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import HF_HUB_CACHE

# ultralytics 在 import 时把 StreamHandler 绑到当时的 stdout，redirect_stdout 拦不住
for _handler in _ultralytics_logger.handlers:
    try:
        if getattr(_handler, "stream", None) is sys.stdout:
            _handler.setStream(sys.stderr)
    except Exception:  # noqa: BLE001 - 日志流重定向失败不应影响主流程
        pass


MODEL_REPO = "Salesforce/GPA-GUI-Detector"
MODEL_FILE = "model.pt"

# 模型单例：进程内只加载一次
_model = None
_model_lock = threading.Lock()


def _cache_dirs() -> list[Path]:
    """模型缓存候选目录（去重、保持顺序）。

    包含 HF 当前生效缓存目录、默认用户缓存目录（~/.cache/huggingface/hub）以及项目内目录：
    .env 里 HF_HOME=./hf_cache 会把缓存指向项目内空目录，而模型实际下载在默认用户缓存中，
    两者都要查，否则会误判未命中并联网。
    """
    dirs = [
        Path(HF_HUB_CACHE),
        Path.home() / ".cache" / "huggingface" / "hub",
        PROJECT_ROOT / "hf_cache" / "hub",
        PROJECT_ROOT / "models",
    ]
    seen: set[str] = set()
    result: list[Path] = []
    for d in dirs:
        key = str(d).lower()
        if key not in seen:
            seen.add(key)
            result.append(d)
    return result


def _find_cached_model() -> str | None:
    """在各候选缓存目录中查找已下载的模型文件。

    除 HF 缓存机制外还直接扫描 snapshots 目录：本机缓存的 blobs 目录可能不完整
    （例如手动拷贝/部分下载），此时 hf_hub_download 的缓存校验会判定未命中，
    而 snapshot 下的实体文件其实是可用的。
    """
    dir_name = f"models--{MODEL_REPO.replace('/', '--')}"
    for base in _cache_dirs():
        try:
            matches = sorted(
                base.glob(f"{dir_name}/snapshots/*/{MODEL_FILE}"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            continue
        for m in matches:
            if m.is_file() and m.stat().st_size > 0:
                return str(m)
        direct = base / MODEL_FILE
        if direct.is_file() and direct.stat().st_size > 0:
            return str(direct)
    return None


def _resolve_model_path() -> str:
    """解析模型路径：环境变量 > 本地缓存文件 > HF 离线缓存 > 联网下载。

    优先使用本地已存在的模型，避免向 huggingface.co 发起网络请求
    （国内网络常因 WinError 10060 长时间重试后失败）。
    需首次下载或更新模型时可设置 HF_ENDPOINT（如 https://hf-mirror.com），
    或直接把 model.pt 放到本地并用 YOLO_MODEL_PATH 指定。
    """
    local = os.getenv("YOLO_MODEL_PATH")
    if local and Path(local).is_file():
        return local

    cached = _find_cached_model()
    if cached:
        return cached

    try:
        return hf_hub_download(MODEL_REPO, MODEL_FILE, local_files_only=True)
    except Exception:
        print(
            f"本地缓存未命中（HF_HUB_CACHE={HF_HUB_CACHE}，可设 YOLO_MODEL_PATH 指定本地模型），"
            "尝试联网下载...",
            file=sys.stderr,
        )
        return hf_hub_download(MODEL_REPO, MODEL_FILE)


def get_model() -> YOLO:
    """返回已加载的模型（单例，线程安全）。"""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                model_path = _resolve_model_path()
                print("开始模型加载！", file=sys.stderr)
                _model = YOLO(model_path)
                print("模型类别映射:", _model.names, file=sys.stderr)
                print("模型加载完成！", file=sys.stderr)
    return _model

def draw_label_safe(img, x1, y1, x2, y2, label, color=(0, 0, 255)):
    """
    在检测框正中央显示大号编号。
    允许遮盖图标（因为AI同时有原图）。
    字号自动适配框大小，确保数字完整显示。
    """
    h, w = img.shape[:2]
    box_h = y2 - y1
    box_w = x2 - x1

    # 字号随框大小调整，保证数字能放进框内
    font_scale = min(box_w, box_h) / 40
    font_scale = max(0.5, min(2.0, font_scale))  # 限制范围 0.5 ~ 2.0
    thickness = max(2, int(font_scale * 2))  # 粗细随字号变化

    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

    cx = (x1 + x2) // 2 - text_w // 2
    cy = (y1 + y2) // 2 + text_h // 2

    margin = 2
    if cx < margin:
        cx = margin
    if cx + text_w > w - margin:
        cx = w - text_w - margin
    if cy - text_h < margin:
        cy = text_h + margin
    if cy > h - margin:
        cy = h - margin

    cv2.rectangle(img, (cx - 4, cy - text_h - 4), (cx + text_w + 4, cy + 4), (255, 255, 255), -1)
    cv2.rectangle(img, (cx - 4, cy - text_h - 4), (cx + text_w + 4, cy + 4), (0, 0, 0), 1)
    cv2.putText(img, label, (cx, cy), font, font_scale, color, thickness)


def yolo_draw(img_path:str):
    model = get_model()

    print("开始推理...", file=sys.stderr)
    start_time = time.time()
    results = model.predict(source=img_path, conf=0.25, save=False)
    end_time = time.time()
    print(f"推理耗时: {(end_time - start_time) * 1000:.2f} 毫秒", file=sys.stderr)

    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    element_map = {}

    for i, box in enumerate(results[0].boxes):
        conf = box.conf[0].item()
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        element_id = i + 1

        element_map[element_id] = {
            "center": (center_x, center_y),
            "bbox": (x1, y1, x2, y2),
            "conf": conf
        }

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # 编号就近避让，避免压住相邻图标
        draw_label_safe(img, x1, y1, x2, y2, str(element_id))

    save_path=Path(__file__).parent/"image"/"annotated_clean.png"
    cv2.imwrite(str(save_path), img)
    print(f"标注完成，共 {len(element_map)} 个图标", file=sys.stderr)
    print(f"标注图: {save_path}", file=sys.stderr)
    print(f"元素映射（{len(element_map)} 个）:", file=sys.stderr)

    for i, (eid, info) in enumerate(element_map.items()):
        if i >= 10:
            print(f"  ... 还有 {len(element_map) - 10} 个", file=sys.stderr)
            break
        print(f"  编号 {eid}: 中心 ({info['center'][0]}, {info['center'][1]}), 置信度 {info['conf']:.2f}", file=sys.stderr)
    return element_map,w,h
if __name__ == '__main__':
    yolo_draw("img.png")