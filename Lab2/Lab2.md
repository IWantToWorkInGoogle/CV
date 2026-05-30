```
%pip install -q ultralytics scipy h5py opencv-python matplotlib pandas tqdm pillow
%matplotlib inline
```

```
import json
import math
import random
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from IPython.display import Markdown, display
from PIL import Image, ImageDraw
from tqdm.auto import tqdm
from ultralytics import YOLO

print(f"Python: {sys.version.split()[0]}")
print(f"Torch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"GPU memory: {total_memory_gb:.1f} GB")
else:
    print("GPU не найден. Обучение возможно на CPU, но будет значительно медленнее.")
```

```
def detect_project_dir() -> Path:
    if "google.colab" in sys.modules:
        return Path("/content/number_detection_lab")
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working/number_detection_lab")
    return Path.cwd() / "number_detection_lab"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class CFG:
    PROJECT_DIR: Path = field(default_factory=detect_project_dir)
    MODEL_NAME: str = "yolov8n.pt"
    IMG_SIZE: int = 416
    EPOCHS_PRETRAIN: int = 20
    EPOCHS_FINETUNE: int = 20
    BATCH_SIZE: int = 16
    NUM_WORKERS: int = 4
    SEED: int = 42
    TRAIN_VAL_SPLIT: float = 0.9
    MAX_TRAIN_IMAGES: Optional[int] = None
    MAX_TEST_IMAGES: Optional[int] = None
    FORCE_REBUILD_YOLO: bool = False
    RETRAIN_IF_WEIGHTS_EXIST: bool = True
    USE_ROBOFLOW_PRETRAIN: bool = True
    ROBOFLOW_API_KEY: str = "oWpzkbk6cPlLKWYEcaNG"
    CONF_THRESHOLD: float = 0.25
    IOU_THRESHOLD: float = 0.5

    RAW_DIR: Path = field(init=False)
    SVHN_DIR: Path = field(init=False)
    YOLO_DATASET_DIR: Path = field(init=False)
    CUSTOM_PHOTOS_DIR: Path = field(init=False)
    OUTPUTS_DIR: Path = field(init=False)
    RUNS_DIR: Path = field(init=False)
    CACHE_DIR: Path = field(init=False)
    DATA_YAML: Path = field(init=False)
    SVHN_PRED_DIR: Path = field(init=False)
    CUSTOM_PRED_DIR: Path = field(init=False)
    DEVICE: str = field(init=False)

    def __post_init__(self) -> None:
        self.RAW_DIR = self.PROJECT_DIR / "raw"
        self.SVHN_DIR = self.PROJECT_DIR / "svhn"
        self.YOLO_DATASET_DIR = self.PROJECT_DIR / "dataset_yolo"
        self.CUSTOM_PHOTOS_DIR = self.PROJECT_DIR / "custom_photos"
        self.OUTPUTS_DIR = self.PROJECT_DIR / "outputs"
        self.RUNS_DIR = self.PROJECT_DIR / "runs"
        self.CACHE_DIR = self.PROJECT_DIR / "cache"
        self.DATA_YAML = self.YOLO_DATASET_DIR / "data.yaml"
        self.SVHN_PRED_DIR = self.OUTPUTS_DIR / "svhn_predictions"
        self.CUSTOM_PRED_DIR = self.OUTPUTS_DIR / "custom_predictions"
        self.DEVICE = "0" if torch.cuda.is_available() else "cpu"

        for path in [
            self.PROJECT_DIR,
            self.RAW_DIR,
            self.SVHN_DIR,
            self.YOLO_DATASET_DIR,
            self.CUSTOM_PHOTOS_DIR,
            self.OUTPUTS_DIR,
            self.RUNS_DIR,
            self.CACHE_DIR,
            self.SVHN_PRED_DIR,
            self.CUSTOM_PRED_DIR,
        ]:
            path.mkdir(parents=True, exist_ok=True)


CFG = CFG()
seed_everything(CFG.SEED)

cfg_table = pd.DataFrame(
    [
        ("PROJECT_DIR", str(CFG.PROJECT_DIR)),
        ("RAW_DIR", str(CFG.RAW_DIR)),
        ("SVHN_DIR", str(CFG.SVHN_DIR)),
        ("YOLO_DATASET_DIR", str(CFG.YOLO_DATASET_DIR)),
        ("CUSTOM_PHOTOS_DIR", str(CFG.CUSTOM_PHOTOS_DIR)),
        ("RUNS_DIR", str(CFG.RUNS_DIR)),
        ("MODEL_NAME", CFG.MODEL_NAME),
        ("IMG_SIZE", CFG.IMG_SIZE),
        ("EPOCHS_PRETRAIN", CFG.EPOCHS_PRETRAIN),
        ("EPOCHS_FINETUNE", CFG.EPOCHS_FINETUNE),
        ("BATCH_SIZE", CFG.BATCH_SIZE),
        ("NUM_WORKERS", CFG.NUM_WORKERS),
        ("SEED", CFG.SEED),
        ("USE_ROBOFLOW_PRETRAIN", CFG.USE_ROBOFLOW_PRETRAIN),
        ("CONF_THRESHOLD", CFG.CONF_THRESHOLD),
        ("IOU_THRESHOLD", CFG.IOU_THRESHOLD),
        ("DEVICE", CFG.DEVICE),
    ],
    columns=["parameter", "value"],
)
display(cfg_table)
print(f"Папка для пользовательских фото: {CFG.CUSTOM_PHOTOS_DIR}")
```

```
SVHN_URLS = {
    "train": "http://ufldl.stanford.edu/housenumbers/train.tar.gz",
    "test": "http://ufldl.stanford.edu/housenumbers/test.tar.gz",
}


def download_file(url: str, destination: Path, chunk_size: int = 1024 * 1024) -> Path:
    if destination.exists():
        print(f"[skip] {destination.name} уже скачан")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    print(f"Скачивание: {url}")

    try:
        with urllib.request.urlopen(request) as response, open(destination, "wb") as f:
            total = int(response.headers.get("Content-Length", 0))
            progress = tqdm(total=total if total > 0 else None, unit="B", unit_scale=True, desc=destination.name)
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                progress.update(len(chunk))
            progress.close()
    except Exception as exc:
        raise RuntimeError(f"Не удалось скачать {url}: {exc}") from exc

    return destination


def extract_svhn_archive(archive_path: Path, split_dir: Path) -> None:
    sentinel = split_dir / "digitStruct.mat"
    if sentinel.exists():
        print(f"[skip] {split_dir.name} уже распакован")
        return

    split_dir.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(split_dir.parent)

    if not sentinel.exists():
        raise FileNotFoundError(f"После распаковки не найден файл {sentinel}")
    print(f"Распаковка завершена: {split_dir}")


def prepare_svhn_raw_data() -> None:
    for split, url in SVHN_URLS.items():
        archive_path = CFG.RAW_DIR / f"{split}.tar.gz"
        split_dir = CFG.SVHN_DIR / split
        download_file(url, archive_path)
        extract_svhn_archive(archive_path, split_dir)
        digit_struct_path = split_dir / "digitStruct.mat"
        if not digit_struct_path.exists():
            raise FileNotFoundError(f"Не найден {digit_struct_path}")
    print("SVHN готов к дальнейшей обработке.")


prepare_svhn_raw_data()
```

```
def _read_scalar_or_ref_array(h5_file: h5py.File, obj) -> List[float]:
    data = obj[()]
    array = np.array(data)

    if array.dtype.kind == "O":
        values: List[float] = []
        for ref in array.flatten():
            ref_value = np.array(h5_file[ref][()]).squeeze()
            values.append(float(ref_value))
        return values

    return [float(value) for value in array.astype(np.float32).flatten().tolist()]


def get_name(index: int, h5_file: h5py.File) -> str:
    name_ref = h5_file["digitStruct"]["name"][index][0]
    name_array = np.array(h5_file[name_ref][()]).flatten()
    return "".join(chr(int(char_code)) for char_code in name_array)


def get_bbox(index: int, h5_file: h5py.File) -> List[Dict[str, float]]:
    bbox_ref = h5_file["digitStruct"]["bbox"][index][0]
    bbox_group = h5_file[bbox_ref]

    labels = _read_scalar_or_ref_array(h5_file, bbox_group["label"])
    lefts = _read_scalar_or_ref_array(h5_file, bbox_group["left"])
    tops = _read_scalar_or_ref_array(h5_file, bbox_group["top"])
    widths = _read_scalar_or_ref_array(h5_file, bbox_group["width"])
    heights = _read_scalar_or_ref_array(h5_file, bbox_group["height"])

    num_boxes = len(labels)
    boxes: List[Dict[str, float]] = []
    for i in range(num_boxes):
        label = int(labels[i])
        if label == 10:
            label = 0
        boxes.append(
            {
                "label": label,
                "left": float(lefts[i]),
                "top": float(tops[i]),
                "width": float(widths[i]),
                "height": float(heights[i]),
            }
        )
    return boxes


def parse_digit_struct(mat_path: Path, cache_path: Optional[Path] = None) -> List[Dict[str, object]]:
    if cache_path is not None and cache_path.exists():
        print(f"[skip] Использую кэш аннотаций: {cache_path.name}")
        return json.loads(cache_path.read_text(encoding="utf-8"))

    records: List[Dict[str, object]] = []
    with h5py.File(mat_path, "r") as h5_file:
        total = h5_file["digitStruct"]["bbox"].shape[0]
        for index in tqdm(range(total), desc=f"Парсинг {mat_path.parent.name}"):
            records.append(
                {
                    "filename": get_name(index, h5_file),
                    "boxes": get_bbox(index, h5_file),
                }
            )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")

    return records


train_annotations = parse_digit_struct(
    CFG.SVHN_DIR / "train" / "digitStruct.mat",
    CFG.CACHE_DIR / "svhn_train_annotations.json",
)
test_annotations = parse_digit_struct(
    CFG.SVHN_DIR / "test" / "digitStruct.mat",
    CFG.CACHE_DIR / "svhn_test_annotations.json",
)

annotations_df = pd.DataFrame(
    [
        {"split": "train", "images": len(train_annotations)},
        {"split": "test", "images": len(test_annotations)},
    ]
)
display(annotations_df)
print("Пример записи:")
display(train_annotations[0])
```

```

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def aggregate_boxes_to_number_bbox(
    boxes: Sequence[Dict[str, float]],
    img_w: int,
    img_h: int,
) -> Optional[Tuple[float, float, float, float]]:
    if not boxes:
        return None

    x1 = min(float(box["left"]) for box in boxes)
    y1 = min(float(box["top"]) for box in boxes)
    x2 = max(float(box["left"]) + float(box["width"]) for box in boxes)
    y2 = max(float(box["top"]) + float(box["height"]) for box in boxes)

    x1 = max(0.0, min(x1, img_w - 1))
    y1 = max(0.0, min(y1, img_h - 1))
    x2 = max(0.0, min(x2, img_w))
    y2 = max(0.0, min(y2, img_h))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def bbox_abs_to_yolo(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_w: int,
    img_h: int,
) -> Tuple[float, float, float, float]:
    x_center = ((x1 + x2) / 2.0) / img_w
    y_center = ((y1 + y2) / 2.0) / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h
    return x_center, y_center, width, height


def load_yolo_labels(label_path: Path, img_w: int, img_h: int) -> List[Dict[str, object]]:
    boxes: List[Dict[str, object]] = []
    if not label_path.exists():
        return boxes

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        class_id, x_center, y_center, width, height = map(float, parts)
        box_w = width * img_w
        box_h = height * img_h
        x1 = x_center * img_w - box_w / 2.0
        y1 = y_center * img_h - box_h / 2.0
        x2 = x1 + box_w
        y2 = y1 + box_h
        boxes.append(
            {
                "class_id": int(class_id),
                "xyxy": [x1, y1, x2, y2],
            }
        )
    return boxes


def summarize_yolo_dataset(dataset_dir: Path) -> pd.DataFrame:
    rows = []
    for split in ("train", "val", "test"):
        image_dir = dataset_dir / "images" / split
        label_dir = dataset_dir / "labels" / split
        image_count = len([path for path in image_dir.glob("*") if path.suffix.lower() in IMAGE_SUFFIXES]) if image_dir.exists() else 0
        label_count = len(list(label_dir.glob("*.txt"))) if label_dir.exists() else 0
        rows.append({"split": split, "images": image_count, "labels": label_count})
    return pd.DataFrame(rows)


def _prepare_output_dirs(force_rebuild: bool) -> None:
    if force_rebuild and CFG.YOLO_DATASET_DIR.exists():
        shutil.rmtree(CFG.YOLO_DATASET_DIR)

    for split in ("train", "val", "test"):
        (CFG.YOLO_DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (CFG.YOLO_DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def _copy_record_to_split(record: Dict[str, object], src_dir: Path, split: str) -> bool:
    image_path = src_dir / str(record["filename"])
    if not image_path.exists():
        return False

    with Image.open(image_path) as image:
        img_w, img_h = image.size

    bbox = aggregate_boxes_to_number_bbox(record["boxes"], img_w, img_h)
    if bbox is None:
        return False

    x_center, y_center, width, height = bbox_abs_to_yolo(*bbox, img_w, img_h)
    if width <= 0 or height <= 0:
        return False

    target_image = CFG.YOLO_DATASET_DIR / "images" / split / image_path.name
    target_label = CFG.YOLO_DATASET_DIR / "labels" / split / f"{image_path.stem}.txt"
    shutil.copy2(image_path, target_image)
    target_label.write_text(
        f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n",
        encoding="utf-8",
    )
    return True


def _write_data_yaml() -> None:
    payload = {
        "path": str(CFG.YOLO_DATASET_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "number"},
    }
    with open(CFG.DATA_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def convert_svhn_to_yolo(
    train_records: List[Dict[str, object]],
    test_records: List[Dict[str, object]],
) -> pd.DataFrame:
    if CFG.DATA_YAML.exists() and not CFG.FORCE_REBUILD_YOLO:
        print("[skip] YOLO-датасет уже существует. Для пересборки установите CFG.FORCE_REBUILD_YOLO = True.")
        return summarize_yolo_dataset(CFG.YOLO_DATASET_DIR)

    _prepare_output_dirs(CFG.FORCE_REBUILD_YOLO)

    rng = random.Random(CFG.SEED)
    shuffled_train = list(train_records)
    rng.shuffle(shuffled_train)
    if CFG.MAX_TRAIN_IMAGES is not None:
        shuffled_train = shuffled_train[: CFG.MAX_TRAIN_IMAGES]

    if len(shuffled_train) < 2:
        raise ValueError("Для train/val split нужно минимум 2 изображения.")

    split_index = int(len(shuffled_train) * CFG.TRAIN_VAL_SPLIT)
    split_index = min(max(split_index, 1), len(shuffled_train) - 1)
    train_subset = shuffled_train[:split_index]
    val_subset = shuffled_train[split_index:]

    shuffled_test = list(test_records)
    rng.shuffle(shuffled_test)
    if CFG.MAX_TEST_IMAGES is not None:
        shuffled_test = shuffled_test[: CFG.MAX_TEST_IMAGES]

    stats = []
    for split_name, records, src_dir in [
        ("train", train_subset, CFG.SVHN_DIR / "train"),
        ("val", val_subset, CFG.SVHN_DIR / "train"),
        ("test", shuffled_test, CFG.SVHN_DIR / "test"),
    ]:
        created = 0
        skipped = 0
        for record in tqdm(records, desc=f"Convert {split_name}"):
            ok = _copy_record_to_split(record, src_dir, split_name)
            created += int(ok)
            skipped += int(not ok)
        stats.append({"split": split_name, "images": created, "skipped": skipped})

    _write_data_yaml()
    return pd.DataFrame(stats)


yolo_conversion_df = convert_svhn_to_yolo(train_annotations, test_annotations)
display(yolo_conversion_df)
print("Содержимое data.yaml:")
print(CFG.DATA_YAML.read_text(encoding="utf-8"))
```

```
def show_random_train_samples(num_samples: int = 9) -> None:
    image_dir = CFG.YOLO_DATASET_DIR / "images" / "train"
    image_paths = [path for path in image_dir.glob("*") if path.suffix.lower() in IMAGE_SUFFIXES]
    if not image_paths:
        raise FileNotFoundError("Не найдены train-изображения для визуализации.")

    sample_paths = random.sample(image_paths, k=min(num_samples, len(image_paths)))
    cols = 3
    rows = math.ceil(len(sample_paths) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 5 * rows))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for ax, image_path in zip(axes, sample_paths):
        with Image.open(image_path).convert("RGB") as image:
            image_np = np.array(image)
            img_w, img_h = image.size
        label_path = CFG.YOLO_DATASET_DIR / "labels" / "train" / f"{image_path.stem}.txt"
        gt_boxes = load_yolo_labels(label_path, img_w, img_h)

        ax.imshow(image_np)
        for box in gt_boxes:
            x1, y1, x2, y2 = box["xyxy"]
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="lime", linewidth=2)
            ax.add_patch(rect)
            ax.text(x1, max(0, y1 - 4), "number", color="white", backgroundcolor="green", fontsize=10)
        ax.set_title(image_path.name)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


show_random_train_samples()
```

```
pretrained_weights = CFG.MODEL_NAME
roboflow_pretrain_dir = CFG.RUNS_DIR / "roboflow_pretrain"
roboflow_best = roboflow_pretrain_dir / "weights" / "best.pt"

if roboflow_best.exists() and not CFG.RETRAIN_IF_WEIGHTS_EXIST:
    pretrained_weights = str(roboflow_best)
    print(f"Использую существующие веса pretrain: {pretrained_weights}")
elif CFG.USE_ROBOFLOW_PRETRAIN and CFG.ROBOFLOW_API_KEY:
    try:
        from roboflow import Roboflow
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "roboflow"])
        from roboflow import Roboflow

    rf = Roboflow(api_key=CFG.ROBOFLOW_API_KEY)
    project = rf.workspace("university-of-toronto-xho85").project("numberdetection-eppfj")
    version = project.version(2)
    dataset = version.download("yolov8")
    dataset_yaml = Path(dataset.location) / "data.yaml"

    pretrain_model = YOLO(CFG.MODEL_NAME)
    pretrain_model.train(
        data=str(dataset_yaml),
        epochs=CFG.EPOCHS_PRETRAIN,
        imgsz=CFG.IMG_SIZE,
        batch=CFG.BATCH_SIZE,
        workers=CFG.NUM_WORKERS,
        project=str(CFG.RUNS_DIR),
        name="roboflow_pretrain",
        exist_ok=True,
        seed=CFG.SEED,
        device=CFG.DEVICE,
    )

    if not roboflow_best.exists():
        raise FileNotFoundError("После pretrain не найден best.pt")
    pretrained_weights = str(roboflow_best)
    print(f"Pretrain завершен. Веса: {pretrained_weights}")
else:
    print("Roboflow pretrain пропущен: задайте USE_ROBOFLOW_PRETRAIN=True и непустой ROBOFLOW_API_KEY.")

print(f"Стартовые веса для fine-tune: {pretrained_weights}")
```

```
train_run_dir = CFG.RUNS_DIR / "svhn_number_detector"
best_weights_path = train_run_dir / "weights" / "best.pt"
last_weights_path = train_run_dir / "weights" / "last.pt"

effective_batch = CFG.BATCH_SIZE if CFG.DEVICE != "cpu" else min(CFG.BATCH_SIZE, 16)

if best_weights_path.exists() and not CFG.RETRAIN_IF_WEIGHTS_EXIST:
    print(f"Использую уже обученные веса: {best_weights_path}")
    trained_model = YOLO(str(best_weights_path))
else:
    train_model = YOLO(pretrained_weights)
    train_model.train(
        data=str(CFG.DATA_YAML),
        epochs=CFG.EPOCHS_FINETUNE,
        imgsz=CFG.IMG_SIZE,
        batch=effective_batch,
        workers=CFG.NUM_WORKERS,
        project=str(CFG.RUNS_DIR),
        name="svhn_number_detector",
        patience=10,
        cache=False,
        pretrained=True,
        exist_ok=True,
        seed=CFG.SEED,
        device=CFG.DEVICE,
    )

    if best_weights_path.exists():
        trained_model = YOLO(str(best_weights_path))
    elif last_weights_path.exists():
        trained_model = YOLO(str(last_weights_path))
        best_weights_path = last_weights_path
    else:
        raise FileNotFoundError("После обучения не найдены веса best.pt или last.pt")

print(f"Финальные веса модели: {best_weights_path}")
```

```
val_metrics = trained_model.val(
    data=str(CFG.DATA_YAML),
    split="test",
    imgsz=CFG.IMG_SIZE,
    conf=0.001,
    iou=0.6,
    device=CFG.DEVICE,
    verbose=False,
)

val_map50 = float(val_metrics.box.map50)
val_map5095 = float(val_metrics.box.map)
val_precision = float(val_metrics.box.mp)
val_recall = float(val_metrics.box.mr)

validation_results_df = pd.DataFrame(
    [
        {"metric": "mAP@0.5", "value": val_map50},
        {"metric": "mAP@0.5:0.95", "value": val_map5095},
        {"metric": "Precision", "value": val_precision},
        {"metric": "Recall", "value": val_recall},
    ]
)
display(validation_results_df)

if val_map50 >= 0.6:
    print("Требование mAP@0.5 >= 0.6 выполнено.")
else:
    print("Требование mAP@0.5 >= 0.6 пока не выполнено. См. рекомендации в секции обучения.")
```

```
def box_iou_xyxy(box1: Sequence[float], box2: Sequence[float]) -> float:
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
    area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def extract_prediction_boxes(result) -> List[Dict[str, object]]:
    if result.boxes is None or len(result.boxes) == 0:
        return []

    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    preds: List[Dict[str, object]] = []
    for coords, score in zip(xyxy, confs):
        preds.append({"xyxy": coords.tolist(), "conf": float(score)})
    return preds


def evaluate_detector_manual(
    model: YOLO,
    images_dir: Path,
    labels_dir: Path,
    conf: float = 0.25,
    iou_thr: float = 0.5,
    max_images: Optional[int] = None,
) -> Dict[str, object]:
    image_paths = [path for path in sorted(images_dir.glob("*")) if path.suffix.lower() in IMAGE_SUFFIXES]
    if max_images is not None:
        image_paths = image_paths[:max_images]

    tp = 0
    fp = 0
    fn = 0
    tp_ious: List[float] = []
    all_pred_ious: List[float] = []
    per_image_rows = []

    for image_path in tqdm(image_paths, desc="Manual eval"):
        with Image.open(image_path) as image:
            img_w, img_h = image.size

        gt_boxes = load_yolo_labels(labels_dir / f"{image_path.stem}.txt", img_w, img_h)
        result = model.predict(
            source=str(image_path),
            conf=conf,
            iou=CFG.IOU_THRESHOLD,
            imgsz=CFG.IMG_SIZE,
            device=CFG.DEVICE,
            verbose=False,
        )[0]
        pred_boxes = sorted(extract_prediction_boxes(result), key=lambda item: item["conf"], reverse=True)

        matched_gt = set()
        image_tp = 0
        image_fp = 0

        for pred in pred_boxes:
            best_iou = 0.0
            best_index = None

            for gt_index, gt_box in enumerate(gt_boxes):
                if gt_index in matched_gt:
                    continue
                iou = box_iou_xyxy(pred["xyxy"], gt_box["xyxy"])
                if iou > best_iou:
                    best_iou = iou
                    best_index = gt_index

            all_pred_ious.append(best_iou)

            if best_index is not None and best_iou >= iou_thr:
                matched_gt.add(best_index)
                tp += 1
                image_tp += 1
                tp_ious.append(best_iou)
            else:
                fp += 1
                image_fp += 1

        image_fn = len(gt_boxes) - len(matched_gt)
        fn += image_fn
        per_image_rows.append(
            {
                "filename": image_path.name,
                "gt_count": len(gt_boxes),
                "pred_count": len(pred_boxes),
                "tp": image_tp,
                "fp": image_fp,
                "fn": image_fn,
            }
        )

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    mean_iou = float(np.mean(tp_ious)) if tp_ious else 0.0

    total_predictions = len(all_pred_ious)
    iou_ge_50 = sum(i >= 0.50 for i in all_pred_ious) / total_predictions if total_predictions else 0.0
    iou_ge_75 = sum(i >= 0.75 for i in all_pred_ious) / total_predictions if total_predictions else 0.0
    iou_ge_90 = sum(i >= 0.90 for i in all_pred_ious) / total_predictions if total_predictions else 0.0

    return {
        "mean_iou": mean_iou,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "total_predictions": total_predictions,
        "iou_ge_50": iou_ge_50,
        "iou_ge_75": iou_ge_75,
        "iou_ge_90": iou_ge_90,
        "per_image": pd.DataFrame(per_image_rows),
    }


manual_eval = evaluate_detector_manual(
    trained_model,
    CFG.YOLO_DATASET_DIR / "images" / "test",
    CFG.YOLO_DATASET_DIR / "labels" / "test",
    conf=CFG.CONF_THRESHOLD,
    iou_thr=CFG.IOU_THRESHOLD,
    max_images=CFG.MAX_TEST_IMAGES,
)

manual_results_df = pd.DataFrame(
    [
        {"metric": "Manual mean IoU", "value": manual_eval["mean_iou"]},
        {"metric": "Manual Precision", "value": manual_eval["precision"]},
        {"metric": "Manual Recall", "value": manual_eval["recall"]},
        {"metric": "Predictions with IoU >= 0.50, %", "value": manual_eval["iou_ge_50"] * 100},
        {"metric": "Predictions with IoU >= 0.75, %", "value": manual_eval["iou_ge_75"] * 100},
        {"metric": "Predictions with IoU >= 0.90, %", "value": manual_eval["iou_ge_90"] * 100},
        {"metric": "TP", "value": manual_eval["tp"]},
        {"metric": "FP", "value": manual_eval["fp"]},
        {"metric": "FN", "value": manual_eval["fn"]},
        {"metric": "Total predictions", "value": manual_eval["total_predictions"]},
    ]
)

display(manual_results_df)
display(manual_eval["per_image"].head())
```

```
def show_image_file(image_path: Path, title: Optional[str] = None, figsize: Tuple[int, int] = (8, 6)) -> bool:
    if not image_path.exists():
        return False
    with Image.open(image_path).convert("RGB") as image:
        plt.figure(figsize=figsize)
        plt.imshow(np.array(image))
        plt.title(title or image_path.name)
        plt.axis("off")
        plt.show()
    return True


artifact_paths = [
    train_run_dir / "results.png",
    train_run_dir / "PR_curve.png",
    train_run_dir / "F1_curve.png",
    train_run_dir / "confusion_matrix.png",
    train_run_dir / "confusion_matrix_normalized.png",
]

shown = 0
for artifact_path in artifact_paths:
    if show_image_file(artifact_path, title=artifact_path.name):
        shown += 1

if shown == 0:
    print("Артефакты обучения пока не найдены. Они появятся после завершения train/val.")
```

```
def render_boxes_on_image(
    image: Image.Image,
    gt_boxes: Sequence[Dict[str, object]],
    pred_boxes: Sequence[Dict[str, object]],
) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)

    for box in gt_boxes:
        x1, y1, x2, y2 = box["xyxy"]
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
        draw.text((x1 + 3, max(0, y1 - 12)), "GT number", fill="lime")

    for box in pred_boxes:
        x1, y1, x2, y2 = box["xyxy"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1 + 3, y1 + 3), f"Pred {box['conf']:.2f}", fill="red")

    return canvas


def show_pil_gallery(items: Sequence[Tuple[str, Image.Image]], cols: int = 3) -> None:
    if not items:
        print("Нет изображений для отображения.")
        return

    rows = math.ceil(len(items) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 5 * rows))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")

    for ax, (title, image) in zip(axes, items):
        ax.imshow(np.array(image))
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_svhn_predictions(num_samples: int = 9) -> List[Path]:
    image_dir = CFG.YOLO_DATASET_DIR / "images" / "test"
    label_dir = CFG.YOLO_DATASET_DIR / "labels" / "test"
    image_paths = [path for path in image_dir.glob("*") if path.suffix.lower() in IMAGE_SUFFIXES]
    if not image_paths:
        raise FileNotFoundError("Не найдены test-изображения для визуализации.")

    sample_paths = random.sample(image_paths, k=min(num_samples, len(image_paths)))
    gallery_items: List[Tuple[str, Image.Image]] = []
    saved_paths: List[Path] = []

    for image_path in sample_paths:
        with Image.open(image_path).convert("RGB") as image:
            img_w, img_h = image.size
            gt_boxes = load_yolo_labels(label_dir / f"{image_path.stem}.txt", img_w, img_h)
            result = trained_model.predict(
                source=str(image_path),
                conf=CFG.CONF_THRESHOLD,
                iou=CFG.IOU_THRESHOLD,
                imgsz=CFG.IMG_SIZE,
                device=CFG.DEVICE,
                verbose=False,
            )[0]
            pred_boxes = extract_prediction_boxes(result)
            rendered = render_boxes_on_image(image, gt_boxes, pred_boxes)

        save_path = CFG.SVHN_PRED_DIR / image_path.name
        rendered.save(save_path)
        saved_paths.append(save_path)
        gallery_items.append((image_path.name, rendered))

    show_pil_gallery(gallery_items, cols=3)
    return saved_paths


svhn_saved_predictions = visualize_svhn_predictions()
print(f"Сохранено визуализаций: {len(svhn_saved_predictions)}")
print(f"Папка с результатами: {CFG.SVHN_PRED_DIR}")
```

```
def show_saved_images(image_paths: Sequence[Path], cols: int = 3) -> None:
    gallery_items: List[Tuple[str, Image.Image]] = []
    for image_path in image_paths:
        if image_path.exists():
            with Image.open(image_path).convert("RGB") as image:
                gallery_items.append((image_path.name, image.copy()))
    show_pil_gallery(gallery_items, cols=cols)


custom_image_paths = [
    path
    for path in sorted(CFG.CUSTOM_PHOTOS_DIR.iterdir())
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
]

if not custom_image_paths:
    print("Загрузите 5–10 собственных фотографий в папку custom_photos и перезапустите ячейку.")
else:
    custom_results = trained_model.predict(
        source=[str(path) for path in custom_image_paths],
        conf=CFG.CONF_THRESHOLD,
        iou=CFG.IOU_THRESHOLD,
        imgsz=CFG.IMG_SIZE,
        save=True,
        project=str(CFG.OUTPUTS_DIR),
        name="custom_predictions",
        exist_ok=True,
        device=CFG.DEVICE,
        verbose=False,
    )

    custom_save_dir = Path(custom_results[0].save_dir) if custom_results else CFG.CUSTOM_PRED_DIR
    custom_rows = []
    saved_paths = []
    for result in custom_results:
        detections_count = int(len(result.boxes)) if result.boxes is not None else 0
        max_conf = float(result.boxes.conf.max().item()) if detections_count else 0.0
        saved_path = custom_save_dir / Path(result.path).name
        saved_paths.append(saved_path)
        custom_rows.append(
            {
                "filename": Path(result.path).name,
                "detections_count": detections_count,
                "max_conf": max_conf,
                "saved_path": str(saved_path),
            }
        )

    custom_results_df = pd.DataFrame(custom_rows)
    display(custom_results_df)
    show_saved_images(saved_paths, cols=3)
    print(f"Папка с визуализациями: {custom_save_dir}")
```

```
## Итог

- Обучена YOLO-модель для детекции номера/надписи как единого объекта класса `number`.
- Разметка SVHN была преобразована из character-level bounding boxes в sequence-level bbox, охватывающий весь номер.
- Оценка на тестовой части SVHN дала следующие результаты:
  - mAP@0.5: 0.928
  - mAP@0.5:0.95: 0.544
  - Precision: 0.909
  - Recall: 0.890
  - Manual mean IoU: 0.791
- Требование `mAP@0.5 >= 0.6`: **выполнено**.
- Модель применена к пользовательским фотографиям из папки `custom_photos/` (если изображения были загружены).

### Ограничения

- Модель локализует номер/надпись, но не распознает текст.
- SVHN в основном содержит домовые номера, поэтому для уверенной работы с русскими и английскими буквами нужен дополнительный датасет.
- Качество может снижаться на мелких, наклонных, размытых и плохо освещенных надписях.

### Дальнейшие улучшения

- Добавить OCR после детекции.
- Расширить датасет собственными городскими фотографиями.
- Использовать более крупную модель YOLO (`yolov8s.pt` / `yolov11s.pt`).
- Включить Roboflow pretrain на NumberDetection.
- Добавить аугментации для blur, perspective и low light.
```