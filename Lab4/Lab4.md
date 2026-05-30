```
import importlib.util
import subprocess
import sys

REQUIRED_PACKAGES = {
    "facenet_pytorch": "facenet-pytorch",
    "torchmetrics": "torchmetrics[image]",
    "torch_fidelity": "torch-fidelity",
}

for module_name, install_name in REQUIRED_PACKAGES.items():
    if importlib.util.find_spec(module_name) is None:
        print(f"Installing {install_name} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", install_name])
```

```
from pathlib import Path
import os
import random
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid

from facenet_pytorch import MTCNN
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

plt.style.use("seaborn-v0_8")
pd.set_option("display.max_columns", 10)
```

```
PROJECT_ROOT = Path.cwd()
DATA_ROOT = PROJECT_ROOT

IMAGE_SIZE = 64
BATCH_SIZE = 32
LATENT_DIM = 128
NUM_EPOCHS = 20
NUM_EPOCHS_COND = 20
LR_G = 1e-4
LR_C = 1e-4
BETAS = (0.0, 0.9)
N_CRITIC = 5
LAMBDA_GP = 10.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

USE_ALIGNED_CELEBA_DIRECTLY = True
MAX_PREPROCESS_IMAGES = 200_000
MAX_TRAIN_IMAGES = 50_000
NUM_METRIC_IMAGES = 1_000

RUN_METRICS = True
USE_AMP = False
SAMPLE_EVERY = 2
NUM_WORKERS = 0

AMP_ENABLED = USE_AMP and DEVICE.type == "cuda"
PIN_MEMORY = DEVICE.type == "cuda"

ARCHIVE_PATH = DATA_ROOT / "archive.zip"
PROCESSED_DIR = DATA_ROOT / "processed_faces"
CHECKPOINT_DIR = DATA_ROOT / "checkpoints"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)
print(f"Using device: {DEVICE}")
print(f"DATA_ROOT: {DATA_ROOT}")
print(f"Using aligned CelebA directly: {USE_ALIGNED_CELEBA_DIRECTLY}")
```

```
def maybe_extract_celeba(archive_path: Path, extract_to: Path) -> None:
    image_dir_candidate = extract_to / "img_align_celeba" / "img_align_celeba"
    attr_candidate_csv = extract_to / "list_attr_celeba.csv"
    attr_candidate_txt = extract_to / "list_attr_celeba.txt"

    already_extracted = image_dir_candidate.exists() and (
        attr_candidate_csv.exists() or attr_candidate_txt.exists()
    )
    if already_extracted:
        print("CelebA already extracted. Reusing existing files.")
        return

    if not archive_path.exists():
        print("archive.zip not found. Skipping automatic extraction.")
        return

    print(f"Extracting {archive_path.name} to {extract_to} ...")
    with zipfile.ZipFile(archive_path, "r") as zip_file:
        zip_file.extractall(extract_to)
    print("Extraction finished.")


def resolve_celeba_paths(data_root: Path):
    image_candidates = [
        data_root / "img_align_celeba" / "img_align_celeba",
        data_root / "img_align_celeba",
        data_root / "celeba" / "img_align_celeba",
    ]
    attr_candidates = [
        data_root / "list_attr_celeba.csv",
        data_root / "list_attr_celeba.txt",
        data_root / "celeba" / "list_attr_celeba.csv",
        data_root / "celeba" / "list_attr_celeba.txt",
    ]

    image_dir = next((path for path in image_candidates if path.exists()), None)
    attr_path = next((path for path in attr_candidates if path.exists()), None)

    if image_dir is None:
        image_matches = [path for path in data_root.rglob("img_align_celeba") if path.is_dir()]
        if image_matches:
            image_dir = image_matches[0]

    if attr_path is None:
        attr_matches = list(data_root.rglob("list_attr_celeba.csv"))
        attr_matches += list(data_root.rglob("list_attr_celeba.txt"))
        if attr_matches:
            attr_path = attr_matches[0]

    return image_dir, attr_path


maybe_extract_celeba(ARCHIVE_PATH, DATA_ROOT)
RAW_IMAGE_DIR, ATTR_PATH = resolve_celeba_paths(DATA_ROOT)

print(f"RAW_IMAGE_DIR: {RAW_IMAGE_DIR}")
print(f"ATTR_PATH: {ATTR_PATH}")
```

```
RESAMPLE = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS


def preprocess_faces(
    source_dir: Path,
    output_dir: Path,
    image_size: int = 64,
    max_images: int = None,
    device: torch.device = DEVICE,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_files = sorted(output_dir.glob("*.jpg"))
    if existing_files:
        print(
            f"Found {len(existing_files):,} preprocessed images in {output_dir}. "
            "Reusing them. Delete the folder if you want to preprocess again."
        )
        return len(existing_files)

    source_files = sorted(source_dir.glob("*.jpg"))
    if max_images is not None:
        source_files = source_files[:max_images]

    if len(source_files) == 0:
        raise RuntimeError(f"No source images found in {source_dir}")

    mtcnn = MTCNN(keep_all=False, device=device)
    saved_count = 0
    skipped_count = 0

    for image_path in tqdm(source_files, desc="Preprocessing faces"):
        try:
            with Image.open(image_path) as pil_image:
                image = pil_image.convert("RGB")
        except Exception:
            skipped_count += 1
            continue

        boxes, probs = mtcnn.detect(image)
        if boxes is None or probs is None:
            skipped_count += 1
            continue

        best_idx = int(np.argmax(probs))
        x1, y1, x2, y2 = boxes[best_idx]
        width, height = image.size

        x1 = int(max(0, np.floor(x1)))
        y1 = int(max(0, np.floor(y1)))
        x2 = int(min(width, np.ceil(x2)))
        y2 = int(min(height, np.ceil(y2)))

        if x2 <= x1 or y2 <= y1:
            skipped_count += 1
            continue

        face = image.crop((x1, y1, x2, y2)).resize((image_size, image_size), RESAMPLE)
        face.save(output_dir / image_path.name, quality=95)
        saved_count += 1

    print(f"Saved {saved_count:,} cropped faces.")
    print(f"Skipped {skipped_count:,} images without a valid detected face.")
    return saved_count


def prepare_training_images(
    source_dir: Path,
    output_dir: Path,
    use_aligned_directly: bool = True,
    image_size: int = 64,
    max_images: int = None,
    device: torch.device = DEVICE,
):
    source_files = sorted(source_dir.glob("*.jpg"))
    if not source_files:
        raise RuntimeError(f"No source images found in {source_dir}")

    if use_aligned_directly:
        usable_count = len(source_files[:max_images]) if max_images is not None else len(source_files)
        print("Using aligned CelebA images directly. Skipping extra MTCNN crop.")
        return source_dir, usable_count

    processed_count = preprocess_faces(
        source_dir=source_dir,
        output_dir=output_dir,
        image_size=image_size,
        max_images=max_images,
        device=device,
    )
    return output_dir, processed_count
```

```
TRAIN_IMAGE_DIR, train_image_count = prepare_training_images(
    source_dir=RAW_IMAGE_DIR,
    output_dir=PROCESSED_DIR,
    use_aligned_directly=USE_ALIGNED_CELEBA_DIRECTLY,
    image_size=IMAGE_SIZE,
    max_images=MAX_PREPROCESS_IMAGES,
    device=DEVICE,
)

print(f"Training image directory: {TRAIN_IMAGE_DIR}")
print(f"Images available for training: {train_image_count:,}")
```

```
def load_attributes(attr_path: Path) -> pd.DataFrame:
    if attr_path.suffix.lower() == ".csv":
        attr_df = pd.read_csv(attr_path)
        filename_column = "image_id" if "image_id" in attr_df.columns else attr_df.columns[0]
        attr_df = attr_df.rename(columns={filename_column: "image_name"}).set_index("image_name")
    else:
        attr_df = pd.read_csv(attr_path, sep=r"\s+", skiprows=1)
        filename_column = attr_df.columns[0]
        attr_df = attr_df.rename(columns={filename_column: "image_name"}).set_index("image_name")

    if "Male" not in attr_df.columns:
        raise KeyError("Column `Male` was not found in the CelebA attributes file.")

    attr_df.index = attr_df.index.astype(str)
    attr_df["Male"] = (attr_df["Male"].astype(int) == 1).astype(int)
    return attr_df


def build_metadata(image_dir: Path, attr_df: pd.DataFrame, max_train_images: int = None, seed: int = 42) -> pd.DataFrame:
    available_images = sorted(path.name for path in Path(image_dir).glob("*.jpg"))
    if not available_images:
        raise RuntimeError(
            "Не найдено ни одного изображения для обучения. Проверьте RAW_IMAGE_DIR или папку preprocessing."
        )

    metadata = pd.DataFrame({"image_name": available_images})
    metadata = metadata.merge(attr_df[["Male"]], left_on="image_name", right_index=True, how="inner")

    if metadata.empty:
        raise RuntimeError(
            "После объединения с атрибутами не осталось изображений. "
            "Проверьте соответствие имен файлов в image_dir и в list_attr_celeba."
        )

    if max_train_images is not None and len(metadata) > max_train_images:
        metadata = (
            metadata.sample(n=max_train_images, random_state=seed)
            .sort_values("image_name")
            .reset_index(drop=True)
        )
    else:
        metadata = metadata.reset_index(drop=True)

    return metadata


image_transform_steps = []
if USE_ALIGNED_CELEBA_DIRECTLY:
    image_transform_steps.append(transforms.CenterCrop(178))

image_transform_steps.extend([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
image_transform = transforms.Compose(image_transform_steps)


class FacesDataset(Dataset):
    def __init__(self, image_dir: Path, metadata: pd.DataFrame, transform=None):
        self.image_dir = Path(image_dir)
        self.metadata = metadata.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx: int):
        row = self.metadata.iloc[idx]
        image_path = self.image_dir / row["image_name"]
        with Image.open(image_path) as pil_image:
            image = pil_image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image


class ConditionalFacesDataset(Dataset):
    def __init__(self, image_dir: Path, metadata: pd.DataFrame, transform=None):
        self.image_dir = Path(image_dir)
        self.metadata = metadata.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx: int):
        row = self.metadata.iloc[idx]
        image_path = self.image_dir / row["image_name"]
        with Image.open(image_path) as pil_image:
            image = pil_image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(int(row["Male"]), dtype=torch.long)
        return image, label


attr_df = load_attributes(ATTR_PATH)
metadata = build_metadata(TRAIN_IMAGE_DIR, attr_df, max_train_images=MAX_TRAIN_IMAGES, seed=SEED)

uncond_dataset = FacesDataset(TRAIN_IMAGE_DIR, metadata, transform=image_transform)
cond_dataset = ConditionalFacesDataset(TRAIN_IMAGE_DIR, metadata, transform=image_transform)

uncond_loader = DataLoader(
    uncond_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    drop_last=True,
)

cond_loader = DataLoader(
    cond_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    drop_last=True,
)

uncond_eval_loader = DataLoader(
    uncond_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    drop_last=False,
)

cond_eval_loader = DataLoader(
    cond_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    drop_last=False,
)

metadata.head()
```

```
print(f"Images after attribute join: {len(metadata):,}")
class_distribution = metadata["Male"].value_counts().sort_index()
class_distribution.index = class_distribution.index.map({0: "female", 1: "male"})
print(class_distribution)
```

```
def denormalize(images: torch.Tensor) -> torch.Tensor:
    return (images * 0.5 + 0.5).clamp(0, 1)


def show_tensor_batch(images: torch.Tensor, title: str = None, nrow: int = 8, figsize=(8, 8)) -> None:
    grid = make_grid(denormalize(images.detach().cpu()), nrow=nrow)
    plt.figure(figsize=figsize)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.show()


real_batch = next(iter(uncond_loader))
show_tensor_batch(real_batch[:32], title="Real CelebA faces", nrow=8, figsize=(10, 10))

female_examples = metadata[metadata["Male"] == 0].head(8)["image_name"].tolist()
male_examples = metadata[metadata["Male"] == 1].head(8)["image_name"].tolist()


def load_example_tensors(file_names):
    tensors = []
    for file_name in file_names:
        with Image.open(PROCESSED_DIR / file_name) as pil_image:
            image = pil_image.convert("RGB")
        tensors.append(image_transform(image))
    return torch.stack(tensors)


if female_examples:
    show_tensor_batch(load_example_tensors(female_examples), title="Female examples", nrow=len(female_examples), figsize=(16, 3))

if male_examples:
    show_tensor_batch(load_example_tensors(male_examples), title="Male examples", nrow=len(male_examples), figsize=(16, 3))
```

```
def weights_init(module: nn.Module) -> None:
    classname = module.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif "BatchNorm" in classname:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 128, img_channels: int = 3, base_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, base_channels * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels, img_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Critic(nn.Module):
    def __init__(self, img_channels: int = 3, base_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, base_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images).view(-1)


def sample_noise(batch_size: int, latent_dim: int, device: torch.device) -> torch.Tensor:
    return torch.randn(batch_size, latent_dim, 1, 1, device=device)


def gradient_penalty(critic: nn.Module, real_images: torch.Tensor, fake_images: torch.Tensor, device: torch.device) -> torch.Tensor:
    batch_size = real_images.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = alpha * real_images.float() + (1 - alpha) * fake_images.float().detach()
    interpolated.requires_grad_(True)

    mixed_scores = critic(interpolated)
    gradients = torch.autograd.grad(
        outputs=mixed_scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


@torch.no_grad()
def generate_samples(generator: nn.Module, num_samples: int = 16, latent_dim: int = 128, device: torch.device = DEVICE, noise: torch.Tensor = None) -> torch.Tensor:
    was_training = generator.training
    generator.eval()
    if noise is None:
        noise = sample_noise(num_samples, latent_dim, device)
    fake_images = generator(noise)
    generator.train(was_training)
    return fake_images


def show_tensor_images(images: torch.Tensor, title: str = None, nrow: int = 8, figsize=(8, 8)) -> None:
    grid = make_grid(denormalize(images.detach().cpu()), nrow=nrow)
    plt.figure(figsize=figsize)
    plt.imshow(grid.permute(1, 2, 0))
    if title is not None:
        plt.title(title)
    plt.axis("off")
    plt.show()
```

```
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA available:", torch.cuda.is_available())
if DEVICE.type == "cuda":
    print(torch.cuda.get_device_name(0))
else:
    print("Using CPU")
```

```
def train_wgan_gp(
    generator: nn.Module,
    critic: nn.Module,
    dataloader: DataLoader,
    generator_optimizer: optim.Optimizer,
    critic_optimizer: optim.Optimizer,
    num_epochs: int,
    latent_dim: int,
    device: torch.device,
    n_critic: int = 5,
    lambda_gp: float = 10.0,
    sample_every: int = 2,
    checkpoint_dir: Path = CHECKPOINT_DIR,
    prefix: str = "unconditional",
):
    history = {
        "step_generator": [],
        "step_critic": [],
        "step_gp": [],
        "step_wasserstein": [],
        "epoch_generator": [],
        "epoch_critic": [],
        "epoch_gp": [],
        "epoch_wasserstein": [],
    }

    best_wasserstein = float("-inf")
    generator_scaler = GradScaler(enabled=AMP_ENABLED)
    critic_scaler = GradScaler(enabled=AMP_ENABLED)
    fixed_noise = sample_noise(16, latent_dim, device)

    for epoch in range(1, num_epochs + 1):
        generator.train()
        critic.train()

        epoch_generator_losses = []
        epoch_critic_losses = []
        epoch_gp_values = []
        epoch_wasserstein_values = []

        progress_bar = tqdm(dataloader, desc=f"Uncond Epoch {epoch}/{num_epochs}")
        total_batches = len(dataloader)

        for batch_idx, real_images in enumerate(progress_bar):
            real_images = real_images.to(device, non_blocking=True)
            current_batch_size = real_images.size(0)

            critic_optimizer.zero_grad(set_to_none=True)
            noise = sample_noise(current_batch_size, latent_dim, device)

            with torch.no_grad():
                fake_images = generator(noise)

            with autocast(enabled=AMP_ENABLED):
                critic_real = critic(real_images)
                critic_fake = critic(fake_images)
                gp = gradient_penalty(critic, real_images, fake_images, device)
                critic_loss = critic_fake.mean() - critic_real.mean() + lambda_gp * gp
                wasserstein_estimate = critic_real.mean() - critic_fake.mean()

            critic_scaler.scale(critic_loss).backward()
            critic_scaler.step(critic_optimizer)
            critic_scaler.update()

            history["step_critic"].append(float(critic_loss.item()))
            history["step_gp"].append(float(gp.item()))
            history["step_wasserstein"].append(float(wasserstein_estimate.item()))

            epoch_critic_losses.append(float(critic_loss.item()))
            epoch_gp_values.append(float(gp.item()))
            epoch_wasserstein_values.append(float(wasserstein_estimate.item()))

            should_update_generator = ((batch_idx + 1) % n_critic == 0) or ((batch_idx + 1) == total_batches)
            if should_update_generator:
                generator_optimizer.zero_grad(set_to_none=True)
                for parameter in critic.parameters():
                    parameter.requires_grad_(False)

                noise = sample_noise(current_batch_size, latent_dim, device)
                with autocast(enabled=AMP_ENABLED):
                    fake_images = generator(noise)
                    generator_loss = -critic(fake_images).mean()

                generator_scaler.scale(generator_loss).backward()
                generator_scaler.step(generator_optimizer)
                generator_scaler.update()

                for parameter in critic.parameters():
                    parameter.requires_grad_(True)

                history["step_generator"].append(float(generator_loss.item()))
                epoch_generator_losses.append(float(generator_loss.item()))

            last_generator_loss = epoch_generator_losses[-1] if epoch_generator_losses else float("nan")
            progress_bar.set_postfix({
                "critic": f"{epoch_critic_losses[-1]:.3f}",
                "gen": f"{last_generator_loss:.3f}" if epoch_generator_losses else "-",
                "gp": f"{epoch_gp_values[-1]:.3f}",
                "w": f"{epoch_wasserstein_values[-1]:.3f}",
            })

        mean_generator_loss = float(np.mean(epoch_generator_losses)) if epoch_generator_losses else float("nan")
        mean_critic_loss = float(np.mean(epoch_critic_losses))
        mean_gp = float(np.mean(epoch_gp_values))
        mean_wasserstein = float(np.mean(epoch_wasserstein_values))

        history["epoch_generator"].append(mean_generator_loss)
        history["epoch_critic"].append(mean_critic_loss)
        history["epoch_gp"].append(mean_gp)
        history["epoch_wasserstein"].append(mean_wasserstein)

        torch.save(generator.state_dict(), checkpoint_dir / f"{prefix}_generator_last.pt")
        torch.save(critic.state_dict(), checkpoint_dir / f"{prefix}_critic_last.pt")

        if mean_wasserstein > best_wasserstein:
            best_wasserstein = mean_wasserstein
            torch.save(generator.state_dict(), checkpoint_dir / f"{prefix}_generator_best.pt")
            torch.save(critic.state_dict(), checkpoint_dir / f"{prefix}_critic_best.pt")

        print(
            f"Epoch {epoch:02d} | G: {mean_generator_loss:.4f} | "
            f"C: {mean_critic_loss:.4f} | GP: {mean_gp:.4f} | W: {mean_wasserstein:.4f}"
        )

        if epoch % sample_every == 0 or epoch == 1 or epoch == num_epochs:
            preview = generate_samples(generator, latent_dim=latent_dim, device=device, noise=fixed_noise)
            show_tensor_images(preview, title=f"Unconditional samples after epoch {epoch}", nrow=4, figsize=(8, 8))

    return history


generator = Generator(latent_dim=LATENT_DIM).to(DEVICE)
critic = Critic().to(DEVICE)
generator.apply(weights_init)
critic.apply(weights_init)

optimizer_g = optim.Adam(generator.parameters(), lr=LR_G, betas=BETAS)
optimizer_c = optim.Adam(critic.parameters(), lr=LR_C, betas=BETAS)

history_uncond = train_wgan_gp(
    generator=generator,
    critic=critic,
    dataloader=uncond_loader,
    generator_optimizer=optimizer_g,
    critic_optimizer=optimizer_c,
    num_epochs=NUM_EPOCHS,
    latent_dim=LATENT_DIM,
    device=DEVICE,
    n_critic=N_CRITIC,
    lambda_gp=LAMBDA_GP,
    sample_every=SAMPLE_EVERY,
    checkpoint_dir=CHECKPOINT_DIR,
    prefix="unconditional_stage_1",
)
```


```
history_uncond_stage_2 = train_wgan_gp(
    generator=generator,
    critic=critic,
    dataloader=uncond_loader,
    generator_optimizer=optimizer_g,
    critic_optimizer=optimizer_c,
    num_epochs=max(5, NUM_EPOCHS // 2),
    latent_dim=LATENT_DIM,
    device=DEVICE,
    n_critic=N_CRITIC,
    lambda_gp=LAMBDA_GP,
    sample_every=SAMPLE_EVERY,
    checkpoint_dir=CHECKPOINT_DIR,
    prefix="unconditional_stage_2",
)

history_uncond_final = history_uncond_stage_2
```

```
class ConditionalGenerator(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        num_classes: int = 2,
        embedding_dim: int = 32,
        img_channels: int = 3,
        base_channels: int = 64,
    ):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, embedding_dim)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + embedding_dim, base_channels * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels, img_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_features = self.label_embedding(labels)
        x = torch.cat([z, label_features], dim=1).unsqueeze(-1).unsqueeze(-1)
        return self.net(x)


class ConditionalCritic(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        image_size: int = 64,
        img_channels: int = 3,
        base_channels: int = 64,
    ):
        super().__init__()
        self.image_size = image_size
        self.label_embedding = nn.Embedding(num_classes, image_size * image_size)
        self.net = nn.Sequential(
            nn.Conv2d(img_channels + 1, base_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_map = self.label_embedding(labels).view(labels.size(0), 1, self.image_size, self.image_size)
        x = torch.cat([images, label_map], dim=1)
        return self.net(x).view(-1)


def sample_conditional_noise(batch_size: int, latent_dim: int, device: torch.device) -> torch.Tensor:
    return torch.randn(batch_size, latent_dim, device=device)


def conditional_gradient_penalty(
    critic: nn.Module,
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    batch_size = real_images.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = alpha * real_images.float() + (1 - alpha) * fake_images.float().detach()
    interpolated.requires_grad_(True)

    mixed_scores = critic(interpolated, labels)
    gradients = torch.autograd.grad(
        outputs=mixed_scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


@torch.no_grad()
def generate_conditional_samples(
    generator: nn.Module,
    labels: torch.Tensor,
    latent_dim: int = 128,
    device: torch.device = DEVICE,
    noise: torch.Tensor = None,
) -> torch.Tensor:
    was_training = generator.training
    generator.eval()
    labels = labels.to(device)
    if noise is None:
        noise = sample_conditional_noise(len(labels), latent_dim, device)
    fake_images = generator(noise, labels)
    generator.train(was_training)
    return fake_images
```

```
def load_matching_weights(target_model: nn.Module, source_state_dict: dict, verbose: bool = True):
    target_state = target_model.state_dict()

    copied = []
    skipped = []

    for name, source_weight in source_state_dict.items():
        if name in target_state and target_state[name].shape == source_weight.shape:
            target_state[name] = source_weight
            copied.append(name)
        else:
            skipped.append(name)

    target_model.load_state_dict(target_state)

    if verbose:
        print(f"Copied: {len(copied)}")
        print(f"Skipped: {len(skipped)}")

        print("\nCopied layers:")
        for name in copied:
            print("  +", name)

        print("\nSkipped layers:")
        for name in skipped:
            print("  -", name)

    return copied, skipped
```

```
def train_conditional_wgan_gp(
    generator: nn.Module,
    critic: nn.Module,
    dataloader: DataLoader,
    generator_optimizer: optim.Optimizer,
    critic_optimizer: optim.Optimizer,
    num_epochs: int,
    latent_dim: int,
    device: torch.device,
    n_critic: int = 5,
    lambda_gp: float = 10.0,
    sample_every: int = 2,
    checkpoint_dir: Path = CHECKPOINT_DIR,
    prefix: str = "conditional",
):
    history = {
        "step_generator": [],
        "step_critic": [],
        "step_gp": [],
        "step_wasserstein": [],
        "epoch_generator": [],
        "epoch_critic": [],
        "epoch_gp": [],
        "epoch_wasserstein": [],
    }

    best_wasserstein = float("-inf")
    generator_scaler = GradScaler(enabled=AMP_ENABLED)
    critic_scaler = GradScaler(enabled=AMP_ENABLED)
    fixed_noise = sample_conditional_noise(8, latent_dim, device)
    female_labels = torch.zeros(8, dtype=torch.long, device=device)
    male_labels = torch.ones(8, dtype=torch.long, device=device)

    for epoch in range(1, num_epochs + 1):
        generator.train()
        critic.train()

        epoch_generator_losses = []
        epoch_critic_losses = []
        epoch_gp_values = []
        epoch_wasserstein_values = []

        progress_bar = tqdm(dataloader, desc=f"Cond Epoch {epoch}/{num_epochs}")
        total_batches = len(dataloader)

        for batch_idx, (real_images, labels) in enumerate(progress_bar):
            real_images = real_images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            current_batch_size = real_images.size(0)

            critic_optimizer.zero_grad(set_to_none=True)
            noise = sample_conditional_noise(current_batch_size, latent_dim, device)

            with torch.no_grad():
                fake_images = generator(noise, labels)

            with autocast(enabled=AMP_ENABLED):
                critic_real = critic(real_images, labels)
                critic_fake = critic(fake_images, labels)
                gp = conditional_gradient_penalty(critic, real_images, fake_images, labels, device)
                critic_loss = critic_fake.mean() - critic_real.mean() + lambda_gp * gp
                wasserstein_estimate = critic_real.mean() - critic_fake.mean()

            critic_scaler.scale(critic_loss).backward()
            critic_scaler.step(critic_optimizer)
            critic_scaler.update()

            history["step_critic"].append(float(critic_loss.item()))
            history["step_gp"].append(float(gp.item()))
            history["step_wasserstein"].append(float(wasserstein_estimate.item()))

            epoch_critic_losses.append(float(critic_loss.item()))
            epoch_gp_values.append(float(gp.item()))
            epoch_wasserstein_values.append(float(wasserstein_estimate.item()))

            should_update_generator = ((batch_idx + 1) % n_critic == 0) or ((batch_idx + 1) == total_batches)
            if should_update_generator:
                generator_optimizer.zero_grad(set_to_none=True)
                for parameter in critic.parameters():
                    parameter.requires_grad_(False)

                noise = sample_conditional_noise(current_batch_size, latent_dim, device)
                with autocast(enabled=AMP_ENABLED):
                    fake_images = generator(noise, labels)
                    generator_loss = -critic(fake_images, labels).mean()

                generator_scaler.scale(generator_loss).backward()
                generator_scaler.step(generator_optimizer)
                generator_scaler.update()

                for parameter in critic.parameters():
                    parameter.requires_grad_(True)

                history["step_generator"].append(float(generator_loss.item()))
                epoch_generator_losses.append(float(generator_loss.item()))

            last_generator_loss = epoch_generator_losses[-1] if epoch_generator_losses else float("nan")
            progress_bar.set_postfix({
                "critic": f"{epoch_critic_losses[-1]:.3f}",
                "gen": f"{last_generator_loss:.3f}" if epoch_generator_losses else "-",
                "gp": f"{epoch_gp_values[-1]:.3f}",
                "w": f"{epoch_wasserstein_values[-1]:.3f}",
            })

        mean_generator_loss = float(np.mean(epoch_generator_losses)) if epoch_generator_losses else float("nan")
        mean_critic_loss = float(np.mean(epoch_critic_losses))
        mean_gp = float(np.mean(epoch_gp_values))
        mean_wasserstein = float(np.mean(epoch_wasserstein_values))

        history["epoch_generator"].append(mean_generator_loss)
        history["epoch_critic"].append(mean_critic_loss)
        history["epoch_gp"].append(mean_gp)
        history["epoch_wasserstein"].append(mean_wasserstein)

        torch.save(generator.state_dict(), checkpoint_dir / f"{prefix}_generator_last.pt")
        torch.save(critic.state_dict(), checkpoint_dir / f"{prefix}_critic_last.pt")

        if mean_wasserstein > best_wasserstein:
            best_wasserstein = mean_wasserstein
            torch.save(generator.state_dict(), checkpoint_dir / f"{prefix}_generator_best.pt")
            torch.save(critic.state_dict(), checkpoint_dir / f"{prefix}_critic_best.pt")

        print(
            f"Epoch {epoch:02d} | G: {mean_generator_loss:.4f} | "
            f"C: {mean_critic_loss:.4f} | GP: {mean_gp:.4f} | W: {mean_wasserstein:.4f}"
        )

        if epoch % sample_every == 0 or epoch == 1 or epoch == num_epochs:
            female_preview = generate_conditional_samples(
                generator,
                labels=female_labels,
                latent_dim=latent_dim,
                device=device,
                noise=fixed_noise,
            )
            male_preview = generate_conditional_samples(
                generator,
                labels=male_labels,
                latent_dim=latent_dim,
                device=device,
                noise=fixed_noise,
            )
            show_tensor_images(female_preview, title=f"Conditional female samples after epoch {epoch}", nrow=8, figsize=(16, 3))
            show_tensor_images(male_preview, title=f"Conditional male samples after epoch {epoch}", nrow=8, figsize=(16, 3))

    return history


cond_generator = ConditionalGenerator(latent_dim=LATENT_DIM).to(DEVICE)
cond_critic = ConditionalCritic(image_size=IMAGE_SIZE).to(DEVICE)

cond_generator.apply(weights_init)
cond_critic.apply(weights_init)

UNCOND_GENERATOR_CKPT = CHECKPOINT_DIR / "unconditional_stage_2_generator_last.pt"
UNCOND_CRITIC_CKPT = CHECKPOINT_DIR / "unconditional_stage_2_critic_last.pt"

uncond_generator_state = torch.load(UNCOND_GENERATOR_CKPT, map_location=DEVICE)
uncond_critic_state = torch.load(UNCOND_CRITIC_CKPT, map_location=DEVICE)

copied_g, skipped_g = load_matching_weights(
    cond_generator,
    uncond_generator_state,
    verbose=True,
)

copied_c, skipped_c = load_matching_weights(
    cond_critic,
    uncond_critic_state,
    verbose=True,
)

optimizer_g_cond = optim.Adam(cond_generator.parameters(), lr=LR_G, betas=BETAS)
optimizer_c_cond = optim.Adam(cond_critic.parameters(), lr=LR_C, betas=BETAS)

history_cond = train_conditional_wgan_gp(
    generator=cond_generator,
    critic=cond_critic,
    dataloader=cond_loader,
    generator_optimizer=optimizer_g_cond,
    critic_optimizer=optimizer_c_cond,
    num_epochs=NUM_EPOCHS_COND,
    latent_dim=LATENT_DIM,
    device=DEVICE,
    n_critic=N_CRITIC,
    lambda_gp=LAMBDA_GP,
    sample_every=SAMPLE_EVERY,
    checkpoint_dir=CHECKPOINT_DIR,
    prefix="conditional_stage_1",
)
```

```
history_cond_stage_2 = train_conditional_wgan_gp(
    generator=cond_generator,
    critic=cond_critic,
    dataloader=cond_loader,
    generator_optimizer=optimizer_g_cond,
    critic_optimizer=optimizer_c_cond,
    num_epochs=max(5, NUM_EPOCHS_COND // 2),
    latent_dim=LATENT_DIM,
    device=DEVICE,
    n_critic=N_CRITIC,
    lambda_gp=LAMBDA_GP,
    sample_every=SAMPLE_EVERY,
    checkpoint_dir=CHECKPOINT_DIR,
    prefix="conditional_stage_2",
)

history_cond_final = history_cond_stage_2
```

```
def images_to_uint8(images: torch.Tensor) -> torch.Tensor:
    images = denormalize(images).clamp(0, 1)
    return (images * 255).round().to(torch.uint8)


@torch.no_grad()
def compute_fid_and_is(
    generator: nn.Module,
    real_loader: DataLoader,
    num_images: int,
    latent_dim: int,
    device: torch.device,
    conditional: bool = False,
    compute_fid: bool = True,
    compute_is: bool = True,
    fake_batch_size: int | None = None,
    fid_feature: int = 2048,
    desc: str = "Metrics",
    label_sampler=None,
) -> dict:
    num_images = min(num_images, len(real_loader.dataset))
    if num_images <= 0:
        raise ValueError("Need at least one image to compute metrics.")

    if not compute_fid and not compute_is:
        raise ValueError("At least one of `compute_fid` or `compute_is` must be True.")

    fake_batch_size = fake_batch_size or BATCH_SIZE

    fid_metric = None
    if compute_fid:
        fid_metric = FrechetInceptionDistance(feature=fid_feature).to(device)
        fid_metric.set_dtype(torch.float64)

    inception_metric = InceptionScore().to(device) if compute_is else None

    was_training = generator.training
    generator.eval()

    try:
        if compute_fid:
            real_seen = 0
            with tqdm(total=num_images, desc=f"{desc}: real", unit="img") as pbar:
                for batch in real_loader:
                    if real_seen >= num_images:
                        break

                    real_images = batch[0] if isinstance(batch, (list, tuple)) else batch
                    real_images = real_images.to(device, non_blocking=(device.type == "cuda"))

                    remaining = num_images - real_seen
                    real_batch = images_to_uint8(real_images[:remaining])

                    fid_metric.update(real_batch, real=True)

                    used = real_batch.size(0)
                    real_seen += used
                    pbar.update(used)

        fake_seen = 0
        with tqdm(total=num_images, desc=f"{desc}: fake", unit="img") as pbar:
            while fake_seen < num_images:
                current_batch_size = min(fake_batch_size, num_images - fake_seen)

                if conditional:
                    if label_sampler is None:
                        labels = torch.randint(0, 2, (current_batch_size,), device=device)
                    else:
                        labels = label_sampler(current_batch_size, device)

                    noise = sample_conditional_noise(current_batch_size, latent_dim, device)
                    fake_images = generator(noise, labels)
                else:
                    noise = sample_noise(current_batch_size, latent_dim, device)
                    fake_images = generator(noise)

                fake_batch = images_to_uint8(fake_images)

                if compute_fid:
                    fid_metric.update(fake_batch, real=False)
                if compute_is:
                    inception_metric.update(fake_batch)

                fake_seen += current_batch_size
                pbar.update(current_batch_size)

        results = {"Images Used": int(num_images)}

        if compute_fid:
            results["FID"] = float(fid_metric.compute().item())

        if compute_is:
            inception_mean, inception_std = inception_metric.compute()
            results["Inception Score"] = float(inception_mean.item())
            results["IS Std"] = float(inception_std.item())

        return results

    finally:
        generator.train(was_training)
        if device.type == "cuda":
            torch.cuda.empty_cache()
```

```
metric_results = {}
metric_image_count = NUM_METRIC_IMAGES

metric_results["Unconditional WGAN-GP"] = compute_fid_and_is(
        generator=generator,
        real_loader=uncond_eval_loader,
        num_images=metric_image_count,
        latent_dim=LATENT_DIM,
        device=DEVICE,
        conditional=False,
    )
```

```
metric_results["Conditional WGAN-GP"] = compute_fid_and_is(
        generator=cond_generator,
        real_loader=cond_eval_loader,
        num_images=metric_image_count,
        latent_dim=LATENT_DIM,
        device=DEVICE,
        conditional=True,
    )
```

```
metric_table = pd.DataFrame(metric_results).T.round(4)
metric_table
```

```
def plot_training_curves(history_uncond: dict, history_cond: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    metrics_to_plot = [
        ("epoch_generator", "Generator loss"),
        ("epoch_critic", "Critic loss"),
        ("epoch_gp", "Gradient penalty"),
    ]

    for column_idx, (key, title) in enumerate(metrics_to_plot):
        axes[0, column_idx].plot(history_uncond[key], marker="o")
        axes[0, column_idx].set_title(f"Unconditional {title}")
        axes[0, column_idx].set_xlabel("Epoch")
        axes[0, column_idx].grid(alpha=0.3)

        axes[1, column_idx].plot(history_cond[key], marker="o", color="tab:orange")
        axes[1, column_idx].set_title(f"Conditional {title}")
        axes[1, column_idx].set_xlabel("Epoch")
        axes[1, column_idx].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(history_uncond["epoch_wasserstein"], marker="o", label="Unconditional")
    plt.plot(history_cond["epoch_wasserstein"], marker="o", label="Conditional")
    plt.title("Wasserstein estimate by epoch")
    plt.xlabel("Epoch")
    plt.ylabel("E[D(real)] - E[D(fake)]")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()


history_uncond_for_plot = globals().get("history_uncond_final", globals().get("history_uncond_stage_2", history_uncond))
history_cond_for_plot = globals().get("history_cond_final", globals().get("history_cond_stage_2", history_cond))
plot_training_curves(history_uncond_for_plot, history_cond_for_plot)
```

```
final_real_batch = next(iter(uncond_eval_loader))
show_tensor_images(final_real_batch[:16], title="Real faces", nrow=4, figsize=(8, 8))

uncond_samples = generate_samples(generator, num_samples=16, latent_dim=LATENT_DIM, device=DEVICE)
show_tensor_images(uncond_samples, title="Unconditional WGAN-GP samples", nrow=4, figsize=(8, 8))

shared_noise = sample_conditional_noise(8, LATENT_DIM, DEVICE)
female_labels = torch.zeros(8, dtype=torch.long, device=DEVICE)
male_labels = torch.ones(8, dtype=torch.long, device=DEVICE)

female_samples = generate_conditional_samples(
    cond_generator,
    labels=female_labels,
    latent_dim=LATENT_DIM,
    device=DEVICE,
    noise=shared_noise,
)
male_samples = generate_conditional_samples(
    cond_generator,
    labels=male_labels,
    latent_dim=LATENT_DIM,
    device=DEVICE,
    noise=shared_noise,
)

show_tensor_images(female_samples, title="Conditional female samples", nrow=8, figsize=(16, 3))
show_tensor_images(male_samples, title="Conditional male samples", nrow=8, figsize=(16, 3))
```

```

```