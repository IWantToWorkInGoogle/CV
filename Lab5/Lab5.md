```
import gc
import json
import math
import os
import platform
import random
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import Markdown, display
from PIL import Image, ImageOps, UnidentifiedImageError
# from skimage import filters

def resolve_project_root() -> Path:
    cwd = Path.cwd().resolve()
    candidates = [
        cwd,
        cwd / "Lab5",
        cwd.parent / "Lab5",
    ]
    for candidate in candidates:
        if (candidate / "text2img_personalization_lab.ipynb").exists():
            return candidate.resolve()
    return cwd

PROJECT_ROOT = resolve_project_root()
DATA_DIR = PROJECT_ROOT / "data"
USER_PHOTOS_DIR = DATA_DIR / "user_photos"
REGULARIZATION_DIR = DATA_DIR / "regularization_people"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PREPARED_USER_DIR = OUTPUTS_DIR / "prepared_user_photos"
LORA_DIR = OUTPUTS_DIR / "lora"
GENERATED_DIR = OUTPUTS_DIR / "generated"
PERSONAL_DIR = GENERATED_DIR / "personal_token"
GENDER_BASELINE_DIR = GENERATED_DIR / "gender_baseline"
OTHER_PEOPLE_DIR = GENERATED_DIR / "other_people_control"
METRICS_DIR = OUTPUTS_DIR / "metrics"
REPORTS_DIR = OUTPUTS_DIR / "reports"

for directory in [
    DATA_DIR,
    USER_PHOTOS_DIR,
    REGULARIZATION_DIR,
    OUTPUTS_DIR,
    PREPARED_USER_DIR,
    LORA_DIR,
    GENERATED_DIR,
    PERSONAL_DIR,
    GENDER_BASELINE_DIR,
    OTHER_PEOPLE_DIR,
    METRICS_DIR,
    REPORTS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

BASE_MODEL = os.environ.get("BASE_MODEL", "runwayml/stable-diffusion-v1-5")
PERSON_TOKEN = os.environ.get("PERSON_TOKEN", "skslabperson")
PERSON_CLASS = os.environ.get("PERSON_CLASS", "man") 
INSTANCE_PROMPT = f"a photo of {PERSON_TOKEN} {PERSON_CLASS}"
CLASS_PROMPT = f"a photo of a {PERSON_CLASS}"

RESOLUTION = int(os.environ.get("RESOLUTION", 512))
TRAIN_STEPS = int(os.environ.get("TRAIN_STEPS", 1000))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 1e-4))
TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", 1))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", 4))
SEED = int(os.environ.get("SEED", 42))
MIXED_PRECISION = os.environ.get("MIXED_PRECISION", "fp16")
NUM_INFERENCE_STEPS = int(os.environ.get("NUM_INFERENCE_STEPS", 30))
GUIDANCE_SCALE = float(os.environ.get("GUIDANCE_SCALE", 7.5))
LORA_R = int(os.environ.get("LORA_R", 16))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", 16))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", 0.0))
CHECKPOINTING_STEPS = int(os.environ.get("CHECKPOINTING_STEPS", 250))
CENTER_CROP = True
ENABLE_XFORMERS = True
ENABLE_GRADIENT_CHECKPOINTING = True
FORCE_RETRAIN = False

MANDATORY_PERSONAL_PROMPTS = [
    f"{PERSON_TOKEN} in a forest, high quality, realism",
    f"{PERSON_TOKEN} in a city, high quality, realism",
    f"{PERSON_TOKEN} in a beach, high quality, realism",
]

EXTRA_PERSONAL_PROMPTS = [
    f"a realistic high quality photo of {PERSON_TOKEN} {PERSON_CLASS} in a cyberpunk city, neon lights, cinematic portrait",
    f"a realistic high quality photo of {PERSON_TOKEN} {PERSON_CLASS} made of polished metal, studio portrait",
    f"a realistic high quality photo of {PERSON_TOKEN} {PERSON_CLASS} in an elven city, fantasy architecture, natural light",
    f"a realistic high quality photo of {PERSON_TOKEN} {PERSON_CLASS} as an astronaut on Mars, cinematic lighting",
    f"a realistic high quality photo of {PERSON_TOKEN} {PERSON_CLASS} in a medieval castle, dramatic light",
]

GENDER_BASELINE_PROMPTS = [
    f"{PERSON_CLASS} in a forest, high quality, realism",
    f"{PERSON_CLASS} in a city, high quality, realism",
    f"{PERSON_CLASS} in a beach, high quality, realism",
]

OTHER_PEOPLE_CONTROL_PROMPTS = [
    "a realistic high quality photo of a young man in a city",
    "a realistic high quality photo of an old man in a forest",
    "a realistic high quality photo of a woman on a beach",
    "a realistic high quality photo of a group of people in a park",
]

def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(SEED)

def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "image"

def list_image_files(directory: Path):
    valid_suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sorted([path for path in directory.iterdir() if path.suffix.lower() in valid_suffixes])

def show_images_grid(image_paths, titles=None, cols=3, figsize=(14, 10)):
    image_paths = [Path(p) for p in image_paths]
    if not image_paths:
        print("No images to display.")
        return
    cols = min(cols, len(image_paths))
    rows = math.ceil(len(image_paths) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()
    for ax in axes:
        ax.axis("off")
    for idx, image_path in enumerate(image_paths):
        with Image.open(image_path).convert("RGB") as image:
            axes[idx].imshow(image)
        axes[idx].axis("off")
        if titles is not None and idx < len(titles):
            axes[idx].set_title(titles[idx], fontsize=10)
    plt.tight_layout()
    plt.show()

def save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

print(f"PROJECT_ROOT = {PROJECT_ROOT}")
print(f"USER_PHOTOS_DIR = {USER_PHOTOS_DIR}")
print(f"Outputs will be written to: {OUTPUTS_DIR}")
```

```
print(f"Python: {platform.python_version()}")
print(f"Platform: {platform.platform()}")
print(f"Torch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device_index = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device_index)
    total_memory_gb = torch.cuda.get_device_properties(device_index).total_memory / 1024**3
    print(f"GPU: {gpu_name}")
    print(f"GPU memory: {total_memory_gb:.2f} GB")
else:
    print("WARNING: CUDA is not available. LoRA training on CPU may take a very long time.")
```

```
VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def prepare_image(path: Path, size: int = RESOLUTION) -> Image.Image:
    with Image.open(path) as raw_image:
        image = ImageOps.exif_transpose(raw_image).convert("RGB")

    min_side = min(image.size)
    left = (image.width - min_side) // 2
    top = (image.height - min_side) // 2
    image = image.crop((left, top, left + min_side, top + min_side))
    image = image.resize((size, size), Image.Resampling.LANCZOS)
    return image

user_image_paths = list_image_files(USER_PHOTOS_DIR) if USER_PHOTOS_DIR.exists() else []

print(f"Found {len(user_image_paths)} user images in {USER_PHOTOS_DIR}")
if len(user_image_paths) < 5:
    raise FileNotFoundError(
        "Place at least 5 user photos into data/user_photos/ before training. "
        "Recommended amount: 10-20 photos."
    )

valid_prepared_paths = []
broken_files = []
original_sizes = []

for path in user_image_paths:
    try:
        with Image.open(path) as raw_image:
            original_sizes.append(raw_image.size)
        prepared = prepare_image(path, size=RESOLUTION)
        output_path = PREPARED_USER_DIR / f"{path.stem}.png"
        prepared.save(output_path)
        valid_prepared_paths.append(output_path)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        broken_files.append((str(path), str(exc)))

print(f"Prepared images: {len(valid_prepared_paths)}")
if broken_files:
    print("Broken files:")
    for item in broken_files:
        print(" -", item[0], "|", item[1])

sizes_df = pd.DataFrame(original_sizes, columns=["width", "height"])
display(sizes_df.describe().round(2))
show_images_grid(valid_prepared_paths[:6], cols=3, figsize=(12, 8))
```

```
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

try:
    from accelerate import Accelerator
except ImportError as exc:
    raise ImportError(
        "Failed to import accelerate. Run the dependency installation cell, restart the kernel, and rerun the notebook from the top."
    ) from exc
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
try:
    from diffusers.utils.import_utils import is_xformers_available
except ImportError:
    from diffusers.utils import is_xformers_available
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from transformers import AutoTokenizer, CLIPTextModel

try:
    from diffusers.utils import convert_state_dict_to_diffusers
except ImportError:
    convert_state_dict_to_diffusers = None

class DreamBoothDataset(Dataset):
    def __init__(self, image_dir: Path, tokenizer, prompt: str, size: int = 512, center_crop: bool = True):
        self.image_paths = sorted(image_dir.glob("*.png"))
        self.tokenizer = tokenizer
        self.prompt = prompt
        crop = transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size)
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                crop,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        with Image.open(path).convert("RGB") as image:
            pixel_values = self.image_transform(image)
        tokenized = self.tokenizer(
            self.prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": tokenized.input_ids[0],
            "path": str(path),
        }

def collate_fn(examples):
    return {
        "pixel_values": torch.stack([example["pixel_values"] for example in examples]),
        "input_ids": torch.stack([example["input_ids"] for example in examples]),
        "paths": [example["path"] for example in examples],
    }

def choose_weight_dtype(mixed_precision: str):
    if mixed_precision == "fp16" and torch.cuda.is_available():
        return torch.float16
    if mixed_precision == "bf16" and torch.cuda.is_available():
        return torch.bfloat16
    return torch.float32

def save_unet_lora_weights(unwrapped_unet, output_dir: Path):
    lora_state_dict = get_peft_model_state_dict(unwrapped_unet)
    if convert_state_dict_to_diffusers is not None:
        lora_state_dict = convert_state_dict_to_diffusers(lora_state_dict)
    StableDiffusionPipeline.save_lora_weights(
        save_directory=output_dir,
        unet_lora_layers=lora_state_dict,
        safe_serialization=True,
    )

def train_dreambooth_lora():
    seed_everything(SEED)
    accelerator = Accelerator(
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        mixed_precision=MIXED_PRECISION if torch.cuda.is_available() else "no",
    )

    weight_dtype = choose_weight_dtype(MIXED_PRECISION)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, subfolder="tokenizer", use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(BASE_MODEL, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(BASE_MODEL, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(BASE_MODEL, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler")

    if not hasattr(unet, "add_adapter"):
        raise AttributeError(
            "The installed diffusers/peft versions do not expose unet.add_adapter(). "
            "Run the dependency installation cell and restart the kernel."
        )

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    vae.eval()
    text_encoder.eval()

    if ENABLE_GRADIENT_CHECKPOINTING and hasattr(unet, "enable_gradient_checkpointing"):
        unet.enable_gradient_checkpointing()

    if ENABLE_XFORMERS and is_xformers_available():
        try:
            unet.enable_xformers_memory_efficient_attention()
            accelerator.print("xFormers attention enabled.")
        except Exception as exc:
            accelerator.print(f"Could not enable xFormers: {exc}")

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_config)

    train_dataset = DreamBoothDataset(
        image_dir=PREPARED_USER_DIR,
        tokenizer=tokenizer,
        prompt=INSTANCE_PROMPT,
        size=RESOLUTION,
        center_crop=CENTER_CROP,
    )
    if len(train_dataset) == 0:
        raise RuntimeError("Prepared dataset is empty. Run the dataset preparation cell first.")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    trainable_params = [parameter for parameter in unet.parameters() if parameter.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable LoRA parameters were created for U-Net.")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )
    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=TRAIN_STEPS,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    device = accelerator.device
    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    unet.train()

    train_config = {
        "base_model": BASE_MODEL,
        "person_token": PERSON_TOKEN,
        "person_class": PERSON_CLASS,
        "instance_prompt": INSTANCE_PROMPT,
        "class_prompt": CLASS_PROMPT,
        "resolution": RESOLUTION,
        "train_steps": TRAIN_STEPS,
        "learning_rate": LEARNING_RATE,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "seed": SEED,
        "mixed_precision": MIXED_PRECISION if torch.cuda.is_available() else "no",
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "center_crop": CENTER_CROP,
        "uses_prior_preservation": False,
    }
    save_json(train_config, LORA_DIR / "training_config.json")

    global_step = 0
    progress_bar = tqdm(range(TRAIN_STEPS), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Training LoRA")
    losses = []

    while global_step < TRAIN_STEPS:
        for batch in train_dataloader:
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(device=device, dtype=weight_dtype)
                input_ids = batch["input_ids"].to(device)

                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device,
                ).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(input_ids)[0]
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unsupported prediction type: {noise_scheduler.config.prediction_type}"
                    )

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                losses.append(float(loss.detach().cpu()))
                progress_bar.update(1)
                progress_bar.set_postfix(loss=f"{losses[-1]:.4f}")

                if global_step % CHECKPOINTING_STEPS == 0 and accelerator.is_main_process:
                    checkpoint_payload = {
                        "global_step": global_step,
                        "loss": losses[-1],
                    }
                    save_json(checkpoint_payload, LORA_DIR / f"checkpoint_step_{global_step}.json")

                if global_step >= TRAIN_STEPS:
                    break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_unet = accelerator.unwrap_model(unet)
        save_unet_lora_weights(unwrapped_unet, LORA_DIR)
        history_df = pd.DataFrame({"step": list(range(1, len(losses) + 1)), "loss": losses})
        history_df.to_csv(LORA_DIR / "loss_history.csv", index=False)
    else:
        history_df = pd.DataFrame()

    del unet, vae, text_encoder, optimizer, train_dataloader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    accelerator.wait_for_everyone()
    return history_df
```

```
def build_generation_pipeline():
    seed_everything(SEED)
    weight_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=weight_dtype,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    lora_files = sorted(list(LORA_DIR.glob("*.safetensors")) + list(LORA_DIR.glob("*.bin")))
    if not lora_files:
        raise FileNotFoundError(
            f"No LoRA weights found in {LORA_DIR}. Make sure the training cell completed successfully."
        )

    weight_name = lora_files[0].name
    pipe.load_lora_weights(str(LORA_DIR), weight_name=weight_name)
    pipe.enable_attention_slicing()

    if ENABLE_XFORMERS and hasattr(pipe, "enable_xformers_memory_efficient_attention") and is_xformers_available():
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as exc:
            print(f"Could not enable xFormers for pipeline: {exc}")

    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    else:
        pipe = pipe.to("cpu")

    return pipe

pipe = build_generation_pipeline()
print("Generation pipeline is ready.")
```

```
def generate_one_image(pipe, prompt: str, seed: int, output_path: Path):
    generator_device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    image = pipe(
        prompt=prompt,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
    ).images[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path

def batch_generate(pipe, prompts, output_dir: Path, group_name: str, seeds_per_prompt: int = 1, base_seed: int = SEED):
    rows = []
    for prompt_index, prompt in enumerate(prompts):
        for variant_idx in range(seeds_per_prompt):
            seed = base_seed + prompt_index * 100 + variant_idx
            filename = f"{prompt_index:02d}_{slugify(prompt)}_seed{seed}.png"
            output_path = output_dir / filename
            generate_one_image(pipe, prompt=prompt, seed=seed, output_path=output_path)
            rows.append(
                {
                    "image_path": str(output_path),
                    "group": group_name,
                    "prompt": prompt,
                    "seed": seed,
                }
            )
    return rows

generated_rows = []
generated_rows.extend(
    batch_generate(
        pipe,
        prompts=MANDATORY_PERSONAL_PROMPTS,
        output_dir=PERSONAL_DIR,
        group_name="personal",
        seeds_per_prompt=2,
        base_seed=SEED,
    )
)
generated_rows.extend(
    batch_generate(
        pipe,
        prompts=EXTRA_PERSONAL_PROMPTS,
        output_dir=PERSONAL_DIR,
        group_name="personal",
        seeds_per_prompt=1,
        base_seed=SEED + 1000,
    )
)
generated_rows.extend(
    batch_generate(
        pipe,
        prompts=GENDER_BASELINE_PROMPTS,
        output_dir=GENDER_BASELINE_DIR,
        group_name="gender_baseline",
        seeds_per_prompt=2,
        base_seed=SEED + 2000,
    )
)
generated_rows.extend(
    batch_generate(
        pipe,
        prompts=OTHER_PEOPLE_CONTROL_PROMPTS,
        output_dir=OTHER_PEOPLE_DIR,
        group_name="other_people_control",
        seeds_per_prompt=1,
        base_seed=SEED + 3000,
    )
)

generated_df = pd.DataFrame(generated_rows)
generated_df = generated_df.sort_values(["group", "prompt", "seed"]).reset_index(drop=True)
generated_df.to_csv(GENERATED_DIR / "metadata.csv", index=False)
save_json(generated_df.to_dict(orient="records"), GENERATED_DIR / "metadata.json")
display(generated_df.head(10))
```

```
personal_preview = list(PERSONAL_DIR.glob("*.png"))[:9]
baseline_preview = list(GENDER_BASELINE_DIR.glob("*.png"))[:6]
control_preview = list(OTHER_PEOPLE_DIR.glob("*.png"))[:6]

print("Personalized generations:")
show_images_grid(personal_preview, cols=3, figsize=(14, 12))

print("Gender baseline generations:")
show_images_grid(baseline_preview, cols=3, figsize=(12, 8))

print("Other people control generations:")
show_images_grid(control_preview, cols=3, figsize=(12, 8))
```

```
from facenet_pytorch import InceptionResnetV1, MTCNN
from transformers import CLIPModel, CLIPProcessor

metadata_json_path = GENERATED_DIR / "metadata.json"
if "generated_rows" not in globals():
    if not metadata_json_path.exists():
        raise FileNotFoundError(
            "Generated metadata not found. Run the image generation cell before calculating metrics."
        )
    generated_rows = json.loads(metadata_json_path.read_text(encoding="utf-8"))

metrics_device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(metrics_device).eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, device=metrics_device)
face_model = InceptionResnetV1(pretrained="vggface2").eval().to(metrics_device)

def compute_clip_score(image_path: Path, prompt: str) -> float:
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(text=[prompt], images=image, return_tensors="pt", padding=True).to(metrics_device)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
        score = (image_embeds * text_embeds).sum(dim=-1).item()
    return float(score)

def compute_face_embedding(image_path: Path):
    image = Image.open(image_path).convert("RGB")
    face_tensor = mtcnn(image)
    if face_tensor is None:
        return None
    if face_tensor.ndim == 3:
        face_tensor = face_tensor.unsqueeze(0)
    face_tensor = face_tensor.to(metrics_device)
    with torch.no_grad():
        embedding = face_model(face_tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.mean(dim=0, keepdim=True)

def compute_user_reference_embedding(image_paths):
    embeddings = []
    for path in image_paths:
        embedding = compute_face_embedding(Path(path))
        if embedding is not None:
            embeddings.append(embedding)
    if not embeddings:
        raise RuntimeError(
            "Face embeddings could not be extracted from user photos. "
            "Use clearer portrait photos where the face is visible."
        )
    reference = torch.cat(embeddings, dim=0).mean(dim=0, keepdim=True)
    reference = reference / reference.norm(dim=-1, keepdim=True)
    return reference

def compute_face_similarity(image_path: Path, user_reference_embedding: torch.Tensor):
    embedding = compute_face_embedding(image_path)
    if embedding is None:
        return np.nan
    score = torch.nn.functional.cosine_similarity(embedding, user_reference_embedding).item()
    return float(score)

def compute_sharpness_score(image_path: Path) -> float:
    gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0
    padded = np.pad(gray, 1, mode="edge")
    laplace_map = (
        padded[1:-1, :-2]
        + padded[1:-1, 2:]
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        - 4.0 * padded[1:-1, 1:-1]
    )
    return float(np.var(laplace_map))

user_reference_embedding = compute_user_reference_embedding(valid_prepared_paths)

metric_rows = []
for row in generated_rows:
    image_path = Path(row["image_path"])
    metric_rows.append(
        {
            **row,
            "clip_score": compute_clip_score(image_path, row["prompt"]),
            "face_similarity": compute_face_similarity(image_path, user_reference_embedding),
            "sharpness_score": compute_sharpness_score(image_path),
        }
    )

results_df = pd.DataFrame(metric_rows)
results_df = results_df.sort_values(["group", "prompt", "seed"]).reset_index(drop=True)
results_df.to_csv(METRICS_DIR / "results.csv", index=False)
save_json(results_df.to_dict(orient="records"), METRICS_DIR / "results.json")
display(results_df.head(10))
```

```
results_csv_path = METRICS_DIR / "results.csv"
if "results_df" not in globals():
    if not results_csv_path.exists():
        raise FileNotFoundError("Metrics CSV not found. Run the metrics calculation cell first.")
    results_df = pd.read_csv(results_csv_path)

group_means = (
    results_df.groupby("group")[["clip_score", "face_similarity", "sharpness_score"]]
    .mean(numeric_only=True)
    .sort_index()
)
display(group_means)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
group_means["clip_score"].plot(kind="bar", ax=axes[0], title="Mean CLIPScore by group")
axes[0].set_ylabel("score")
group_means["face_similarity"].plot(kind="bar", ax=axes[1], title="Mean Face Similarity by group")
axes[1].set_ylabel("score")
group_means["sharpness_score"].plot(kind="bar", ax=axes[2], title="Mean Sharpness by group")
axes[2].set_ylabel("score")
plt.tight_layout()
plt.show()

top_preview_df = (
    results_df.sort_values(
        by=["group", "face_similarity", "clip_score"], ascending=[True, False, False]
    )
    .groupby("group")
    .head(4)
)

preview_titles = [
    f"{row.group}\nCLIP={row.clip_score:.3f}, Face={row.face_similarity:.3f}, Sharp={row.sharpness_score:.4f}"
    for row in top_preview_df.itertuples()
]
show_images_grid(top_preview_df["image_path"].tolist(), titles=preview_titles, cols=4, figsize=(18, 10))
```