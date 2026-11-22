"""Image preprocessing and pairing with precomputed enhancement outputs."""
from pathlib import Path
from PIL import Image
from torchvision import transforms

IMG_EXTENSIONS={".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGE_SIZE = (224, 224)
IMAGENET_MEAN=(0.485, 0.456, 0.406)
IMAGENET_STD=(0.229, 0.224, 0.225)

TRANSFORM = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def list_original_images(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(folder)
    images = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS)
    if not images:
        raise FileNotFoundError(f"No fundus images in {folder}")
    return images


def paired_enhancement(original: Path, enhanced_dir: Path) -> Path:
    """By convention, pyramid images have {original_stem}_enhanced.png."""
    folder=Path(enhanced_dir)
    candidates=[folder/f"{original.stem}_enhanced.png", folder/f"{original.stem}.png",
                folder/original.name]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing precomputed enhancement for {original.name}; expected {candidates[0]}")


def load_tensor(path: Path):
    with Image.open(path) as image:
        return TRANSFORM(image.convert("RGB")).unsqueeze(0)
