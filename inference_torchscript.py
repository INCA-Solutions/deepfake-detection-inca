import os
import sys
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import CLIPProcessor
from glob import glob

# Auto-select device and dtype (CUDA > MPS > CPU)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    DTYPE = torch.bfloat16
    AUTOCast_KW = dict(device_type="cuda", dtype=DTYPE)
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    DTYPE = torch.bfloat16  # match TorchScript internal cast on MPS
    AUTOCast_KW = None  # autocast isn't widely supported on MPS
else:
    DEVICE = torch.device("cpu")
    DTYPE = torch.float32
    AUTOCast_KW = None  # No autocast on CPU for maximum compatibility

print(f"Using device={DEVICE}, dtype={DTYPE}")

torch.set_float32_matmul_precision("high")

# Prefer local weights if present, else download from Hugging Face Hub
local_path = "weights/model.torchscript"
if os.path.exists(local_path):
    model_path = local_path
else:
    repo_id = "yermandy/deepfake-detection"
    filename = "model.torchscript"
    model_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir="weights")

# Load TorchScript model on CPU to avoid MPS float64 dtype issues, then cast and move
model = torch.jit.load(model_path, map_location="cpu")
# Put model in eval mode and move to correct dtype/device
model.eval()
model = model.to(DTYPE).to(DEVICE)

# Load preprocessing function
preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Load some images from a folder (recursive)
# Default to repo-relative datasets/inca/09_Tampered images, allow CLI override as first arg
_repo_root = os.path.dirname(os.path.abspath(__file__))
_default_dir = os.path.join(_repo_root, "datasets", "inca", "09_Tampered images")
image_dir = sys.argv[1] if len(sys.argv) > 1 else _default_dir
image_dir = os.path.abspath(os.path.expanduser(image_dir))

if not os.path.isdir(image_dir):
    print(f"Directory does not exist: {image_dir}")
    sys.exit(1)

# Collect common image extensions
_exts = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp", "*.tiff", "*.tif", "*.gif")
paths = []
for _e in _exts:
    # recursive search
    paths.extend(glob(os.path.join(image_dir, "**", _e), recursive=True))
# Sort for deterministic order and drop hidden files
paths = sorted(p for p in paths if not os.path.basename(p).startswith("."))

print(f"Found {len(paths)} images in {image_dir} (recursive)")

if not paths:
    print(f"No images found in: {image_dir}")
    sys.exit(0)

# Move inputs to the correct dtype first, then device
# (processing will happen per-batch below)

# Inference in batches to limit memory usage
BATCH_SIZE = int(os.environ.get("BATCH", "16"))

def _chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]

processed = 0

with torch.no_grad():
    for batch_paths in _chunked(paths, BATCH_SIZE):
        # To pillow images
        pillow_images = []
        for image_path in batch_paths:
            try:
                pillow_images.append(Image.open(image_path))
            except Exception as e:
                print(f"Skipping unreadable image: {image_path} ({e})")
        if not pillow_images:
            continue

        # To tensors
        batch_images = torch.stack([
            preprocess(images=image, return_tensors="pt")["pixel_values"][0]
            for image in pillow_images
        ])

        # Move to dtype/device
        batch_images = batch_images.to(DTYPE).to(DEVICE)

        # Forward pass
        logits = model(batch_images)

        # Post-process on CPU in float32 to avoid bfloat16 limitations
        logits = logits.detach().to("cpu", dtype=torch.float32)
        softmax_output = torch.softmax(logits, dim=1).numpy()

        for path, (p_real, p_fake) in zip(batch_paths, softmax_output):
            print(f"p(real) = {p_real:.4f}, p(fake) = {p_fake:.4f}, image: {path}")

        processed += len(batch_paths)

        # Close images to free resources
        for im in pillow_images:
            try:
                im.close()
            except Exception:
                pass

print(f"Processed {processed} images.")

