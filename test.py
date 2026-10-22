
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm

from src.data import list_images, tensor_to_rgb8
from src.model import LGAN
import src.metrics as image_metrics


def image_index(folder):
    paths = list_images(folder)
    index = {path.stem.lower(): path for path in paths}
    if len(index) != len(paths):
        raise ValueError(f"Duplicate image stems in: {folder}")
    if not index:
        raise ValueError(f"No images found in: {folder}")
    return index


def read_resized(path):
    with Image.open(path) as image:
        return image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="/root/autodl-tmp/LSUI/input", help="Original UIEB/LSUI raw/test image folder")
    parser.add_argument("--target", default="/root/autodl-tmp/LSUI/GT", help="Original UIEB/LSUI reference/GT folder")
    parser.add_argument("--checkpoint",default="/root/autodl-tmp/LSUI/enhanced/best.pth", help="Trained best.pth or model state dict")
    parser.add_argument("--output", default="/root/autodl-tmp/LSUI/test_output", help="Output folder for 256x256 predictions, metrics.csv and summary.txt")
    parser.add_argument("--split", default="/root/autodl-tmp/LSUI/enhanced/split.json",help="Explicit split.json containing train/validation filename lists")
    parser.add_argument("--subset", choices=("validation", "train"), default="validation",
                        help="Saved split to evaluate; train evaluates training images without updating weights")
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--metric-details", action="store_true", help="Print/save raw terms and contributions")
    args = parser.parse_args()
    profile = image_metrics.CORRECTED_PROFILE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model" in checkpoint and checkpoint.get("model_variant") != "LGAN-parallel-v1":
        raise ValueError("Checkpoint is not LGAN-parallel-v1; earlier models used a different branch layout")
    config = checkpoint.get("model_config", {})
    model = LGAN(features=config.get("features", args.features),
                 num_blocks=config.get("num_blocks", args.blocks))
    model.load_state_dict(checkpoint.get("model", checkpoint))
    model.to(device).eval()

    inputs, targets = image_index(args.input), image_index(args.target)
    split = checkpoint.get("data_split")
    split_source = "checkpoint data_split"
    sidecar = checkpoint_path.parent / "split.json"
    if args.split or (split is None and sidecar.is_file()):
        split_path = Path(args.split) if args.split else sidecar
        with split_path.open(encoding="utf-8") as stream:
            split = json.load(stream)
        split_source = str(split_path.resolve())
    if split is not None:
        names = split.get(args.subset) if isinstance(split, dict) else None
        if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
            raise ValueError(f"Split must contain a nonempty {args.subset} filename list")
        keys = [Path(name).stem.lower() for name in names]
        if len(set(keys)) != len(keys):
            raise ValueError(f"Duplicate {args.subset} image stems in split")
        print(f"Subset: {args.subset} | Split source: {split_source}", flush=True)
    else:
        if args.subset == "train":
            raise ValueError("--subset train requires checkpoint data_split or --split/sibling split.json")
        keys = sorted(inputs)
        print("No validation split found: evaluating ALL images in --input; supply a test-only folder.", flush=True)
    missing_inputs = [key for key in keys if key not in inputs]
    missing_targets = [key for key in keys if key not in targets]
    if missing_inputs or missing_targets:
        raise ValueError(f"Missing inputs: {missing_inputs[:10]}; missing GT: {missing_targets[:10]}")

    output = Path(args.output).resolve()
    if output in (Path(args.input).resolve(), Path(args.target).resolve()):
        raise ValueError("--output must differ from the input and GT folders")
    output.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | Pairs: {len(keys)}", flush=True)
    print(f"Metrics module: {Path(image_metrics.__file__).resolve()}", flush=True)
    print(f"Metric profile: {profile}", flush=True)
    print("Protocol: raw -> RGB -> Pillow bilinear 256x256 -> model float output; "
          "GT -> RGB -> Pillow bilinear 256x256 -> float [0,1]; evaluate before saving PNG, no crop.", flush=True)
    rows = []
    with torch.inference_mode():
        for key in tqdm(keys, desc="Inferring and evaluating at 256x256"):
            x = TF.to_tensor(read_resized(inputs[key])).unsqueeze(0).to(device)
            enhanced = model(x)[0]
            prediction = tensor_to_rgb8(enhanced)
            reference = TF.to_tensor(read_resized(targets[key]))
            if prediction.shape != (256, 256, 3):
                raise ValueError(f"Unexpected model output shape: {prediction.shape}")
            scores = image_metrics.compute_metrics(enhanced, reference, details=args.metric_details)
            Image.fromarray(prediction).save(output / (inputs[key].stem + ".png"))
            rows.append({"filename": inputs[key].name, **scores})

    metric_names = ["MSE", "PSNR", "SSIM", "UCIQE", "UIQM"]
    metric_names += [name for name in rows[0] if name not in ["filename", *metric_names]]
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["filename", *metric_names])
        writer.writeheader()
        writer.writerows(rows)
    # Average per-image scores, rather than PSNR of a pooled MSE.
    summary = (f"Subset: {args.subset if split is not None else 'all input images'}\n"
               f"Checkpoint: {checkpoint_path.resolve()}\n"
               f"Pairs: {len(rows)}\n"
               f"Metrics module: {Path(image_metrics.__file__).resolve()}\n"
               f"Metric profile: {profile}\n"
               "Metric input: floating model output before PNG quantization\n"
               + "".join(f"Mean {name}: {np.mean([row[name] for row in rows]):.6f}\n"
                         for name in metric_names))
    (output / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")
    print(f"256x256 predictions, metrics.csv and summary.txt saved to: {output}")


if __name__ == "__main__":
    main()
