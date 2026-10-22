
import argparse
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as skimage_ssim


EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
CORRECTED_PROFILE = (
    "corrected-v1 | UCIQE float RGB -> signed CIELAB, ALL Lab channels /100, "
    "L99%-L1%, C/L (L=0 -> 0) | UIQM corrected left:right trimmed mean, "
    "full-image variance, RGB[0,255], 15x15 full blocks, "
    "Sobel magnitude (reflect, peak-normalized), RGB weights 0.299/0.587/0.114, "
    "-mean(r*ln(r)) across RGB | explicit convention, not author-verified"
)
UIQM_WEIGHTS = (0.0282, 0.2953, 3.5753)
UCIQE_WEIGHTS = (0.4680, 0.2745, 0.2576)


def _uciqe_terms(img):
    image = _check_image(img)
    lab = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2LAB).astype(np.float64) / 100.0
    luminance, a, b = np.moveaxis(lab, -1, 0)
    chroma = np.sqrt(a ** 2 + b ** 2)
    ordered = np.sort(luminance.ravel())
    contrast = ordered[int(ordered.size * 0.99)] - ordered[int(ordered.size * 0.01)]
    saturation = np.divide(chroma, luminance, out=np.zeros_like(chroma), where=luminance != 0)
    return float(np.std(chroma)), float(contrast), float(np.mean(saturation))


def _uiqm_terms(img):
    """Return corrected UICM, UISM and UIConM using 15x15 full blocks."""
    image = _check_image(img).astype(np.float64) * 255.0
    if min(image.shape[:2]) < 15:
        raise ValueError("UIQM requires height and width >= 15")
    r, g, b = np.moveaxis(image, -1, 0)
    means, variances = [], []
    for opponent in (r - g, (r + g) / 2 - b):
        ordered = np.sort(opponent.ravel())
        left = math.ceil(0.1 * ordered.size)
        right = ordered.size - math.floor(0.1 * ordered.size)
        mean = ordered[left:right].mean()
        means.append(mean)
        variances.append(np.mean((opponent - mean) ** 2))
    uicm = -0.0268 * np.hypot(*means) + 0.1586 * np.sqrt(sum(variances))
    block_size = 15
    rows = image.shape[0] // block_size
    cols = image.shape[1] // block_size

    def extrema(values):
        blocks = values[
            :rows * block_size, :cols * block_size
            ].reshape(rows, block_size, cols, block_size, -1)
        return blocks.min(axis=(1, 3, 4)), blocks.max(axis=(1, 3, 4))

    uism = 0.0
    for channel, weight in zip((r, g, b), (0.299, 0.587, 0.114)):
        dx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT)
        dy = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT)
        edge = np.hypot(dx, dy)
        peak = edge.max()
        if peak > 0:
            edge *= 255.0 / peak
        low, high = extrema(edge * channel)
        valid = (low > 0) & (high > 0)
        uism += weight * 2 * np.log(high[valid] / low[valid]).sum() / low.size
    low, high = extrema(image)
    valid = (high > low) & (high + low > 0)
    ratio = (high[valid] - low[valid]) / (high[valid] + low[valid])
    uiconm = -np.sum(ratio * np.log(ratio)) / low.size
    return float(uicm), float(uism), float(uiconm)


def corrected_uciqe(img):
    return float(sum(w * term for w, term in zip(UCIQE_WEIGHTS, _uciqe_terms(img))))


def corrected_uiqm(img):
    return float(sum(w * term for w, term in zip(UIQM_WEIGHTS, _uiqm_terms(img))))


def metric_components(img):
    result = {}
    for metric, names, weights, terms in (
        ("UIQM", ("UICM", "UISM", "UIConM"), UIQM_WEIGHTS, _uiqm_terms(img)),
        ("UCIQE", ("sigma_C", "con_L", "mu_S"), UCIQE_WEIGHTS, _uciqe_terms(img)),
    ):
        result[metric] = float(sum(w * term for w, term in zip(weights, terms)))
        for name, weight, term in zip(names, weights, terms):
            result[f"{metric}_{name}"] = term
            result[f"{metric}_{name}_contribution"] = weight * term
    return result


def read_rgb(path, resize=None):
    with Image.open(path) as image:
        image = image.convert("RGB")
        if resize is not None:
            image = image.resize(tuple(resize), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32) / 255.0


def _check_image(image):
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy().transpose(1, 2, 0)
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) == 0:
        raise ValueError("Expected a nonempty H x W x 3 RGB array or C x H x W tensor")
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    elif not np.issubdtype(image.dtype, np.floating):
        raise ValueError("Expected RGB uint8 or floating values in [0, 1]")
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("Expected finite RGB floating values in [0, 1]")
    return image


def manual_uciqe(img):

    return corrected_uciqe(img)


def manual_uiqm(img):

    return corrected_uiqm(img)


def getUIQM(img_rgb):

    return corrected_uiqm(img_rgb)


def getUCIQE(img_BGR):

    return corrected_uciqe(_check_image(img_BGR)[:, :, ::-1])


def reference_metrics(enhanced, reference):
    """MSE/PSNR/SSIM on RGB floats, without uint8 quantization."""
    image, target = _check_image(enhanced), _check_image(reference)
    if image.shape != target.shape:
        raise ValueError(f"Prediction/reference sizes differ: {image.shape} vs {target.shape}")
    mse = float(np.mean((image - target) ** 2))
    psnr = float("inf") if mse == 0 else -10 * math.log10(mse)
    try:
        ssim = skimage_ssim(image, target, data_range=1.0, channel_axis=-1)
    except TypeError:
        ssim = skimage_ssim(image, target, data_range=1.0, multichannel=True)
    return {"MSE": mse, "PSNR": psnr, "SSIM": float(ssim)}


def compute_metrics(enhanced, reference=None, details=False):
    """Five corrected scores; omit reference for UCIQE/UIQM only."""
    image = _check_image(enhanced)
    result = reference_metrics(image, reference) if reference is not None else {}
    if details:
        result.update(metric_components(image))
    else:
        result.update(UCIQE=corrected_uciqe(image), UIQM=corrected_uiqm(image))
    return result


def _image_index(folder):
    paths = {}
    for path in sorted(Path(folder).iterdir()):
        if path.is_file() and path.suffix.lower() in EXTS:
            key = path.stem.lower()
            if key in paths:
                raise ValueError(f"Duplicate image stem: {paths[key]} and {path}")
            paths[key] = path
    if not paths:
        raise RuntimeError(f"No images found in: {folder}")
    return paths


def _validation_predictions(predictions, split_path):
    with Path(split_path).open(encoding="utf-8") as stream:
        names = json.load(stream).get("validation")
    if not isinstance(names, list) or not names or not all(isinstance(name, str) for name in names):
        raise ValueError(f"Expected a nonempty validation filename list in {split_path}")
    keys = [Path(name).stem.lower() for name in names]
    if len(set(keys)) != len(keys):
        raise ValueError(f"Duplicate validation image stems in {split_path}")
    missing = [name for name, key in zip(names, keys) if key not in predictions]
    if missing:
        raise ValueError(f"Missing {len(missing)} validation predictions: {missing[:10]}")
    selected = {key: predictions[key] for key in keys}
    print(f"Validation split: {Path(split_path).absolute()}")
    print(f"Selected validation predictions: {len(selected)}; ignored files outside split: {len(predictions) - len(selected)}")
    return selected


def _check_pair_sizes(predictions, targets):
    mismatches = []
    for key in sorted(predictions.keys() & targets.keys()):
        with Image.open(predictions[key]) as image:
            pred_size = image.size
        with Image.open(targets[key]) as image:
            ref_size = image.size
        if pred_size != ref_size:
            mismatches.append(f"pred={predictions[key].absolute()} {pred_size}; "
                              f"target={targets[key].absolute()} {ref_size}")
    if mismatches:
        raise ValueError(f"Prediction/reference sizes differ for {len(mismatches)} pairs (width, height):\n"
                         + "\n".join(mismatches[:10])
                         + "\nCheck dataset paths and prediction provenance; no automatic resizing was performed.")


def main(default_resize=None):
    p = argparse.ArgumentParser()
    p.add_argument("--metric-details", action="store_true",
                   help="Print raw terms and weighted contributions")
    p.add_argument("--pred", default="/root/autodl-tmp/LSUI/enhanced/enhanced_best", help="Prediction folder; default: enhanced_best")
    p.add_argument("--target",default="/root/autodl-tmp/LSUI/GT", help="Optional reference folder for MSE/PSNR/SSIM")
    p.add_argument("--split", default="/root/autodl-tmp/LSUI/enhanced/split.json",help="Validation split.json; auto-detected beside an enhanced_best folder")
    p.add_argument("--resize", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=(256, 256),
                   help="Resize prediction and reference in memory with Pillow bilinear before all metrics")
    args = p.parse_args()
    minimum_size = 15
    if args.resize is not None and min(args.resize) < minimum_size:
        p.error(f"--resize WIDTH HEIGHT must both be >= {minimum_size} for UIQM")
    predictions = _image_index(args.pred)
    pred_folder = Path(args.pred)
    automatic_split = pred_folder.parent / "split.json"
    split_path = args.split or (automatic_split if pred_folder.name == "enhanced_best" and automatic_split.is_file() else None)
    if split_path:
        predictions = _validation_predictions(predictions, split_path)
    targets = _image_index(args.target) if args.target else {}
    print(f"Prediction folder: {pred_folder.absolute()}", flush=True)
    if args.target:
        print(f"Reference folder: {Path(args.target).absolute()}", flush=True)
        if split_path:
            missing = sorted(predictions.keys() - targets.keys())
            if missing:
                raise ValueError(f"Missing references for {len(missing)} validation predictions: {missing[:10]}")
        _check_pair_sizes(predictions, targets)
    pair_count = len(predictions.keys() & targets.keys())
    if args.target and not pair_count:
        raise RuntimeError("No matched prediction/target filenames found.")
    if args.resize is not None:
        print(f"Evaluation size (width x height): {args.resize[0]} x {args.resize[1]} | "
              "Pillow bilinear | resize before all metrics | no crop", flush=True)
    else:
        print("Evaluation size: original image sizes", flush=True)
    scores = {name: [] for name in ("MSE", "PSNR", "SSIM", "UIQM", "UCIQE")}
    for key, path in predictions.items():
        reference = read_rgb(targets[key], args.resize) if key in targets else None
        try:
            metrics = compute_metrics(read_rgb(path, args.resize), reference, details=args.metric_details)
        except ValueError as error:
            raise ValueError(f"{path.name}: {error}") from error
        for name, value in metrics.items():
            scores.setdefault(name, []).append(value)
    print(f"Metric profile: {CORRECTED_PROFILE}")
    print(f"Images (UIQM/UCIQE): {len(predictions)}")
    if args.target:
        print(f"Pairs (MSE/PSNR/SSIM): {pair_count}; unmatched predictions: {len(predictions) - pair_count}")
    for name, values in scores.items():
        if values:
            print(f"Mean {name}: {np.mean(values):.6f}")


if __name__ == "__main__":
    main()
