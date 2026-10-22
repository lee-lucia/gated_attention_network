import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from src.model import LGAN
from src.losses import ReconstructionLoss
from src.data import training_datasets, tensor_to_rgb8
from benchmark import benchmark_model


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def phase_time(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


@torch.no_grad()
def validate(model, loader, device, criterion):
    model.eval()
    totals = {"test_loss": 0.0, "test_l1": 0.0}
    count = 0
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        pred = model(x)
        totals["test_loss"] += criterion(pred, y).item() * x.size(0)
        totals["test_l1"] += torch.mean(torch.abs(pred - y)).item() * x.size(0)
        count += x.size(0)
    if count == 0:
        raise ValueError("The test/validation loader is empty")
    return {name: value / count for name, value in totals.items()}


@torch.no_grad()
def save_enhanced_images(model, loader, device, output_dir):
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for x, _, names in loader:
        predictions = model(x.to(device, non_blocking=True))
        for prediction, name in zip(predictions, names):
            Image.fromarray(tensor_to_rgb8(prediction)).save(output_dir / (Path(name).stem + ".png"))
            count += 1
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="/root/autodl-tmp/LSUI",
                    help="Pre-split train/val folders, UIEB raw/reference, or LSUI input/GT")
    ap.add_argument("--save-dir", type=str,default="/root/autodl-tmp/LSUI/enhanced",
                    help="Directory to save checkpoints, logs, and best enhanced images")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--patch-size", type=int, default=256, help="Resize training pairs to this square size")
    ap.add_argument("--train-ratio", type=float, default=0.7,
                    help="Split fraction for flat UIEB/LSUI folders")
    ap.add_argument("--lr", type=float, default=5e-3)#
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--ssim-weight", type=float, default=0.5, help="SSIM loss weight.")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--features", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--speed-warmup", type=int, default=10)
    ap.add_argument("--speed-iterations", type=int, default=100)
    args = ap.parse_args()
    if args.speed_warmup < 0 or args.speed_iterations < 1:
        ap.error("speed-warmup must be nonnegative and speed-iterations must be positive")
    if args.save_dir is None:
        args.save_dir = (Path(args.resume).resolve().parent if args.resume else
                         Path(__file__).resolve().parent / "runs" / f"lgan_mse_{Path(args.data).resolve().name.lower()}")
    args.save_dir = str(Path(args.save_dir).resolve())

    seed_everything(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_set, val_set = training_datasets(args.data, args.patch_size, args.train_ratio)
    split = {"train": [ip.name for ip, _ in train_set.pairs],
             "validation": [ip.name for ip, _ in val_set.pairs]}
    print(f"Dataset: {len(train_set)} training / {len(val_set)} validation pairs")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
        drop_last=False)
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda"))

    model = LGAN(features=args.features, num_blocks=args.blocks).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr)
    criterion = ReconstructionLoss(args.ssim_weight).to(device)
    print(f"Loss: MSE + {args.ssim_weight:g} * (1 - SSIM); best checkpoint uses validation L1", flush=True)
    start_epoch, best_val = 1, float("inf")
    best_summary = None
    images_dir = Path(args.save_dir) / "enhanced_best"

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        if ckpt.get("loss_variant") != "mse-ssim-zero-padding-v1":
            raise ValueError("Resume requires an MSE-version checkpoint; start a new run for the MSE experiment")
        if ckpt.get("model_variant") != "LGAN-parallel-v1":
            raise ValueError("Resume requires an LGAN-parallel-v1 checkpoint; earlier models used serial branches")
        if ckpt["model_config"] != {"features": args.features, "num_blocks": args.blocks}:
            raise ValueError("Resume --features/--blocks must match the saved model_config")
        if ckpt["training_config"]["ssim_weight"] != args.ssim_weight:
            raise ValueError("Resume --ssim-weight must match the saved training_config")
        if ckpt["data_split"] != split:
            raise ValueError("Resume dataset split differs from the saved data_split")
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = ckpt.get("best_val", best_val)
        previous_best = ckpt.get("best_summary") or ckpt.get("best_metrics")
        if previous_best is not None:
            best_summary = {key: previous_best[key] for key in ("epoch", "train_loss", "test_loss", "test_l1")}
        print(f"Resumed from epoch {start_epoch - 1}")

    with open(os.path.join(args.save_dir, "split.json"), "w", encoding="utf-8") as stream:
        json.dump(split, stream, ensure_ascii=False, indent=2)

    print("\nMeasuring network complexity and batch-1 inference speed at 256x256...", flush=True)
    statistics = benchmark_model(model, torch.zeros(1, 3, 256, 256, device=device),
                                 args.speed_warmup, args.speed_iterations)
    print(f"Params: {statistics['parameters']:,} ({statistics['parameters'] / 1e3:.3f} K)")
    print(f"MACs (incl. GAP): {statistics['thop_style_ops'] / 1e9:.6f} G | "
          f"FLOPs (approx.): {statistics['approximate_flops'] / 1e9:.6f} G")
    print(f"FPS: {statistics['fps']:.2f} | Latency: {statistics['latency_ms']:.3f} ms | "
          f"Warmup: {args.speed_warmup} | Runs: {args.speed_iterations}")
    print("FPS measures model inference only, excluding image I/O and metric computation.")
    print(f"Best enhanced images (exported once after training): {images_dir}", flush=True)
    with open(os.path.join(args.save_dir, "benchmark.json"), "w", encoding="utf-8") as stream:
        json.dump(statistics, stream, indent=2)

    columns = ["epoch", "lr", "train_loss", "test_loss", "test_l1"]
    history_path = Path(args.save_dir) / "history.csv"
    if args.resume and history_path.exists():
        # Keep earlier rows readable when resuming an older, wider CSV.
        with history_path.open(newline="", encoding="utf-8") as stream:
            columns = next(csv.reader(stream))
    else:
        with history_path.open("w", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=columns).writeheader()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = phase_time(device)
        model.train()
        running, count = 0.0, 0
        epoch_lr = optimizer.param_groups[0]["lr"]
        for x, y, _ in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
            count += x.size(0)

        scheduler.step()
        train_loss = running / max(count, 1)
        train_end = phase_time(device)
        values = validate(model, val_loader, device, criterion)
        test_end = phase_time(device)
        val_loss = values["test_l1"]
        print(f"Epoch [{epoch}/{args.epochs}] train_loss={train_loss:.6f} | "
              f"test_loss={values['test_loss']:.6f} | test_L1={val_loss:.6f} | lr={epoch_lr:.2e}", flush=True)
        with history_path.open("a", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=columns).writerow(
                dict(epoch=epoch, lr=epoch_lr, train_loss=train_loss, **values))

        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            best_summary = dict(epoch=epoch, train_loss=train_loss, **values)
        state = {
            "model_variant": "LGAN-parallel-v1",
            "loss_variant": "mse-ssim-zero-padding-v1",
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "best_val": best_val,
            "model_config": {
                "features": args.features, "num_blocks": args.blocks,
            },
            "training_config": vars(args),
            "data_split": split,
            "test_losses": values,
            "best_summary": best_summary,
        }
        torch.save(state, os.path.join(args.save_dir, "latest.pth"))
        if is_best:
            torch.save(state, os.path.join(args.save_dir, "best.pth"))
            with open(os.path.join(args.save_dir, "best_loss.json"), "w", encoding="utf-8") as stream:
                json.dump(best_summary, stream, indent=2)
            print("  Saved best.pth (image export deferred until training finishes)", flush=True)
        save_end = phase_time(device)
        print(f"  Time: train={train_end - epoch_start:.2f}s | test={test_end - train_end:.2f}s | "
              f"checkpoint/log={save_end - test_end:.2f}s | total={save_end - epoch_start:.2f}s", flush=True)

    print(f"Training finished. Best validation L1: {best_val:.6f}")
    if best_summary is not None:
        print(f"Best epoch: {best_summary['epoch']} | test_loss={best_summary['test_loss']:.6f}")
    print(f"Checkpoints and logs: {args.save_dir}")
    best_path = Path(args.save_dir) / "best.pth"
    if best_path.is_file():
        print("Exporting best-model enhanced images once...", flush=True)
        export_start = phase_time(device)
        best_state = torch.load(best_path, map_location=device)
        model.load_state_dict(best_state["model"])
        saved = save_enhanced_images(model, val_loader, device, images_dir)
        export_end = phase_time(device)
        print(f"Exported {saved} enhanced images in {export_end - export_start:.2f}s", flush=True)
        print(f"Enhanced test/validation images (best model): {images_dir}")
        print("Run metrics.py separately on this folder to calculate image-quality metrics once.")
    else:
        print(f"Image export skipped: best checkpoint is missing: {best_path}")


if __name__ == "__main__":
    main()
