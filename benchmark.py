
import argparse
import platform
import time

import torch
from torch import nn

from src.model import LGAN


@torch.no_grad()
def operation_counts(model, image):
    counts = {"conv_linear_macs": 0, "pool_ops": 0}

    def count(module, inputs, output):
        if isinstance(module, nn.Conv2d):
            counts["conv_linear_macs"] += (output.numel() * (module.in_channels // module.groups)
                                           * module.kernel_size[0] * module.kernel_size[1])
        elif isinstance(module, nn.Linear):
            counts["conv_linear_macs"] += output.numel() * module.in_features
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            counts["pool_ops"] += inputs[0].numel() + output.numel()

    hooks = [module.register_forward_hook(count) for module in model.modules()
             if isinstance(module, (nn.Conv2d, nn.Linear, nn.AdaptiveAvgPool2d))]
    try:
        model(image)
    finally:
        for hook in hooks:
            hook.remove()
    counts["parameters"] = sum(parameter.numel() for parameter in model.parameters())
    counts["thop_style_ops"] = counts["conv_linear_macs"] + counts["pool_ops"]
    return counts


@torch.inference_mode()
def benchmark_model(model, image, warmup=10, iterations=100):
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup must be nonnegative and iterations must be positive")
    was_training = model.training
    model.eval()
    try:
        counts = operation_counts(model, image)
        for _ in range(warmup):
            model(image)
        if image.is_cuda:
            torch.cuda.synchronize(image.device)
        start = time.perf_counter()
        for _ in range(iterations):
            model(image)
        if image.is_cuda:
            torch.cuda.synchronize(image.device)
        elapsed = time.perf_counter() - start
    finally:
        model.train(was_training)
    return dict(counts, approximate_flops=2 * counts["thop_style_ops"],
                latency_ms=elapsed / iterations * 1000, fps=iterations / elapsed,
                input_shape=list(image.shape), device=str(image.device),
                hardware=torch.cuda.get_device_name(image.device) if image.is_cuda else platform.processor(),
                torch_version=str(torch.__version__), cpu_threads=torch.get_num_threads(),
                warmup=warmup, iterations=iterations)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--threads", type=int, default=0, help="0 keeps PyTorch's default CPU threads")
    args = parser.parse_args()
    if args.size < 1 or args.warmup < 0 or args.iterations < 1 or args.threads < 0:
        parser.error("size/iterations must be positive; warmup/threads must be nonnegative")
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    model = LGAN(args.features, args.blocks).to(device).eval()
    image = torch.rand(1, 3, args.size, args.size, device=device)
    counts = benchmark_model(model, image, args.warmup, args.iterations)
    print(f"Device: {device}; hardware: {torch.cuda.get_device_name(0) if device.type == 'cuda' else platform.processor()}")
    print(f"PyTorch: {torch.__version__}; CPU threads: {torch.get_num_threads()}; input: {tuple(image.shape)}")
    print(f"Parameters: {counts['parameters']:,}")
    print(f"Conv/Linear MACs: {counts['conv_linear_macs'] / 1e9:.9f} G")
    print(f"THOP-style ops (includes GAP): {counts['thop_style_ops'] / 1e9:.9f} G")
    print(f"Approx. FLOPs (2 x THOP-style ops): {2 * counts['thop_style_ops'] / 1e9:.9f} G")
    print(f"Measured latency: {counts['latency_ms']:.3f} ms; FPS: {counts['fps']:.2f}")
    print("Batch-1 model inference only, random weights; excludes I/O. FPS depends on hardware and runtime.")


if __name__ == "__main__":
    main()
