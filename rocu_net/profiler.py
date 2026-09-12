from __future__ import annotations

import io
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .utils import synchronize


def count_conv_linear_macs(model: nn.Module, example: torch.Tensor) -> int:
    """Count multiply-accumulates for Conv2d and Linear modules."""
    total = 0
    hooks = []

    def conv_hook(module: nn.Conv2d, inputs, output):
        nonlocal total
        del inputs
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (
            module.in_channels // module.groups
        )
        total += int(output.numel() * kernel_ops)

    def linear_hook(module: nn.Linear, inputs, output):
        nonlocal total
        del inputs
        total += int(output.numel() * module.in_features)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
    try:
        with torch.inference_mode():
            model(example)
    finally:
        for hook in hooks:
            hook.remove()
    return total


def _device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple Metal Performance Shaders"
    return platform.processor() or platform.machine() or "CPU"


def _state_dict_size_bytes(model: nn.Module) -> int:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell()


@torch.inference_mode()
def benchmark_latency(
    model: nn.Module,
    example: torch.Tensor,
    device: torch.device,
    *,
    warmup_iterations: int = 30,
    benchmark_iterations: int = 100,
) -> tuple[list[float], dict[str, float | None]]:
    model.eval()
    for _ in range(int(warmup_iterations)):
        model(example)
    synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    rss_before = None
    try:
        import psutil

        rss_before = psutil.Process().memory_info().rss
    except ImportError:
        pass

    timings_ms: list[float] = []
    if device.type == "cuda":
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(benchmark_iterations)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(benchmark_iterations)]
        for start, end in zip(starts, ends):
            start.record()
            model(example)
            end.record()
        synchronize(device)
        timings_ms = [float(start.elapsed_time(end)) for start, end in zip(starts, ends)]
    else:
        for _ in range(int(benchmark_iterations)):
            synchronize(device)
            start_time = time.perf_counter()
            model(example)
            synchronize(device)
            timings_ms.append((time.perf_counter() - start_time) * 1000.0)

    rss_after = None
    try:
        import psutil

        rss_after = psutil.Process().memory_info().rss
    except ImportError:
        pass
    memory = {
        "peak_cuda_allocated_mb": (
            torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else None
        ),
        "peak_cuda_reserved_mb": (
            torch.cuda.max_memory_reserved(device) / (1024**2) if device.type == "cuda" else None
        ),
        "cpu_rss_change_mb": (
            (rss_after - rss_before) / (1024**2)
            if rss_before is not None and rss_after is not None
            else None
        ),
    }
    return timings_ms, memory


def profile_model(
    model: nn.Module,
    device: torch.device,
    *,
    image_size: tuple[int, int] = (352, 352),
    batch_size: int = 1,
    in_channels: int = 3,
    warmup_iterations: int = 30,
    benchmark_iterations: int = 100,
    checkpoint_path: str | Path | None = None,
) -> dict:
    model = model.to(device).eval()
    example = torch.randn(batch_size, in_channels, *image_size, device=device)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    macs_batch = count_conv_linear_macs(model, example)
    macs_per_image = macs_batch / batch_size
    with torch.inference_mode():
        diagnostic_outputs = model(example)
    routing: dict[str, float | int | bool] | None = None
    if isinstance(diagnostic_outputs, dict) and "routing_gate_1" in diagnostic_outputs:
        fractions = [
            float(diagnostic_outputs[key].float().mean())
            for key in ("routing_fraction_1", "routing_fraction_2")
        ]
        gate_means = [
            float(diagnostic_outputs[key].float().mean())
            for key in ("routing_gate_1", "routing_gate_2")
        ]
        gate_cells = [
            int(diagnostic_outputs[key].numel())
            for key in ("routing_gate_1", "routing_gate_2")
        ]
        total_cells = sum(gate_cells)
        active_cells = sum(
            int(round(fraction * cells))
            for fraction, cells in zip(fractions, gate_cells)
        )
        routing = {
            "stage1_active_fraction": fractions[0],
            "stage2_active_fraction": fractions[1],
            "stage1_gate_mean": gate_means[0],
            "stage2_gate_mean": gate_means[1],
            "solver_parent_cells_total": total_cells,
            "solver_parent_cells_active_estimate": active_cells,
            "solver_parent_cells_skipped_estimate": total_cells - active_cells,
            "solver_skip_fraction_estimate": (
                (total_cells - active_cells) / total_cells if total_cells else 0.0
            ),
            "hard_routing_inference": bool(
                getattr(getattr(model, "rocu1", None), "hard_routing_inference", False)
            ),
        }
    timings_ms, memory = benchmark_latency(
        model,
        example,
        device,
        warmup_iterations=warmup_iterations,
        benchmark_iterations=benchmark_iterations,
    )
    mean_latency = float(statistics.fmean(timings_ms))
    checkpoint_size_mb = None
    if checkpoint_path is not None and Path(checkpoint_path).is_file():
        checkpoint_size_mb = Path(checkpoint_path).stat().st_size / (1024**2)
    solver_iterations = None
    if hasattr(model, "occupancy1"):
        solver_iterations = int(model.occupancy1.solver_iterations)
    elif hasattr(model, "rocu1"):
        solver_iterations = int(model.rocu1.solver_iterations)
    result = {
        "inference": {
            "tta": getattr(model, "tta", "none"),
            "forward_passes": 4 if getattr(model, "tta", "none") == "flip" else 1,
        },
        "input": {
            "shape": [batch_size, in_channels, image_size[0], image_size[1]],
            "batch_size": batch_size,
        },
        "parameters": {
            "total": int(total_parameters),
            "trainable": int(trainable_parameters),
            "millions": total_parameters / 1e6,
            "fp32_parameter_size_mb": total_parameters * 4 / (1024**2),
            "serialized_state_dict_mb": _state_dict_size_bytes(model) / (1024**2),
            "checkpoint_file_mb": checkpoint_size_mb,
        },
        "computation": {
            "macs_per_image": int(macs_per_image),
            "gmacs_per_image": macs_per_image / 1e9,
            "flops_per_image": int(2 * macs_per_image),
            "gflops_per_image": 2 * macs_per_image / 1e9,
            "flops_convention": "FLOPs = 2 x MACs",
            "macs_scope": "Conv2d and Linear only; occupancy solver elementwise operations excluded",
            "solver_iterations_per_block": solver_iterations,
        },
        "runtime": {
            "warmup_iterations": int(warmup_iterations),
            "benchmark_iterations": int(benchmark_iterations),
            "latency_mean_ms_per_batch": mean_latency,
            "latency_median_ms_per_batch": float(np.median(timings_ms)),
            "latency_p95_ms_per_batch": float(np.percentile(timings_ms, 95)),
            "images_per_second": batch_size * 1000.0 / mean_latency,
        },
        "memory": memory,
        "environment": {
            "device_type": device.type,
            "device_name": _device_name(device),
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cpu_threads": torch.get_num_threads(),
        },
    }
    if routing is not None:
        result["confidence_routing"] = routing
    return result
