import torch
import numpy as np
import torch.utils.benchmark as TBenchmark
from triton import testing
from robust_kbench.sandbox.results import TimeEvalResult


def time_function_kernel_bench(
    func,
    *args,
    warmup_time: int = 25,
    n_iter: int = 10000,
    kernel_name: str | None = None,
) -> TimeEvalResult:
    """Time a function using Cuda events."""
    # https://github.com/ScalingIntelligence/KernelBench/blob/5c88b2319076e8d44b9901914de7b45d220944e9/src/eval.py#L628
    # Warmup trials
    for _ in range(warmup_time):
        func(*args)
        torch.cuda.synchronize(device="cuda")

    elapsed_times = []

    # Runtime evaluation trials
    for _ in range(n_iter):
        # create event marker default is not interprocess
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        func(*args)
        end_event.record()
        # Synchronize to ensure the events have completed
        torch.cuda.synchronize(device="cuda")
        # Calculate the elapsed time in milliseconds
        elapsed_time_ms = start_event.elapsed_time(end_event)
        elapsed_times.append(elapsed_time_ms)

    return TimeEvalResult(
        mean_time=np.mean(elapsed_times),
        median_time=np.median(elapsed_times),
        iqr_time=np.percentile(elapsed_times, 75) - np.percentile(elapsed_times, 25),
    )


def time_function_torch_bench(
    func,
    *args,
    warmup_time: int = 25,
    n_iter: int = 10000,
    kernel_name: str | None = None,
) -> TimeEvalResult:
    """Time a function using torch.benchmark."""
    globals = {
        "args": args,
        "fn": func,
    }
    # Not necessarily needed?
    for _ in range(warmup_time):
        func(*args)
        torch.cuda.synchronize(device="cuda")

    # Runtime evaluation trials with torch.benchmark
    measurement = TBenchmark.Timer(
        stmt="fn(*args)",
        globals=globals,
        label=None,
        sub_label=None,
        description=None,
    ).blocked_autorange(min_run_time=1)
    return TimeEvalResult(
        mean_time=measurement.mean * 1000,
        median_time=measurement.median * 1000,
        iqr_time=measurement.iqr * 1000,
    )


def time_function_triton(
    func,
    *args,
    warmup_time: int = 25,
    n_iter: int = 10000,
    kernel_name: str | None = None,
) -> TimeEvalResult:
    """Time a function using triton.testing."""

    def wrapper():
        return func(*args)

    # Warmup trials
    for _ in range(warmup_time):
        wrapper()
        torch.cuda.synchronize(device="cuda")

    # Runtime evaluation trials
    elapsed_times = testing.do_bench(
        wrapper,
        warmup=warmup_time,
        rep=n_iter,
        return_mode="all",
    )
    return TimeEvalResult(
        mean_time=np.mean(elapsed_times),
        median_time=np.median(elapsed_times),
        iqr_time=np.percentile(elapsed_times, 75) - np.percentile(elapsed_times, 25),
    )


def time_function_kineto(
    func,
    *args,
    warmup_time: int = 25,
    n_iter: int = 10000,
    kernel_name: str | None = None,
) -> TimeEvalResult:
    """Time a function using PyTorch's Kineto profiler
    Adapted from https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/utils.py#L80
    """

    # By default, flush L2 with an excessive 8GB memset to give the GPU some (literal) chill time without full idle
    # this avoid thermal throttling while keeping DVFS at max clocks (slight gain vs sleep / more consistent on GH200)
    flush_l2_size = int(8e9 // 4)
    if n_iter > 2000:
        # if we want to be thermally limited, we need to run many iterations non-stop for a fairly long time
        # and spend as little time as possible doing memset and other setup work (80MiB should be enough to flush L2)
        flush_l2_size = int(80e6 // 4)

    # Initial warmup call
    for _ in range(warmup_time):
        func(*args)

    # Profile
    schedule = torch.profiler.schedule(wait=0, warmup=warmup_time, active=1, repeat=1)
    profiler = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA], schedule=schedule
    )

    with profiler:
        for _ in range(2):
            # Run the benchmarked function
            for _ in range(n_iter):
                torch.empty(flush_l2_size, dtype=torch.int, device="cuda").zero_()
                func(*args)

            profiler.step()

    # Parse the profiling table
    prof_lines = (
        profiler.key_averages()
        .table(sort_by="cuda_time_total", max_name_column_width=100)
        .split("\n")
    )

    print(f"Warning: Kineto eval reports the cuda total time including the flush L2 time.")

    # Extract the self cuda time total
    # NOTE: the order matters here, we want to check for "s" last
    units = {"ms": 1e3, "us": 1e6, "ns": 1e9, "s": 1}
    kernel_times = []
    for line in prof_lines:
        if "self cuda time total" in line.lower():
            time_str = line.split(":")[-1].strip()
            for unit, scale in units.items():
                if unit in time_str:
                    kernel_times.append(float(time_str.replace(unit, "")) / scale)
                    break
            break
        
    # Profiler directly returns the average time
    return TimeEvalResult(
        mean_time=np.mean(kernel_times),
        median_time=0.0,
        iqr_time=0.0,
    )
