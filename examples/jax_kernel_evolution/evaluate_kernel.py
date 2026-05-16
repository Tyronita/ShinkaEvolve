#!/usr/bin/env python3
"""
JAX Kernel Evolution — evaluation harness.

Invoked by ShinkaEvolve as a subprocess for every proposal:
    python evaluate_kernel.py --program_path <mutated.cc> --results_dir <out>

Writes metrics.json  { combined_score: float, execution_time_mean: float, ... }
         correct.json { correct: bool, error: str | None }

Mode:
  - dry-run: syntax-check only (g++ -fsyntax-only), no GPU needed
  - full:     Docker-based compile + JAX benchmark (VM)
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


EVOLVE_START = re.compile(r"(?://|#|!)?\s*EVOLVE-BLOCK-START")
EVOLVE_END   = re.compile(r"(?://|#|!)?\s*EVOLVE-BLOCK-END")

DEFAULT_METRICS_ON_ERROR = {
    "combined_score": 0.0,
    "execution_time_mean": 0.0,
    "execution_time_std": 0.0,
    "num_successful_runs": 0,
    "num_valid_runs": 0,
    "num_invalid_runs": 0,
    "all_validation_errors": [],
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JAX kernel evolution evaluator")
    parser.add_argument("--program_path", type=str, required=True,
                        help="Path to the mutated .cc file")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory to write results")
    parser.add_argument("--jax_repo", type=str, default=None,
                        help="Path to the JAX repository root")
    parser.add_argument("--docker_image", type=str, default=None,
                        help="Docker image for VM-based compilation")
    parser.add_argument("--dry_run", type=str, default="true",
                        help="Dry-run mode (true/false)")
    return parser.parse_args()


def _fingerprint_mutable_regions(code: str) -> str:
    """Compute SHA256 of all EVOLVE-BLOCK content."""
    mutable_parts = []
    starts = [m.end() for m in EVOLVE_START.finditer(code)]
    ends   = [m.start() for m in EVOLVE_END.finditer(code)]
    for s, e in zip(starts, ends):
        mutable_parts.append(code[s:e])
    combined = "\n".join(mutable_parts)
    return hashlib.sha256(combined.encode()).hexdigest()


def _syntax_check(path: str) -> tuple[bool, str]:
    """Run g++ -fsyntax-only on the file. Returns (ok, error_msg)."""
    try:
        result = subprocess.run(
            ["g++", "-fsyntax-only", "-std=c++20", "-x", "c++", path],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return True, ""
        return False, result.stderr[:2000]
    except FileNotFoundError:
        return False, "g++ not found on this system"
    except subprocess.TimeoutExpired:
        return False, "syntax check timed out"
    except Exception as e:
        return False, str(e)


def _compile_with_docker(
    program_path: str,
    jax_repo: str,
    docker_image: str,
    results_dir: str,
) -> tuple[bool, str]:
    """Compile the mutated kernel inside a Docker container.

    Steps:
      1. Copy mutated file into jax_repo (replacing original)
      2. docker run --volume jax_repo:/workspace/jax ...
      3. Build the kernel library
    """
    try:
        program_path = Path(program_path).resolve()
        jax_repo = Path(jax_repo).resolve()
        results_dir = Path(results_dir).resolve()

        # Infer relative path within JAX repo
        try:
            rel = program_path.relative_to(jax_repo)
        except ValueError:
            # If program_path is outside jax_repo (e.g. in shinka results dir),
            # copy it into the correct location
            return False, "program_path must be inside jax_repo for Docker build"

        dst = jax_repo / rel

        # Copy the mutated file to the jax repo
        import shutil
        shutil.copy2(program_path, dst)
        print(f"Copied mutated file to {dst}")

        # Build command — replace with actual build steps
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{jax_repo}:/workspace/jax",
            "-v", f"{results_dir}:/workspace/results",
            docker_image,
            "bash", "-c",
            f"cd /workspace/jax && "
            f"bazel build //jaxlib/{rel.parent.as_posix()}:kernel_lib 2>&1",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            return True, ""
        return False, result.stderr[:2000]
    except FileNotFoundError:
        return False, "Docker not available"
    except subprocess.TimeoutExpired:
        return False, "Docker build timed out (600s)"
    except Exception as e:
        return False, str(e)


def _run_benchmark(
    results_dir: str,
) -> tuple[float, float]:
    """Run a JAX benchmark that exercises the affected kernel functions.

    Returns (mean_execution_time, std_execution_time).
    Placeholder: returns (1.0, 0.0) in dry-run mode.
    """
    # TODO: implement JAX benchmark that calls the specific kernel
    # 1. Run JAX test suite subset
    # 2. Time the specific kernel calls
    # 3. Save benchmark results
    return 1.0, 0.0


def _compute_speedup(baseline_time: float, new_time: float) -> float:
    """Compute speedup. > 1.0 means faster."""
    if new_time <= 0 or baseline_time <= 0:
        return 0.0
    return baseline_time / new_time


def save_results(results_dir: str, metrics: dict, correct: bool, error: str | None = None):
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "correct.json"), "w") as f:
        json.dump({"correct": correct, "error": error}, f, indent=2)
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Results saved to {results_dir}")
    print(f"  correct.json: correct={correct}, error={error}")
    print(f"  metrics.json: combined_score={metrics.get('combined_score', 'N/A')}")


def main():
    args = _parse_args()
    results_dir = args.results_dir
    program_path = args.program_path
    dry_run = args.dry_run.lower() in ("true", "1", "yes")
    os.makedirs(results_dir, exist_ok=True)

    start_time = time.perf_counter()

    try:
        code = Path(program_path).read_text(encoding="utf-8")
    except Exception as e:
        save_results(results_dir, DEFAULT_METRICS_ON_ERROR, False, str(e))
        return

    # 1. Compute fingerprint of mutated regions
    fingerprint = _fingerprint_mutable_regions(code)
    print(f"Mutated regions fingerprint: {fingerprint}")

    # 2. Syntax-check or full Docker compilation
    if dry_run:
        ok, err = _syntax_check(program_path)
    else:
        if not args.jax_repo or not args.docker_image:
            save_results(results_dir, DEFAULT_METRICS_ON_ERROR, False,
                         "full mode requires --jax_repo and --docker_image")
            return
        ok, err = _compile_with_docker(program_path, args.jax_repo,
                                       args.docker_image, results_dir)

    if not ok:
        save_results(results_dir, DEFAULT_METRICS_ON_ERROR, False, err)
        return

    # 3. Benchmark
    mean_time, std_time = _run_benchmark(results_dir)

    # 4. Speedup vs baseline (stored in results_dir from previous run)
    baseline_path = os.path.join(results_dir, "..", "baseline_time.json")
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline = json.load(f)
        baseline_time = baseline.get("mean_time", 1.0)
        speedup = _compute_speedup(baseline_time, mean_time)
    else:
        speedup = 1.0  # first run — no comparison
        baseline_time = mean_time

    metrics = {
        "combined_score": speedup,
        "execution_time_mean": mean_time,
        "execution_time_std": std_time,
        "fingerprint": fingerprint,
        "baseline_time": baseline_time,
        "num_successful_runs": 1,
        "num_valid_runs": 1,
        "num_invalid_runs": 0,
        "all_validation_errors": [],
    }

    elapsed = time.perf_counter() - start_time
    metrics["execution_time_mean"] = elapsed if dry_run else mean_time

    save_results(results_dir, metrics, True)


if __name__ == "__main__":
    main()
