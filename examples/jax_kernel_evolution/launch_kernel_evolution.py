#!/usr/bin/env python3
r"""Launch JAX kernel evolution via ShinkaEvolve.

Usage:
    # Dry-run: syntax check only (no GPU needed)
    python launch_kernel_evolution.py ^
        --jax-repo C:\path\to\jax ^
        --kernel-file jaxlib\cpu\lapack_kernels.cc ^
        --results-dir results\lapack_evo

    # Full run (on VM with Docker)
    python launch_kernel_evolution.py ^
        --jax-repo /path/to/jax ^
        --kernel-file jaxlib/gpu/hybrid_kernels.cc ^
        --results-dir results/hybrid_evo ^
        --docker-image evolvebench.azurecr.io/jax-builder:latest ^
        --no-dry-run
"""

import argparse
import sys
import os
import json
import shutil
import tempfile
from pathlib import Path
from shinka.core import ShinkaEvolveRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig


REPO_ROOT = Path(__file__).resolve().parents[2]  # experiments/ShinkaEvolve

EVALUATE_KERNEL_PATH = REPO_ROOT / "examples" / "jax_kernel_evolution" / "evaluate_kernel.py"
SUPPORTED_EXT = {".cc", ".cpp", ".cxx", ".cu.cc", ".cu"}


def _infer_language(path: Path) -> str:
    name = path.name.lower()
    if ".cu." in name or name.endswith(".cu"):
        return "cuda"
    return "cpp"


def main():
    parser = argparse.ArgumentParser(description="Launch JAX kernel evolution")
    parser.add_argument("--jax-repo", type=str, required=True,
                        help="Path to the JAX repository")
    parser.add_argument("--kernel-file", type=str, required=True,
                        help="Relative path to kernel file within JAX repo, "
                             "e.g. jaxlib/cpu/lapack_kernels.cc")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Output directory for run artifacts")
    parser.add_argument("--num-generations", type=int, default=20,
                        help="Number of generations (default: 20)")
    parser.add_argument("--docker-image", type=str,
                        default="evolvebench.azurecr.io/jax-builder:latest",
                        help="Docker image for compilation (full mode)")
    parser.add_argument("--model", type=str, default="openrouter/qwen/qwen3-coder",
                        help="LLM model for proposals")
    parser.add_argument("--num-islands", type=int, default=1,
                        help="Number of islands (default: 1, increase per kernel file)")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Dry-run: syntax check only")
    parser.add_argument("--no-dry-run", action="store_true",
                        help="Full: Docker compile + benchmark")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--no-verbose", action="store_false", dest="verbose")
    args = parser.parse_args()

    dry_run = args.dry_run and not args.no_dry_run
    jax_repo = os.path.abspath(args.jax_repo)
    init_path = os.path.join(jax_repo, args.kernel_file)
    init_path_obj = Path(init_path)
    results_dir = os.path.abspath(args.results_dir)

    if not os.path.exists(init_path):
        print(f"Error: kernel file not found: {init_path}")
        sys.exit(1)
    if not EVALUATE_KERNEL_PATH.exists():
        print(f"Error: evaluate_kernel.py not found at {EVALUATE_KERNEL_PATH}")
        sys.exit(1)

    language = _infer_language(init_path_obj)
    print(f"Language: {language}")
    print(f"Kernel:   {init_path}")

    # Build configs
    evo_config = EvolutionConfig(
        num_generations=args.num_generations,
        patch_types=["diff", "fix"],
        patch_type_probs=[0.9, 0.1],
        llm_models=[args.model],
        llm_kwargs={"temperatures": [0.0, 0.5], "max_tokens": 16384},
        meta_rec_interval=10,
        language=language,
        init_program_path=str(init_path),
        results_dir=str(results_dir),
        job_type="local",
        embedding_model=None,
        code_embed_sim_threshold=0.99,
        task_sys_msg=(
            "You are a high-performance computing expert specializing in "
            "JAX C++ kernel optimization. Only modify code inside "
            "EVOLVE-BLOCK-START..EVOLVE-BLOCK-END markers. "
            "Keep function signatures unchanged. "
            "Goal: maximize runtime speedup."
        ),
    )

    db_config = DatabaseConfig(
        db_path=os.path.join(results_dir, "evolution_db.sqlite"),
        num_islands=args.num_islands,
        archive_size=40,
        migration_interval=10,
        migration_rate=0.0,
        island_elitism=True,
    )

    job_config = LocalJobConfig(
        eval_program_path=str(EVALUATE_KERNEL_PATH),
        extra_cmd_args={
            "jax_repo": jax_repo,
            "docker_image": args.docker_image,
            "dry_run": str(dry_run).lower(),
        },
    )

    # Read init program and evaluate scripts as strings
    init_program_str = Path(init_path).read_text(encoding="utf-8")
    evaluate_str = EVALUATE_KERNEL_PATH.read_text(encoding="utf-8")

    print(f"\nLaunching ShinkaEvolve for JAX kernel evolution:")
    print(f"  Kernel:     {args.kernel_file}")
    print(f"  Language:   {language}")
    print(f"  Gens:       {args.num_generations}")
    print(f"  Mode:       {'dry-run' if dry_run else 'full'}")
    print(f"  Results:    {results_dir}")
    print()

    runner = ShinkaEvolveRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        init_program_str=init_program_str,
        evaluate_str=evaluate_str,
        banner_style="minimal",
        verbose=args.verbose,
    )
    runner.run()


if __name__ == "__main__":
    main()
