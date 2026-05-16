# JAX Kernel Evolution — ShinkaEvolve Integration

## Goal
Evolve C++ kernels in the JAX codebase using the ShinkaEvolve program evolution framework. The LLM proposes SEARCH/REPLACE patches within `EVOLVE-BLOCK-START`..`EVOLVE-BLOCK-END` markers in kernel `.cc` files, and each proposal is evaluated by Docker-based compilation + benchmarking.

## Architecture

```
ShinkaEvolveRunner (async_runner.py)
├── Proposal coordinator ──→ LLM (bandit-routed) ──→ SEARCH/REPLACE diff
├── Evaluator subprocess ──→ evaluate_kernel.py ──→ Docker: compile + benchmark
├── Database (SQLite) ────→ combined_score = speedup
├── Island manager ───────→ 1 island per .cc file, ring migration
└── Meta summarizer ──────→ every 10 gen
```

## Config Overrides

| Parameter | Value |
|---|---|
| `task.program_language` | `cpp` |
| `task.task_sys_msg` | Custom C++ kernel optimization prompt |
| `task.init_program_path` | Annotated `.cc` file path (one per island) |
| `database.num_islands` | = number of kernel files (3–5) |
| `evolution.patch_type_probs` | diff: 0.8, full: 0.1, fix: 0.1 |
| `evolution.max_generations` | 100 |
| `cluster.job_type` | local |
| `cluster.eval_program_path` | evaluate_kernel.py |

## Files

### `evaluate_kernel.py`
The eval subprocess called by ShinkaEvolve for every proposal. Receives:
- `--program_path` : mutated `.cc` file
- `--results_dir` : output directory for `metrics.json` + `correct.json`

Pipeline:
1. Extract mutated kernel region (EVOLVE-BLOCK content)
2. Compute SHA256 fingerprint of mutated code
3. Compile via Docker (`nvcc` + JAX deps)
4. Benchmark against pre-compiled baseline
5. Write `metrics.json: combined_score = speedup` (or 0 if compile fails)
6. Write `correct.json: {fingerprint_match: bool}`

### `config/kernel_evolution/` — Hydra config overrides
- `kernel_evolution_medium.yaml` : island + evolution + task config
- `kernel_cluster_local.yaml` : local cluster type with Docker eval

### `evaluate/Dockerfile`
Staged build:
- Base: NVIDIA CUDA 12.4 + GCC toolchain
- JAX build deps (Bazelisk, Python, protobuf, numpy)
- Entrypoint: runs `python evaluate_kernel.py`
- Mounts: mutated `.cc` file, baseline binary, output dir

## Workflow

1. Pre-run: build baseline binary via Docker, store fingerprint
2. Each generation:
   a. Proposer LLM generates SEARCH/REPLACE diff within EVOLVE-BLOCK
   b. `apply_diff.py` applies patch to kernel file
   c. `evaluate_kernel.py` compiles + benchmarks via Docker
   d. Speedup score stored in database
3. Selection: programs with speedup > 0 are accepted
4. Migration: every 10 gen, elite programs migrate between islands
5. Meta: every 10 gen, summarizer reviews top programs

## Next Steps

- [ ] Build baseline binary via Docker
- [ ] Test evaluate_kernel.py with a manual patch
- [ ] Run 1-gen test to validate full pipeline
- [ ] Migrate to VM for longer runs
