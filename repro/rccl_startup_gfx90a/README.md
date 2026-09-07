# Multi-node vLLM startup stalls on LUMI — RCCL/Slingshot repro

A LUMI user reports multi-node vLLM startup stalling for 10-60+ minutes on
`standard-g`, **not in one fixed phase**: different runs freeze at different stages,
and clearing one stage does not prevent a stall at a later one. Setting
`NCCL_MAX_NCHANNELS=8` brings the same jobs to health in 210-645 s. They ask whether
this is known LUMI behaviour or their misconfiguration, whether there is a better fix,
what capping channels costs in collective bandwidth, and what the recommended
`NCCL_*`/`FI_CXI_*` baseline is.

Two things make this repo a clean instrument. It sets **no** `NCCL_*`/`RCCL_*`/`FI_CXI_*`
variable anywhere (`git log --all -S'NCCL_'` is empty), and the
`lumi-aif-singularity-bindings` module sets only `SINGULARITY_BIND` and
`SLURM_MPI_TYPE=pmi2` — so our multi-node runs sit at exactly the autodetected defaults
the report describes. And the root `README.md` shows we have been paying this already:
its multi-node recipes need `STARTUP_TIMEOUT_S` of 2700, 3600, and **14400** seconds,
while `run_vllm_demo_multinode.sh:62` computes the elapsed time into `$SECONDS` and
throws it away one line later.

## Pinned environment

| | |
| --- | --- |
| Container | `/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260807_115122/lumi-multitorch-full-...sif` (pinned, never the `latest` symlink) |
| Account | `project_462000131` |
| RCCL | `librccl.so.1.0.70002`, net plugin `/usr/lib/x86_64-linux-gnu/librccl-net-ofi.so` |
| libfabric | 1.22.0, CXI provider |
| Task layout, rungs 1/2/5 | the LUMI AI Guide's: 8 tasks/node, 7 cpus/task, `--mem-per-gpu=60G`, `--cpu-bind=v,mask_cpu=<8 masks>` |
| Task layout, rungs 3/4/6 | the repo's own: 1 task/node, 56 cpus/task, vLLM spawning its own workers |
| Models | `openai/gpt-oss-120b` (61 GB) for iteration, `moonshotai/Kimi-K2-Instruct-0905` (959 GB) for confirmation |

Override with `CONTAINER=`, `MODEL=`, `MODE=`, `STALL_TIMEOUT_S=`,
`STARTUP_TIMEOUT_S=`, `VARIANTS_ONLY=`. Results land in `results/job_<jobid>/`
(gitignored).

## The two hypotheses this is built to separate

| | mechanism | where the cost lands |
| --- | --- | --- |
| **H-A** | `NCCL_SOCKET_IFNAME` is unset, so RCCL's out-of-band bootstrap may pick `nmn0`/`bond0`/`net*` instead of the four `hsn*` Slingshot NICs | the `init` phase |
| **H-B** | `NCCL_RUNTIME_CONNECT` defers per-peer connection setup, so every **new communicator** pays a fresh burst costing ~`channels x peers` | every `*_first` phase |

H-B explains "stalls at different phases" directly, since vLLM builds TP, PP, world and
all2all groups and hits each one's first collective at a different startup stage. But
H-A better explains the one result that most constrains the space: the reporter saw
**no** change from `FI_CXI_DEFAULT_CQ_SIZE` or `FI_CXI_RX_MATCH_MODE=software`, which is
exactly what you would expect if the bottleneck is the bootstrap socket rather than the
CXI data path. They are not exclusive; the ladder attributes rather than picks.

## Run it, cheapest first

**1. RCCL phases, no model, no vLLM — minutes.** The decisive rung. Does the stall
reproduce, and does pinning the interface alone remove it?

```bash
sbatch repro/rccl_startup_gfx90a/run_rccl_probe.sh
```

**2. Full variant table and node scaling.** Dose-response on the channel cap, the
`cpu_bind` A/B, and the smallest world size that reproduces.

```bash
MODE=sweep sbatch repro/rccl_startup_gfx90a/run_rccl_probe.sh
MODE=sweep sbatch --nodes=4 repro/rccl_startup_gfx90a/run_rccl_probe.sh
MODE=sweep sbatch --nodes=8 repro/rccl_startup_gfx90a/run_rccl_probe.sh
```

Then read `results/job_<jobid>/sweep.md`.

**3. Interface evidence.** A timing win is not proof that H-A was the mechanism; the
`NCCL_DEBUG=INFO` INIT trace has to name a different interface in the two variants.
This is a separate job because INFO logging from every rank changes the timings.

```bash
MODE=debug sbatch repro/rccl_startup_gfx90a/run_rccl_probe.sh
```

**4. Real vLLM startup, phase by phase.** Does the microbenchmark signature carry into
the actual service?

```bash
sbatch repro/rccl_startup_gfx90a/run_vllm_startup.sh
MODE=sweep sbatch --nodes=4 repro/rccl_startup_gfx90a/run_vllm_startup.sh
```

**5. Confirmation at realistic scale**, against the root README's 4-node recipe that
needed a four-hour startup timeout.

```bash
MODEL=moonshotai/Kimi-K2-Instruct-0905 PP_SIZE=4 STARTUP_TIMEOUT_S=14400 \
EXTRA_VLLM_ARGS="--trust-remote-code --quantization fp8 --kv-cache-dtype fp8 --max-model-len 16384 --max-num-seqs 32 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.95" \
sbatch --nodes=4 --time=06:00:00 repro/rccl_startup_gfx90a/run_vllm_startup.sh
```

**6. What capping channels costs.** The number the reporter actually needs.

```bash
sbatch repro/rccl_startup_gfx90a/run_bandwidth.sh
sbatch --nodes=4 repro/rccl_startup_gfx90a/run_bandwidth.sh
sbatch --nodes=8 repro/rccl_startup_gfx90a/run_bandwidth.sh
```

Then read `results/job_<jobid>/bandwidth.md`.

**7. Is the weight-streamer symptom even the same bug?** One node, so no inter-node
communication can be blamed.

```bash
sbatch repro/rccl_startup_gfx90a/run_weightload.sh
READ_ONLY=1 MODEL=moonshotai/Kimi-K2-Instruct-0905 \
  sbatch repro/rccl_startup_gfx90a/run_weightload.sh
```

## What each file does

| file | role |
| --- | --- |
| `run_rccl_probe.sh` | rungs 1-3: sbatch entry, holds the variant table (on the host, so `cpu_bind` can change `srun` flags) |
| `rccl_probe.py` | times `init`, first/second/fresh-communicator collectives; watchdog records `STALL` and exits rather than holding nodes |
| `in_container_probe.sh` | per-task driver: caches, rank env, CXI snapshots either side |
| `run_vllm_startup.sh` | rungs 4-5: real vLLM startup, records the `$SECONDS` the repo's launcher discards |
| `in_container_vllm.sh` | mirrors `launch_vllm_multinode_rank.sh`, plus per-variant log separation |
| `phase_timeline.py` | turns rank logs into a phase timeline, anchored on vLLM's own markers including `Using ['PYNCCL'] all-reduce` |
| `run_bandwidth.sh` / `bandwidth_sweep.py` | rung 6: all_reduce / all_gather / reduce_scatter bus bandwidth vs channel cap |
| `summarize_bandwidth.py` | writes `results/bandwidth.md`, headlining the loss in the 128 MiB - 1 GiB training band |
| `run_weightload.sh` | rung 7: single-node load-format sweep, isolating the second symptom |
| `lustre_read.sh` | raw checkpoint read throughput — the floor under any weight-load time |
| `cxi_counters.sh` | Cassini telemetry before/after each phase |
| `collect_env.sh` | environment capture; unlike the sibling case's, its filter includes `NCCL_`/`RCCL_`/`FI_` |
| `summarize_sweep.py` | writes `results/sweep.md`, healthy rows first |
| `env_baseline.sh` | the recommended baseline — a template until jobs justify each line |

## How to read the verdict

| verdict | meaning |
| --- | --- |
| `FAST` | within 3x the fastest variant at this world size |
| `SLOW` | completed, but more than 3x the fastest |
| `STALL` | hit `STALL_TIMEOUT_S`; the phase it stalled in is recorded |
| `ERROR` | the variant failed to run; message in its JSON |

Phases are aggregated by **slowest rank**, not mean: a collective is only as fast as its
slowest participant, and one rank bootstrapping over the wrong interface is enough to
hold up all 64. A stall in `init` points at H-A; a stall in a `*_first` phase points at
H-B.

Two controls must hold or the run means nothing. `net_socket` must never stall — it
takes RCCL off the CXI path entirely, so if it stalls the harness or the allocation is
at fault. And the baseline must be submitted **twice** at the stalling world size: a
stall that appears once is a race and has to be described as one.

## Findings

Record conclusions in `FINDINGS.md`. Nothing there is a measurement until a job has
produced it, and every claim carries the Slurm job id that did.
