#!/bin/bash
# Measured RCCL/libfabric tuning for LUMI (MI250X, gfx90a, Slingshot 11).
#
# Source this after whatever module/cache setup your job already does.
#
# THIS FILE IS A TEMPLATE UNTIL THE JOBS HAVE RUN. Every line below is either
# measured here, inherited from the bug study with its job ids, or deliberately left
# commented out. A candidate stays commented out until a Slurm job recorded in
# FINDINGS.md justifies it, **with its measured cost stated**. That rule is
# inherited verbatim from env_baseline.sh, and it is the entire difference between
# this file and HPE's tuning guide.
#
# Where the numbers come from: every claim quotes a Stage-4 job id, never a
# Stage-2 one. The top effect in a screen is biased upward by selection, so the
# screen nominates candidates and Stage 4 prices them on fresh allocations.
#
# Supersedes repro/rccl_startup_gfx90a/env_baseline.sh once Stage 5 passes.

# --- inherited, already measured -------------------------------------------------------
# Prevents the multi-node RCCL startup hang. 0 stalls in 18 attempts across 4 and 8
# nodes against a baseline that hung 21 of 23 (jobs 21838111, 21838977, 21838978),
# at no measurable bandwidth cost (88.1 vs 87.6 GB/s, jobs 21838863, 21791400).
#
# Root cause (job 21844179): libfabric defaults to the `memhooks` monitor here, which
# detects remapping by intercepting userspace allocator calls and does not reliably
# see ROCm memory operations, so a stale registration is never invalidated.
#     userfaultfd  0/5      kdreg2  0/5      disabled  0/5      memhooks  4/5
#
# This is a WORKAROUND for an open RCCL/libfabric bug (laifs-container-recipes#44).
# Keep generous startup timeouts until that is closed.
#   justified by: jobs 21838111, 21838977, 21838978, 21838863    cost: none measured
export FI_MR_CACHE_MONITOR=userfaultfd

# DO NOT use kdreg2 here. It is requested, it reaches the rank, and libfabric refuses
# it -- `kdreg2 monitor not available` (job 22119061) -- even though /dev/kdreg2 exists
# on the nodes. The monitor that results is neither kdreg2 nor userfaultfd, so setting
# it buys an unknown fallback in place of the one value measured to fix the hang.
#
# This withdraws env_baseline.sh:37-38, which called kdreg2 "an equally valid fix" on
# the strength of a 0/5 stall count. Whatever produced that 0/5, it was not kdreg2.
#   refuted by: job 22119061

# Not a comms setting, but the one measured win: ~167 s of every cold 8-node vLLM
# launch is one-time MIOpen compilation plus first touch of the weights on Lustre.
# /tmp is node-local, so this only pays off when Slurm reuses nodes.
#   justified by: jobs 21822747, 21822748    cost: none
export MIOPEN_CUSTOM_CACHE_DIR="/tmp/miopen-cache-${USER}"
export MIOPEN_USER_DB_PATH="/tmp/miopen-config-${USER}"

# --- candidates, pending measurement ---------------------------------------------------
# Screened as factors 1-12. Uncomment only with a Stage-4 job id and a measured cost.
# Expect most of these to come back confidently inert: that is a result worth
# shipping, not a failed study. A list of knobs measurably worthless on LUMI gfx90a,
# each with a job id, saves the next person the rediscovery.

# AMD documents a 5-10x small-message latency penalty on MI200 without this. No log
# line and no fd to inspect, so the B0/B1 latency A/B is the only evidence there is
# -- and the container is pinned at ROCm 7.0 while AMD documents the knob for 7.13+.
#   justified by: <job id>    cost: <measured>
# export HSA_NO_SCRATCH_RECLAIM=1

# MSCCL++ cannot be enabled on this container at all. RCCL says so itself:
#   NCCL WARN MSCCL++: Cannot enable MSCCL++; environment is not MSCCL compatible
# So the knob is a no-op on gfx90a here, not an untested opportunity.
#   refuted by: job 22119061
# export RCCL_MSCCLPP_ENABLE=1

# Channel count. Note the direction: the bug study only ever *capped* channels, and
# capping costs 20% of bulk bandwidth at 8 and 56% at 4 (jobs 21791400, 21838863).
# Raising the floor is the untested direction.
#   justified by: <job id>    cost: <measured>
# export NCCL_MIN_NCHANNELS=32
# export NCCL_MAX_NCHANNELS=16
# export NCCL_NCHANNELS_PER_NET_PEER=2

# Default is 4 MiB. Interacts with channel count, which is why the pair gets its own
# 3x3 grid at Stage 3b rather than being read off the screen's main effects.
#   justified by: <job id>    cost: <measured>
# export NCCL_BUFFSIZE=8388608

# HPE recommends CROSS_NIC=1 "on large systems" with no LUMI measurement behind it.
#   justified by: <job id>    cost: <measured>
# export NCCL_CROSS_NIC=1
# export NCCL_NET_GDR_READ=1

# HPE's rendezvous set, moved as one factor. THRESHOLD=0 forces *every* message
# through the rendezvous path, which the bug study measured as inert for the hang and
# nobody has ever measured for latency. The hypothesis is that it costs in B0/B1.
# FI_CXI_RDZV_PROTO=alt_read additionally needs the driver property rdzv_get_en=0 and
# falls back silently without it, so it may be untestable from userspace here.
#   justified by: <job id>    cost: <measured>
# export FI_CXI_RDZV_PROTO=alt_read
# export FI_CXI_RDZV_EAGER_SIZE=0
# export FI_CXI_RDZV_THRESHOLD=0
# export FI_CXI_RDZV_GET_MIN=0
# export FI_CXI_DEFAULT_TX_SIZE=2048

# NCCL_ALGO and NCCL_PROTO force a choice globally and override RCCL's per-size
# tuner, which is usually a loss. The only question Stage 3a asks is whether the
# tuner picks wrong at the decode size. Recommending them per-size would need an
# ncclTunerPlugin_v6, which is deliberately out of scope for this study.
# NCCL_PROTO=LL128 is not merely untested, it is unavailable: RCCL 2.26.6 has no
# LL128 path for bf16 all-reduce and fails outright (job 22118353):
#   no algorithm/protocol available for function AllReduce with datatype ncclBfloat16
# Since every collective in this serving path is bf16, LL128 cannot ship as a blanket
# setting regardless of what it would do for throughput.
#   refuted by: job 22118353
#   justified by: <job id>    cost: <measured>
# export NCCL_ALGO=Ring

# --- measured, and NOT recommended -----------------------------------------------------
# NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3 was recommended three times during the bug
# investigation and refuted three times against independent single-variant
# allocations: world_first 15.75 s vs baseline 15.73 s (jobs 21818931, 21818930),
# vLLM startup 287/123 s vs 290/123 s cold/warm (jobs 21822748, 21822747). Every
# earlier apparent win was position in the job or cache warming.
#   refuted by: jobs 21818930, 21818931, 21822747, 21822748

# FI_CXI_DEFAULT_CQ_SIZE and FI_CXI_RX_MATCH_MODE changed nothing for the reporter
# and nothing for this repo's probe. Two independent nulls; dropped before the screen
# rather than spending a design column on a third.

# --- DO NOT SET -------------------------------------------------------------------------
# NCCL_NET_GDR_LEVEL=PHB hangs the first cross-node collective on every rank,
# deterministically, at 4 nodes (32/32) and 8 nodes (64/64) -- jobs 21790392, 21790393,
# 21794114. At 4 nodes the background stall rate is zero, so this is unambiguous.
# It comes from HPE's ccl_env.sh (see laifs-container-recipes#30); the official LUMI
# AI Guide does not set it. Do not carry it in from HPE guidance or third-party
# examples. Note ROCm >= 6.2 already defaults to this behaviour, so setting it
# explicitly buys nothing even where it does not hang.
# export NCCL_NET_GDR_LEVEL=PHB   # <-- deliberately left unset

# --- diagnostics (off by default) --------------------------------------------------------
# NCCL_DEBUG=INFO makes every rank write a log; at 32+ ranks that measurably changes
# the startup timings it is meant to explain. Turn it on to identify the selected
# interface and transport, never to measure.
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,TUNING
# export NCCL_DEBUG_FILE=/runtime/nccl_rank%d.log
