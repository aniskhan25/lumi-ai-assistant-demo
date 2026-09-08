#!/bin/bash
# Recommended NCCL/RCCL + libfabric baseline for multi-node RCCL on LUMI.
#
# Source this AFTER the LUMI AI Guide's setup.sh, which it is meant to extend rather
# than replace:
#
#   source ../setup.sh
#   source repro/rccl_startup_gfx90a/env_baseline.sh
#
# TEMPLATE. Every line below is either (a) already established by the LUMI AI Guide, or
# (b) a candidate this investigation has not yet decided. A candidate stays commented
# out until a Slurm job in FINDINGS.md justifies it, with its bandwidth cost stated.
# Shipping an unjustified NCCL variable is how cargo-cult tuning starts, and the whole
# point of the reporter's fourth question is that no measured baseline exists yet.

# --- measured, and worth setting -----------------------------------------------------
# Cuts first-collective setup time from ~15.7 s to ~9.8 s (about 38%) consistently at
# 16, 32 and 64 ranks. It is not a channel cap, so it costs no collective bandwidth.
# It does NOT prevent the multi-node startup hang: job 21818931 stalled on its first
# attempt with this set. Set it for the setup-time win, not as a fix.
#   justified by: jobs 21790359, 21790392, 21790393, 21811437    cost: none measured
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3

# --- DO NOT SET -----------------------------------------------------------------------
# NCCL_NET_GDR_LEVEL=PHB hangs the first cross-node collective on every rank,
# deterministically, at 4 nodes (32/32) and 8 nodes (64/64) -- jobs 21790392, 21790393,
# 21794114. At 4 nodes the background stall rate is zero, so this is unambiguous.
# The LUMI AI Guide sets it in 5-experiment-tracking/run_*.sh, which are single-node
# jobs where it is harmless; do not carry that line into a multi-node script.
# export NCCL_NET_GDR_LEVEL=PHB   # <-- deliberately left unset

# --- candidates, pending measurement --------------------------------------------------
# Uncomment only with a job id and a measured cost recorded in FINDINGS.md.

# Establish all peer connections during init instead of lazily at each communicator's
# first collective. Decides hypothesis 2 (H-B). Expected trade: slower, more
# predictable init; no later per-phase bursts.
#   justified by: <job id>    cost: <measured>
# export NCCL_RUNTIME_CONNECT=0

# The reported workaround. Do NOT adopt this without the bandwidth number from
# results/bandwidth.md: it is the setting the reporter explicitly does not want to pay
# for on bandwidth-bound training jobs, and capping channels costs most at exactly the
# large message sizes those jobs use.
#   justified by: <job id>    cost: <measured, from bandwidth.md training band>
# export NCCL_MAX_NCHANNELS=8

# Narrower than a global channel cap: limits only per-net-peer channels, so intra-node
# collectives keep their full width.
#   justified by: <job id>    cost: <measured>
# export NCCL_NCHANNELS_PER_NET_PEER=1

# The reporter measured no difference from either of these, which is itself evidence:
# it argues the bottleneck is not CQ depth or the hardware match cache. Kept here only
# so the next person does not have to rediscover that they do not help.
# export FI_CXI_DEFAULT_CQ_SIZE=131072
# export FI_CXI_RX_MATCH_MODE=software

# --- diagnostics (off by default) -----------------------------------------------------
# NCCL_DEBUG=INFO makes every rank write a log; at 64 ranks that measurably changes the
# startup timings it is meant to explain. Turn it on to identify the selected interface
# and transport, not to measure.
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH
# export NCCL_DEBUG_FILE=/runtime/nccl_rank%d.log
