#!/bin/bash
# Recommended NCCL/RCCL + libfabric baseline for multi-node RCCL on LUMI.
#
# Source this after whatever module/cache setup your job already does.
#
# The official LUMI AI Guide (Lumi-supercomputer/LUMI-AI-Guide) sets no
# NCCL_*/RCCL_*/FI_* variable in any lesson, so there is no guide baseline to extend --
# its multi-node lesson runs at stock defaults, which is exactly where the intermittent
# hang lives.
#
# Every line below is either measured here or deliberately left unset. A candidate stays
# commented out until a Slurm job in FINDINGS.md justifies it, with its cost stated.
# Shipping an unjustified NCCL variable is how cargo-cult tuning starts, and the point of
# the reporter's fourth question is that no measured baseline exists yet.

# --- THE FIX -------------------------------------------------------------------------
# Prevents the multi-node RCCL startup hang. 0 stalls in 18 attempts across 4 and 8
# nodes, against a baseline that hung 21 of 23 (jobs 21838111, 21838977, 21838978).
# Costs nothing: uncapped all_reduce bus bandwidth is 88.1 GB/s with it set against
# 87.6 GB/s without (jobs 21838863, 21791400), i.e. identical within noise -- unlike
# NCCL_MAX_NCHANNELS=8, which also stops the hang but costs 21% of training-band
# bandwidth.
#
# Identified by LUMI support; originates with Samuel Antao (AMD), "Extreme Scale AI",
# Move your AI to LUMI, June 2026. Also named in laifs-container-recipes#30.
#
# This is a WORKAROUND for an open RCCL/libfabric bug (laifs-container-recipes#44), not
# a root-cause fix. Keep generous startup timeouts until that is closed.
#   justified by: jobs 21838111, 21838977, 21838978, 21838863    cost: none measured
export FI_MR_CACHE_MONITOR=userfaultfd

# --- HPE also recommends these; they are inert for this failure ----------------------
# HPE's full RCCL list, relayed by LUMI support, adds ten more variables. Measured at
# 4 nodes with 8 communicators (job 21838977): the full set gives 0/5, and the set with
# FI_MR_CACHE_MONITOR REMOVED gives 4/5 -- identical to baseline. So the monitor does all
# the work and the remainder changes nothing here. Left unset rather than carried as
# unexamined ballast; they may still matter for other workloads.
#   HSA_FORCE_FINE_GRAIN_PCIE=1  FI_CXI_DISABLE_HOST_REGISTER=1
#   FI_CXI_DEFAULT_CQ_SIZE=131072  FI_CXI_RDZV_PROTO=alt_read
#   FI_CXI_RDZV_EAGER_SIZE=0  FI_CXI_RDZV_THRESHOLD=0  FI_CXI_RDZV_GET_MIN=0
#   FI_CXI_DEFAULT_TX_SIZE=2048  NCCL_CROSS_NIC=1  FI_CXI_RX_MATCH_MODE=hybrid
# Note FI_CXI_DISABLE_HOST_REGISTER=1 on its own leaves 3/5 stalls (job 21838111).

# --- measured, and NOT recommended ---------------------------------------------------
# NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3 was recommended three times during this
# investigation and refuted three times. Measured against independent single-variant
# allocations it changes nothing: world_first 15.75 s vs baseline 15.73 s (jobs 21818931,
# 21818930), vLLM startup 287/123 s vs 290/123 s cold/warm (jobs 21822748, 21822747), and
# it stalled 1 of 2 fresh allocations. Every earlier apparent win was position in the job
# or cache warming. It is left unset rather than cargo-culted.
#   refuted by: jobs 21818930, 21818931, 21822747, 21822748

# --- DO NOT SET -----------------------------------------------------------------------
# NCCL_NET_GDR_LEVEL=PHB hangs the first cross-node collective on every rank,
# deterministically, at 4 nodes (32/32) and 8 nodes (64/64) -- jobs 21790392, 21790393,
# 21794114. At 4 nodes the background stall rate is zero, so this is unambiguous.
# It comes from HPE's ccl_env.sh (see laifs-container-recipes#30, which has tracked this
# hang since April 2026); the official LUMI AI Guide does not set it. Do not carry it
# into a multi-node script from HPE guidance or third-party examples.
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

# --- not a comms setting, but the one measured win ------------------------------------
# ~167 s of every cold 8-node vLLM launch is one-time MIOpen kernel compilation plus
# first-touch of the weights on Lustre (290 s cold vs 123 s warm, both variants, jobs
# 21822747/21822748). A persistent per-user MIOpen cache avoids re-paying the
# compilation half of that. This repo's launchers use a per-job mktemp -d instead, which
# guarantees paying it every launch. /tmp is node-local, so this only helps when Slurm
# reuses nodes.
#   justified by: jobs 21822747, 21822748    cost: none
export MIOPEN_CUSTOM_CACHE_DIR="/tmp/miopen-cache-${USER}"
export MIOPEN_USER_DB_PATH="/tmp/miopen-config-${USER}"

# --- diagnostics (off by default) -----------------------------------------------------
# NCCL_DEBUG=INFO makes every rank write a log; at 64 ranks that measurably changes the
# startup timings it is meant to explain. Turn it on to identify the selected interface
# and transport, not to measure.
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH
# export NCCL_DEBUG_FILE=/runtime/nccl_rank%d.log
