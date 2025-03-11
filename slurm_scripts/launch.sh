#!/bin/bash

# MIOPEN needs some initialisation for the cache as the default location
# does not work on LUMI as Lustre does not provide the necessary features.
export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-$SLURM_NODEID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH

# Report affinity
#echo "Rank $SLURM_PROCID --> $(taskset -p \$\$)"
# Start conda environment inside the container
$WITH_CONDA
# Set interfaces to be used by RCCL.
# This is needed as otherwise RCCL tries to use a network interface it has
# noa ccess to on LUMI.
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB
#export NCCL_DMABUF_ENABLE=1
export HSA_FORCE_FINE_GRAIN_PCIE=1


unset CXI_FORK_SAFE
unset CXI_FORK_SAFE_HP
unset FI_CXI_DISABLE_CQ_HUGETLB
export LD_LIBRARY_PATH=/pfs/lustrep3/scratch/project_462000394/containers/for-turkunlp-team/deps-2025-02-24/shs-libcxi-install/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/pfs/lustrep3/scratch/project_462000394/containers/for-turkunlp-team/deps-2025-02-24/libfabric-install-master/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/pfs/lustrep3/scratch/project_462000394/containers/for-turkunlp-team/deps-2025-02-24/aws-ofi-rccl2-install:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/pfs/lustrep3/scratch/project_462000394/containers/for-turkunlp-team/deps-2025-02-24/rccl-2025-02-26-85eb1f1/lib:$LD_LIBRARY_PATH

if [[ $SLURM_PROCID -eq 0 ]]; then
    echo "LINK LIBRARY" $LD_LIBRARY_PATH
fi


# The usual PyTorch initialisations (also needed on NVIDIA)
# Note that since we fix the port ID it is not possible to run, e.g., two
# instances via this script using half a node each.
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$((8 * SLURM_NNODES))
#echo "WORLD_SIZE IS:" $WORLD_SIZE
#echo "LOCAL_ID:" $SLURM_LOCALID

python3 -u "$@"
#torchrun \
#    --nproc-per-node=8 \
#    --nnodes $SLURM_NNODES \
#    --node_rank $SLURM_PROCID \
#    --rdzv_backend c10d \
#    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
#    "$@"