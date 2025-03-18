#!/bin/bash

# MIOPEN needs some initialisation for the cache as the default location
# does not work on LUMI as Lustre does not provide the necessary features.
export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-$SLURM_NODEID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH

# Start conda environment inside the container
$WITH_CONDA
# Set interfaces to be used by RCCL.
# This is needed as otherwise RCCL tries to use a network interface it has
# noa ccess to on LUMI.
export NCCL_SOCKET_IFNAME=hsn

export LD_LIBRARY_PATH=/pfs/lustrep3/scratch/project_462000394/containers/for-turkunlp-team/deps-2025-02-24/shs-libcxi-install/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/pfs/lustrep3/scratch/project_462000394/containers/for-turkunlp-team/deps-2025-02-24/libfabric-install-master/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/pfs/lustrep3/scratch/project_462000394/containers/for-turkunlp-team/deps-2025-02-24/aws-ofi-rccl2-install:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/pfs/lustrep3/scratch/project_462000394/containers/for-turkunlp-team/deps-2025-02-24/rccl-2025-02-26-85eb1f1/lib:$LD_LIBRARY_PATH

if [[ $SLURM_PROCID -eq 0 ]]; then
    echo "LINK LIBRARY" $LD_LIBRARY_PATH
fi

#Sanity check
#Check links for cxi plugin and new libfabric
if [ "$SLURM_PROCID" -eq 0 ]; then
    echo "LDD librccl" $(ldd /pfs/lustrep3/scratch/project_462000394/containers/for-turkunlp-team/deps-2025-02-19/aws-ofi-rccl-install/librccl-net.so)
    echo "LD_LIBRARY" $LD_LIBRARY_PATH
    echo "LDD libfabric.so" $(ldd /scratch/project_462000394/containers/for-turkunlp-team/deps-2025-02-19/libfabric-install-1.22.0/lib/libfabric.so)
fi

export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$((8 * SLURM_NNODES))

python3 -u "$@"