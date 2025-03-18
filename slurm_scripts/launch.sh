#!/bin/bash

# MIOPEN needs some initialisation for the cache as the default location
# does not work on LUMI as Lustre does not provide the necessary features.
export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-$SLURM_NODEID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH

# Start conda environment inside the container
$WITH_CONDA
source /scratch/project_462000615/villekom/sam_container_venv/bin/activate
# Set interfaces to be used by RCCL.
# This is needed as otherwise RCCL tries to use a network interface it has
# no access to on LUMI.
#Use high speed NIC instead of mnt, or something else slower
export NCCL_SOCKET_IFNAME=hsn
#Use GPU Direct RDMA when GPU and NIC are on the same NUMA node
#This might cause hangs and other performance issues
export NCCL_NET_GDR_LEVEL=PHB

export LD_LIBRARY_PATH=/pfs/lustrep3/scratch/project_462000394/containers/for-turkunlp-team/deps-2025-02-24/shs-libcxi-install/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/pfs/lustrep3/scratch/project_462000394/containers/for-turkunlp-team/deps-2025-02-24/libfabric-install-master/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/pfs/lustrep3/scratch/project_462000394/containers/for-turkunlp-team/deps-2025-02-24/aws-ofi-rccl2-install:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/pfs/lustrep3/scratch/project_462000394/containers/for-turkunlp-team/deps-2025-02-24/rccl-2025-02-26-85eb1f1/lib:$LD_LIBRARY_PATH

if [[ $SLURM_PROCID -eq 0 ]]; then
    echo "LINK LIBRARY" $LD_LIBRARY_PATH
fi

#Sanity check for cxi plugin and libfabric links
if [ "$SLURM_PROCID" -eq 0 ]; then
    echo "LDD librccl" $(ldd /pfs/lustrep3/scratch/project_462000394/containers/for-turkunlp-team/deps-2025-02-24/aws-ofi-rccl2-install/librccl-net.so)
    echo "LD_LIBRARY" $LD_LIBRARY_PATH
    echo "LDD libfabric.so" $(ldd /scratch/project_462000394/containers/for-turkunlp-team/deps-2025-02-24/libfabric-install-master/lib/libfabric.so)
fi

export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$((8 * SLURM_NNODES))

python3 -u "$@"