#!/bin/bash

# MIOPEN needs some initialisation for the cache as the default location
# does not work on LUMI as Lustre does not provide the necessary features.
export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-$SLURM_NODEID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH

# Report affinity
#echo "Rank $SLURM_PROCID --> $(taskset -p \$\$)"
# Start conda environment inside the container
$WITH_CONDA
#source /scratch/project_462000615/villekom/megatron_venv/bin/activate
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
export LD_LIBRARY_PATH=/pfs/lustrep3/scratch/project_462000394/containers/for-turkunlp-team/deps-2025-02-19/aws-ofi-rccl-install:$LD_LIBRARY_PATH
export PATH=/pfs/lustrep3/scratch/project_462000394/containers/for-turkunlp-team/deps-2025-02-19/libfabric-install-1.22.0/bin/fi_info:$PATH

if [ "$SLURM_PROCID" -eq 0 ]; then
    echo "LDD librccl" $(ldd /pfs/lustrep3/scratch/project_462000394/containers/for-turkunlp-team/deps-2025-02-19/aws-ofi-rccl-install/librccl-net.so)
    echo "LD_LIBRARY" $LD_LIBRARY_PATH
    echo "LDD libfabric.so" $(ldd /scratch/project_462000394/containers/for-turkunlp-team/deps-2025-02-19/libfabric-install-1.22.0/lib/libfabric.so)
    #fi_info -p cxi
fi

export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

python3 -u "$@"