#!/bin/bash -eu

# MIOPEN needs some initialisation for the cache as the default location
# does not work on LUMI as Lustre does not provide the necessary features.
export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-$SLURM_NODEID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH

# Start conda environment inside the container
# $WITH_CONDA
#Venv outside the container
# source venv/bin/activate
# Set interfaces to be used by RCCL
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB


# unset CXI_FORK_SAFE
# unset CXI_FORK_SAFE_HP
# unset FI_CXI_DISABLE_CQ_HUGETLB

# export LD_LIBRARY_PATH=/pfs/lustrep3/scratch/project_462000394/containers/for-turkunlp-team/deps-2025-02-24/shs-libcxi-install/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/pfs/lustrep3/scratch/project_462000394/containers/for-turkunlp-team/deps-2025-02-24/libfabric-install-master/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/pfs/lustrep3/scratch/project_462000394/containers/for-turkunlp-team/deps-2025-02-24/aws-ofi-rccl2-install:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/pfs/lustrep3/scratch/project_462000394/containers/for-turkunlp-team/deps-2025-02-24/rccl-2025-02-26-85eb1f1/lib:$LD_LIBRARY_PATH

#export LD_LIBRARY_PATH=/pfs/lustrep2/scratch/project_462000125/samantao/tickets/rccl-repros/from-ville/rccl-2025-02-26-85eb1f1-lib:$LD_LIBRARY_PATH

if [[ $SLURM_PROCID -eq 0 ]]; then
    echo "LINK LIBRARY" $LD_LIBRARY_PATH
fi


export NCCL_DEBUG_FILE=/tmp/sfantao-nccl-logs-$SLURM_JOB_NAME-$SLURM_PROCID.log 

#Sanity check for cxi plugin and libfabric links
if [ "$SLURM_PROCID" -eq 0 ]; then
    echo "LDD librccl" $(ldd /opt/aws-ofi-rccl/librccl-net.so)
    echo "LD_LIBRARY" $LD_LIBRARY_PATH
    echo "LDD libfabric.so" $(ldd /opt/libfabric/lib/libfabric.so)
    /opt/libfabric/bin/fi_info -p cxi
fi

# if [ "$SLURM_PROCID" -eq 0 ]; then
#     echo "LDD librccl" $(ldd /pfs/lustrep3/scratch/project_462000394/containers/for-turkunlp-team/deps-2025-02-24/aws-ofi-rccl2-install/librccl-net.so)
#     echo "LD_LIBRARY" $LD_LIBRARY_PATH
#     echo "LDD libfabric.so" $(ldd /scratch/project_462000394/containers/for-turkunlp-team/deps-2025-02-24/libfabric-install-master/lib/libfabric.so)
#     /scratch/project_462000394/containers/for-turkunlp-team/deps-2025-02-24/libfabric-install-master/bin/fi_info -p cxi
# fi

#Distributed args
#These need to be set after srun launches, or use interpolation with bash -c and \
#https://github.com/stas00/ml-engineering/blob/master/orchestration/slurm/launchers/srun-launcher.slurm
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$((8 * SLURM_NNODES))

python3 -u "$@"