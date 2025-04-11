#!/bin/bash

export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3

# unset CXI_FORK_SAFE
# unset CXI_FORK_SAFE_HP
# unset FI_CXI_DISABLE_CQ_HUGETLB


# export LD_LIBRARY_PATH=/pfs/lustrep3/scratch/project_462000394/containers/for-turkunlp-team/deps-2025-02-24/shs-libcxi-install/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/pfs/lustrep3/scratch/project_462000394/containers/for-turkunlp-team/deps-2025-02-24/libfabric-install-master/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/pfs/lustrep3/scratch/project_462000394/containers/for-turkunlp-team/deps-2025-02-24/aws-ofi-rccl2-install:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/pfs/lustrep3/scratch/project_462000394/containers/for-turkunlp-team/deps-2025-02-24/rccl-2025-02-26-85eb1f1/lib:$LD_LIBRARY_PATH

#export LD_LIBRARY_PATH=/pfs/lustrep2/scratch/project_462000125/samantao/tickets/rccl-repros/from-ville/rccl-2025-02-26-85eb1f1-lib:$LD_LIBRARY_PATH

# /opt/rccltests/all_reduce_perf -o sum -b 4294967296 -e 4294967296 -i 41943040 -n 1
/opt/rccltests/all_reduce_perf -z 1 -b 2M -e 4096M -f 2 -g 1 -t 1 -R 1 -n 80 -w 5 -d half
