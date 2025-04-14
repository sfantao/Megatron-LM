#!/bin/bash
#SBATCH --job-name=megLM-34B-16N
#SBATCH --nodes=16
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=8
#SBATCH --mem=480G
#SBATCH --partition=standard-g
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
SBATCH --account=project_462000615                     #Insert your own project number here
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -eox pipefail
echo "Starting bash script"
module purge
#Save the current working directory for shorter path variables
wd=(`pwd`)

# LR from LLaMa 2 70B.
LEARNING_RATE=1.5e-4
MIN_LR=1.5e-5

#COMPILER
export CC=gcc-12
export CXX=g++-12

#DISTRIBUTED ARGS
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999
export CUDA_DEVICE_MAX_CONNECTIONS=1                    #This is needed for sequence paralellism

#OMP THREADING
export OMP_NUM_THREADS=1                                #Virtual threads to accompany the HW threads set by slurm

#HSA=Heterogeneous System Architecture
#AMD provides HSA through ROCR https://rocm.docs.amd.com/projects/ROCR-Runtime/en/docs-6.2.4/index.html#hsa-runtime-api-and-runtime-for-rocm
export HSA_ENABLE_SDMA=0                                #https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-gpu-aware-mpi-readme/
                                                        #https://rocm.docs.amd.com/en/docs-6.2.0/conceptual/gpu-memory.html#system-direct-memory-access
                                                        #HSA_ENABLE_SDMA=0 -- Uses blit kernels for comms through the gpu -- more bandwith between gcd's
                                                        #With HSA_ENABLE_SDMA=1 capped at 50GB/S uni-directional bandwidth
                                                        #Enables overlapping communication with computation
#export HSA_ENABLE_PEER_SDMA=0                          #Enable or disable SDMA engines for device-to-device copies
                                                        #https://rocm.docs.amd.com/projects/ROCR-Runtime/en/docs-6.2.4/api-reference/environment_variables.html
#export HSA_XNACK=1                                     #A fallback for pagefault https://rocm.docs.amd.com/en/docs-6.2.4/conceptual/gpu-memory.html#xnack


export PYTHONWARNINGS=ignore
#####
#DEBUGGING, INCREASE VERBOSITY IN LOGS

#TORCH
export TORCH_SHOW_CPP_STACKTRACES=1                     #C++ traceback for pytorch crashes
#export TORCH_DIST_INIT_BARRIER=1                       #Initialize torch distributed new group instance with a barrier to synchronize ranks

#LIBFABRIC
#export FI_LOG_LEVEL=warn                               #Very verbose Libfabric logging. May cause performance degredation when set to 'debug'
#export FI_LOG_PROV=cxi                                  #Determine which libfabric provider to get logs from
#export FI_CXI_ODP=1                                    #On demand paging. This should be an alternative to the CXI_FORK_SAFE for unsafe fork. See more with man intro_mpi
#export FI_HMEM_DISABLE_P2P=1                           
#export FI_HMEM=rocr                                     #Ensure that libfabric uses rocr's hmem instead of system
#export FI_HMEM_ROCR_USE_DMABUF=0
#export FI_MR_ROCR_CACHE_MONITOR_ENABLED=1               #Detecting ROCR device memory (FI_HMEM_ROCR) changes made between the device virtual addresses used by an application and the underlying device physical pages.
#export SLINGSHOT_DEVICES=cxi0,cxi1                     #
#export FI_CXI_RDZV_GET_MIN=16364                       #Min byte size of message for cxi to use rendezvouz instead of eager messaging
#export FI_CXI_RDZV_THRESHOLD=32728                     #Max byte size
#export FI_CXI_DEFAULT_TX_SIZE=4096                     #Default is 1024
#export FI_CXI_RX_SIZE=4096
#export FI_CXI_RX_MATCH_MODE=software                   #Address matching can be either hardware(cxi) or software

export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DEFAULT_CQ_SIZE=131072

#NCCL/RCCL
#export RCCL_KERNEL_COLL_TRACE_ENABLE=1                  #Print c++ code traceback for nccl collectives
#export NCCL_DEBUG=INFO                                  #Verbose logging from RCCL
#export NCCL_DEBUG_SUBSYS=INIT,COLL                            #Specify which stuff is logged from NCCL_DEBUG=INFO. Choices are P2P(Peer2Peer), COLL(Collecetives) etc.
#export NCCL_DEBUG_FILE=logs/nccl-$SLURM_JOB_NAME.log    #Move verbose nccl logging to its own file
export NCCL_NCHANNELS_PER_PEER=32                       #Increases the amount of channels available for p2p communication. Should be a slight performance boost
#export HIP_LAUNCH_BLOCKING=1                           #MAKES NCCL COLLECTIVES SYNCHRONOUS INSTEAD OF ASYNC
#export NCCL_PROTO=Simple,LL                            #Simple or LL instead of ll128
#export NCCL_CHECKS_DISABLE=1                           #Disable checks before collectives, which increases performance
#export NCCL_DMABUF_ENABLE=1                            #Enable DMA buffers from the nccl side

#HIP/AMD DEBUG
#export AMD_SERIALIZE_KERNEL=3                          #Verbose logs from HIP
#export AMD_LOG_LEVEL=1                                 #Very verbose logging. In our context seems to concern hipified cuda kernels

#TRITON
export TRITON_ALWAYS_COMPILE=1                          #Always compile ops again instead of caching. TE Triton ops are compiled ahead of time anyway though

#TransformerEngine debug
#export NVTE_DEBUG=1                                    #Shows things like what dot product attention implementation is being used
#export NVTE_DEBUG_LEVEL=2

#CHECKPOINTS/TB/WANDB
CHECKPOINT_PATH=ckpts/$SLURM_JOB_NAME-$SLURM_JOB_ID
TENSORBOARD_PATH=tb_logs/$SLURM_JOB_NAME-$SLURM_JOB_ID

# sets TRAIN_DATA and VALIDATION_DATA
TRAIN_DATA=reproducible/redpajama-text_document
MERGES=reproducible/merges.txt
VOCAB=reproducible/vocab.json

#PARALLELISM ARGS
PP_SIZE=4
TP_SIZE=4
VPP_SIZE=2

#LLama 34B
MODEL=LLAMA-34B
NLAYERS=56
NHIDDEN=7168
NHEADS=56
FFN_HIDDEN_SIZE=20480
SEQ_LEN=4096
NUM_KV_HEADS=8
NUM_QUERY_GROUPS=8

#TODO
#Fix this to be 7B
#MODEL=LLAMA2_7B
#NHIDDEN=4096
#FFN_HIDDEN_SIZE=14336
#NLAYERS=32
#NHEADS=32
#SEQ_LEN=2048
#NUM_KV_HEADS=8
#GROUP_SIZE=$(( ${NHEADS} / ${NUM_KV_HEADS} ))
#NUM_GROUPS=$(( ${NHEADS} / ${GROUP_SIZE} ))
#NUM_QUERY_GROUPS=$NUM_GROUPS

#BATCH SIZE
#GBS can be adjusted according to node count
#Max 1024
GLOBAL_BATCH_SIZE=1024
#MBS max 1-2 with the default setup                        
MICRO_BATCH_SIZE=2

#LEARNING RATE
LR=1.5e-4
MIN_LR=1.5e-5

LOG_INTERVAL=1
EVAL_STEPS=0
INIT_METHOD_STD=0.00747017

OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-5 \
    --use-distributed-optimizer \
    --lr $LR \
    --min-lr $MIN_LR \
    --lr-decay-style cosine \
    --clip-grad 1.0 \
    --weight-decay 1e-1 \
    "

#PYTORCH PROFILER ARGS
PROFILE_ARGS=" \
    --profile \
    --use-pytorch-profiler \
    --profile-step-start 6 \
    --profile-step-end 7 \
    --profile-ranks 0 54 \
    "

GPT_ARGS=" \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --max-position-embeddings $SEQ_LEN \
    --seq-length $SEQ_LEN \
    --train-iters 10 \
    --data-path $TRAIN_DATA \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --tokenizer-type GPT2BPETokenizer \
    --vocab-file $VOCAB \
    --merge-file $MERGES \
    --bf16 \
    --disable-bias-linear \
    --init-method-std $INIT_METHOD_STD \
    --normalization RMSNorm \
    --seed 42 \
    --untie-embeddings-and-output-weights \
    --swiglu \
    --attention-dropout 0 \
    --hidden-dropout 0 \
    --attention-softmax-in-fp32 \
    --accumulate-allreduce-grads-in-fp32 \
    --use-rotary-position-embeddings \
    --group-query-attention \
    --num-query-groups $NUM_QUERY_GROUPS \
    --distributed-timeout-minutes 10 \
    --no-gradient-accumulation-fusion \
    --no-bias-swiglu-fusion \
    --recompute-activations \
    $OPTIMIZER_ARGS \
    $PROFILE_ARGS \
    "

OUTPUT_ARGS=" \
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-queue-size 5 \
    --log-throughput \
    --log-progress \
    --log-params-norm \
    --log-interval 1 \
    "
PARALLEL_ARGS="\
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --sequence-parallel \
"

if (( VPP_SIZE > 1)); then
    PARALLEL_ARGS="$PARALLEL_ARGS \
    --num-layers-per-virtual-pipeline-stage $VPP_SIZE"
fi


CMD=" \
    pretrain_gpt.py \
    $GPT_ARGS \
    $PARALLEL_ARGS \
    $OUTPUT_ARGS \
    --dataloader-type single \
    --num-workers 5 \
    \
    --profile-ranks 0 63 \
    --profile-step-start 3 \
    --profile-step-end 4 \
    --use-pytorch-profiler \
    --profile \
    --train-iters 5 \
    "



echo $CMD
echo $PARALLEL_ARGS


c="fe"

# Bind mask for one hardware thread per core
# Set hardware threads on slurm with --threads-per-core. Default is 1.
BIND_MASK="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

echo "START $SLURM_JOBID: $(date)"
echo "NNODES" $SLURM_NNODES
echo "CPUS PER TASK" $SLURM_CPUS_PER_TASK


#SINGULARITY ENVIRONMENT
#CONTAINER=/scratch/project_462000394/containers/for-turkunlp-team/lumi/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.6.0-dockerhash-0fb1415058b3.sif

#export CONTAINER=/scratch/project_462000394/containers/for-turkunlp-team/lumi/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.6.0-dockerhash-0fb1415058b3.sif
#export SINGULARITY_BIND=/boot,/pfs,/scratch,/projappl,/project,/flash,/appl,/opt/cray,/var/spool/slurmd,/usr/lib64/libjansson.so.4,/pfs/lustrep3/scratch/project_462000394/containers/for-turkunlp-team/deps-2025-02-19

#export CONTAINER=/appl/local/containers/tested-containers/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.6.0-dockerhash-36e16fb5b67b.sif
export CONTAINER=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.6.0.sif
export SINGULARITY_BIND=/boot,/pfs,/scratch,/projappl,/project,/flash,/appl,/opt/cray,/var/spool/slurmd
Nodes=16

#Run an rccl allreduce as a smoke test for unavailable NIC
#The unavailable NIC might produce the following warning message
########
#[PG ID 0 PG GUID 0 Rank 72]  using GPU 0 to perform barrier as devices used by this process are currently unknown.
#This can potentially cause a hang if this rank to GPU mapping is incorrect. Specify device_ids in barrier() to force use of a particular device, or call init_process_group() with a device_id.
########
#This doesn't crash or cause hangs to my run ever
#This was recommended by Samuel

echo "Running smoketest"
chmod +x ./slurm_scripts/smoketest.sh
srun -N $Nodes \
    -n $((Nodes*8)) \
    --cpu-bind=mask_cpu:$BIND_MASK \
    --gpus=$((Nodes*8)) \
    singularity exec \
    -B $PWD \
    $CONTAINER \
    ./slurm_scripts/smoketest.sh

echo "Smoketest done"

# # exit 0
sleep 2

echo "CONTAINER" $CONTAINER
export PWD=(`pwd -P`)

chmod +x ./slurm_scripts/launch.sh
srun -N $Nodes \
    -n $((Nodes*8)) \
    --cpu-bind=mask_cpu:$BIND_MASK \
    --gpus=$((Nodes*8)) \
    singularity exec \
    -B $PWD \
    $CONTAINER \
    ./slurm_scripts/launch.sh \
    $CMD |& tee train_log.log

echo "END $SLURM_JOBID: $(date)"
