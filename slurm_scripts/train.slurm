#!/bin/bash
#SBATCH --job-name=test-megLM-te
#SBATCH --nodes=2
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=8
#SBATCH --mem=480G
#SBATCH --partition=dev-g
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH --account=project_462000615
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

#set -eox pipefail
echo "Starting bash script"
module purge
module load PrgEnv-cray
module load LUMI/24.03 partition/G
#Save the current working directory for shorter path variables
wd=(`pwd`)

# LR from LLaMa 2 70B.
LEARNING_RATE=1.5e-4
MIN_LR=1.5e-5

#COMPILER
export CC=clang
export CXX=clang++

#DISTRIBUTED ARGS
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999
export WORLD_SIZE=$SLURM_NTASKS #This is valid only if using srun as launcher
export CUDA_DEVICE_MAX_CONNECTIONS=1 #This is needed for sequence paralellism

#OMP THREADING
export OMP_NUM_THREADS=1
#export HSA_ENABLE_SDMA=1

#DEBUGGING, INCREASE VERBOSITY IN LOGS
export MIOPEN_ENABLE_LOGGING=1
export PYTHONWARNINGS=ignore
export TORCH_SHOW_CPP_STACKTRACES=1 #Hip kernels i think
#export TORCH_DISTRIBUTED_DEBUG=DETAIL #More spesific logs and importantly does extra sync checks on collectives. I get a different crash with this set to DETAIL
export NCCL_DEBUG=INFO
export RCCL_KERNEL_COLL_TRACE_ENABLE=1 #Dont know what this does
#export AMD_LOG_LEVEL=1 #Very verbose. In our context seems to concern hipified cuda kernels
export NCCL_DEBUG_SUBSYS=ALL #Spesify which stuff is logged from NCCL_DEBUG=INFO. Choices are P2P, COLL(Collecetives) etc.
export NCCL_DEBUG_FILE=logs/nccl-debug-$SLURM_JOB_NAME.log #Move verose nccl logging to its own file
#export HIP_LAUNCH_BLOCKING=1 #MAKES NCCL COLLECTIVES SYNCHRONOUS INSTEAD OF ASYNC
#export NCCL_P2P_DISABLE=1
#export AMD_SERIALIZE_KERNEL=3

NVTE_ROCM_ARCH=gfx90a
PYTORCH_ROCM_ARCH=gfx90a

#CHECKPOINTS/TB/WANDB
CHECKPOINT_PATH=/scratch/project_462000353/villekom/europa-checkpoints/33B_checkpoints
TENSORBOARD_PATH="/scratch/project_462000353/villekom/logs/europa-production/tensorboard/meg-LM-33B.$SLURM_JOB_ID"

# sets TRAIN_DATA and VALIDATION_DATA
source $wd/data/europa_data_flash.sh
DATA_CACHE=/flash/project_462000353/temp_cache/
MERGES=/scratch/project_462000353/europa-tokenizer/merges.txt
VOCAB=/scratch/project_462000353/europa-tokenizer/vocab.json

#PARALLELISM ARGS
PP_SIZE=1
TP_SIZE=1
VPP_SIZE=1

#LLama 34B
#MODEL=EUROPA-34B
#NLAYERS=56
#NHIDDEN=7168
#NHEADS=56
#FFN_HIDDEN_SIZE=20480
#SEQ_LEN=4096
#NUM_KV_HEADS=8

#TODO Calculate params for Tensorboard
MODEL=LLAMA2_7B
NHIDDEN=2048
FFN_HIDDEN_SIZE=5504
NLAYERS=24
NHEADS=16
SEQ_LEN=2048
NUM_KV_HEADS=4

#TOKENS
TOTAL_TOKENS=3_000_000_000_000
TOTAL_TOKENS=${TOTAL_TOKENS//_}    # drop "_" for bash math
TRAIN_SAMPLES=$((TOTAL_TOKENS/SEQ_LEN))

#BATCH SIZE
#Scaling gbs for smaller node count
GLOBAL_BATCH_SIZE=512
GBS_IN_TOKENS=SEQ_LEN*GLOBAL_BATCH_SIZE # 4096*1024 for viking
TARGET_NODES=512
TEST_NODES=$SLURM_NNODES
SCALING_FACTOR=$((TARGET_NODES/TEST_NODES))
SCALED_GLOBAL_BATCH_SIZE=$((GLOBAL_BATCH_SIZE/SCALING_FACTOR))
MICRO_BATCH_SIZE=1
echo "SCALED GBS" $SCALED_GLOBAL_BATCH_SIZE

#LEARNING RATE
LR=1.5e-4
MIN_LR=1.5e-5
LR_DECAY_SAMPLES=$TRAIN_SAMPLES
LR_WARMUP_SAMPLES=$((2000*GLOBAL_BATCH_SIZE))
# Also from LLaMa 2.
NUM_QUERY_GROUPS=8

LOG_INTERVAL=1
SAVE_INTERVAL=5000
EVAL_INTERVAL=10000
EVAL_STEPS=100
INIT_METHOD_STD=0.00747017

# These are the same as LLaMa 2.
OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-5 \
    --use-distributed-optimizer \
    --lr $LR \
    --min-lr $MIN_LR \
    --lr-decay-style cosine \
    --lr-decay-samples $LR_DECAY_SAMPLES \
    --lr-warmup-samples $LR_WARMUP_SAMPLES \
    --clip-grad 1.0 \
    --weight-decay 1e-1 \
    "

#PYTORCH PROFILER ARGS
PROFILE_ARGS=" \
    --profile pt \
    --profile_steps 10,15 \
    --profile-ranks 0 \
    "

GPT_ARGS=" \
    --data-cache-path $DATA_CACHE \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-samples $TRAIN_SAMPLES \
    --tokenizer-type GPT2BPETokenizer \
    --vocab-file $VOCAB \
    --merge-file $MERGES \
    --bf16 \
    --disable-bias-linear \
    --init-method-std $INIT_METHOD_STD \
    --make-vocab-size-divisible-by 128 \
    --no-gradient-accumulation-fusion \
    --normalization RMSNorm \
    --seed 42 \
    --untie-embeddings-and-output-weights \
    --use-flash-attn \
    --swiglu \
    --attention-dropout 0 \
    --hidden-dropout 0 \
    --attention-softmax-in-fp32 \
    --accumulate-allreduce-grads-in-fp32 \
    --use-rotary-position-embeddings \
    --no-bias-dropout-fusion \
    --no-masked-softmax-fusion \
    --group-query-attention \
    --num-query-groups $NUM_QUERY_GROUPS \
    --recompute-activations \
    --distributed-timeout-minutes 30 \
    $OPTIMIZER_ARGS \
    "

OUTPUT_ARGS=" \
    --save $CHECKPOINT_PATH \
    --log-interval $LOG_INTERVAL \
    --save-interval $SAVE_INTERVAL \
    --eval-interval $EVAL_INTERVAL \
    --eval-iters $EVAL_STEPS \
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-queue-size 5 \
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
    --train-data-path $TRAIN_DATA \
    --valid-data-path $VALIDATION_DATA \
    --dataloader-type single \
    --num-workers 2 \
    "

echo $CMD
echo $PARALLEL_ARGS

#DEBUGGING
#export MIOPEN_ENABLE_LOGGING=1
export PYTHONWARNINGS=ignore
export TORCH_SHOW_CPP_STACKTRACES=1 #Hip kernels i think
#export TORCH_DISTRIBUTED_DEBUG=INFO #More spesific logs and importantly does extra sync checks on collectives. I get a different crash with this set to DETAIL
export NCCL_DEBUG=INFO
export RCCL_KERNEL_COLL_TRACE_ENABLE=1 #Dont know what this does
#export AMD_LOG_LEVEL=1 #Very verbose. In our context seems to concern hipified cuda kernels
export NCCL_DEBUG_SUBSYS=ALL #Spesify which stuff is logged from NCCL_DEBUG=INFO. Choices are P2P, COLL(Collecetives) etc.
export NCCL_DEBUG_FILE=logs/nccl-debug-$SLURM_NNODES.log #Move verose nccl logging to its own file
#export HIP_LAUNCH_BLOCKING=1 #MAKES NCCL COLLECTIVES SYNCHRONOUS INSTEAD OF ASYNC

#NETWORK
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_TIMEOUT=3600           # 1 hour in seconds
#export NCCL_BLOCKING_WAIT=1

#LOGGING
export PYTHONWARNINGS=ignore

#Fused attention
export NVTE_FUSED_ATTN_CK=1
export NVTE_FUSED_ATTN_AOTRITON=0

#TransformerEngine
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=2
export NVTE_FLASH_ATTN=1
c="fe"

# Bind mask for one thread per core
BIND_MASK_1="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

# Bind mask for two threads per core
BIND_MASK_2="0x${c}00000000000000${c}000000000000,0x${c}00000000000000${c}00000000000000,0x${c}00000000000000${c}0000,0x${c}00000000000000${c}000000,0x${c}00000000000000${c},0x${c}00000000000000${c}00,0x${c}00000000000000${c}00000000,0x${c}00000000000000${c}0000000000"

BIND_MASK="$BIND_MASK_1"

# Check if SLURM_JOB_NAME is set, meaning the script is invoked by sbatch
if [ -z "$SLURM_JOB_NAME" ]; then
    # If not set, create a log file dynamically
    SLURM_JOB_NAME="interactive"
fi

echo "START $SLURM_JOBID: $(date)"
echo "NNODES" $SLURM_NNODES
echo "CPUS PER TASK" $SLURM_CPUS_PER_TASK


CONTAINER=/scratch/project_462000353/containers/sif_images/rocm-6.2.1-python-3.12-pytorch-2.6-vllm-0.6-TE.sif
echo "CONTAINER" $CONTAINER
export SINGULARITY_BIND=/pfs,/scratch,/projappl,/project,/flash,/appl,/usr/lib64/libjansson.so.4,/usr/lib64/libcxi.so.1,/opt/cray,/var/spool/slurmd
export PWD=(`pwd -P`)
srun --label --cpu-bind=mask_cpu:$BIND_MASK \
    singularity exec \
    -B $PWD \
    $CONTAINER \
    ./slurm_scripts/launch.sh \
    $CMD

echo "END $SLURM_JOBID: $(date)"
