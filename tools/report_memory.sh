#!/bin/bash
#SBATCH --job-name=test-megLM-TE-samcontainer-1N
#SBATCH --nodes=16
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=8
#SBATCH --mem=480G
#SBATCH --partition=standard-g
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH --account=project_462000615
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err


wd=(`pwd`)

# LR from LLaMa 2 70B.
LEARNING_RATE=1.5e-4
MIN_LR=1.5e-5


#DISTRIBUTED ARGS
export CUDA_DEVICE_MAX_CONNECTIONS=1 #This is needed for sequence paralellism

#PARALLELISM ARGS
PP_SIZE=4
TP_SIZE=2
VPP_SIZE=2

#LLama 34B
MODEL=EUROPA-34B
NLAYERS=56
NHIDDEN=7168
NHEADS=56
FFN_HIDDEN_SIZE=20480
SEQ_LEN=4096
NUM_KV_HEADS=8
NUM_QUERY_GROUPS=8

#TOKENS
TOTAL_TOKENS=3_000_000_000_000
TOTAL_TOKENS=${TOTAL_TOKENS//_}    # drop "_" for bash math
TRAIN_SAMPLES=$((TOTAL_TOKENS/SEQ_LEN))

#BATCH SIZE
#Scaling gbs for smaller node count
GLOBAL_BATCH_SIZE=1024
GBS_IN_TOKENS=SEQ_LEN*GLOBAL_BATCH_SIZE # 4096*1024 for viking
TARGET_NODES=512
TEST_NODES=$SLURM_NNODES
SCALING_FACTOR=$((TARGET_NODES/TEST_NODES))
SCALED_GLOBAL_BATCH_SIZE=$((GLOBAL_BATCH_SIZE/SCALING_FACTOR))
MICRO_BATCH_SIZE=1
echo "SCALED GBS" $SCALED_GLOBAL_BATCH_SIZE

MERGES=/scratch/project_462000353/europa-tokenizer/merges.txt
VOCAB=/scratch/project_462000353/europa-tokenizer/vocab.json

#LEARNING RATE
LR=1.5e-4
MIN_LR=1.5e-5
LR_DECAY_SAMPLES=$TRAIN_SAMPLES
LR_WARMUP_SAMPLES=$((2000*GLOBAL_BATCH_SIZE))
# Also from LLaMa 2.

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

GPT_ARGS=" \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --max-position-embeddings $SEQ_LEN \
    --seq-length $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-samples $TRAIN_SAMPLES \
    --tokenizer-type GPT2BPETokenizer \
    --bf16 \
    --disable-bias-linear \
    --init-method-std $INIT_METHOD_STD \
    --make-vocab-size-divisible-by 128 \
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
    --distributed-timeout-minutes 30 \
    --no-gradient-accumulation-fusion \
    --no-bias-swiglu-fusion \
    --vocab-file $VOCAB \
    --merge-file $MERGES \
    --recompute activations \
    $OPTIMIZER_ARGS \
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


python3 -u report_theoretical_memory.py $GPT_ARGS $PARALLEL_ARGS $OPTIMIZER_ARGS