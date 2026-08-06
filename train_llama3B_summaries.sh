#!/bin/bash
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --partition=h200
#SBATCH --job-name="simple-rl"
export HF_HOME=/scratch-ssd/$USER/.cache/huggingface
export XDG_CACHE_HOME=/scratch-ssd/$USER/.cache
export UV_CACHE_DIR=/scratch-ssd/$USER/.cache/uv
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export WANDB_CACHE_DIR="/scratch-ssd/$USER/.cache/wandb"
export WANDB_DIR="/scratch-ssd/$USER"
export WANDB_API_KEY="ae5357c956169358a187cc70668d0b78265e6412"



VENV_DIR="/scratch-ssd/$USER/simpleRL-reason"
PROJECT_DIR="$HOME/projects/simpleRL-reason"
RESULTS_DIR="$HOME/projects/simpleRL-reason/results"

# create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ] || [ ! -f "$VENV_DIR/bin/python" ]; then
    echo " Virtualenv not found. Creating and activating a new one with uv"
    uv venv "$VENV_DIR" --clear --python 3.10
    source "$VENV_DIR/bin/activate"

    # install the project dependecies 
    cd "$PROJECT_DIR"
    uv pip install "setuptools<70.0.0" wheel packaging ninja
     
    uv pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
    uv pip install flash-attn --no-build-isolation
    uv pip install xformers==0.0.27.post2
    uv pip install vllm==0.5.4
    uv pip install "ray[default]==2.10.0"
    uv pip install git+https://github.com/ozeliger/pyairports.git pycountry
    uv pip install wandb
    uv pip install -e .
    uv pip install "click<8.1.8"
    
else
    source "$VENV_DIR/bin/activate"
    echo "Virtualenv at $VENV_DIR activated"

    # install the project dependecies 
    cd "$PROJECT_DIR"
    uv pip install "setuptools<70.0.0" wheel packaging ninja
     
    uv pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
    uv pip install flash-attn --no-build-isolation
    uv pip install xformers==0.0.27.post2
    uv pip install vllm==0.5.4
    uv pip install "ray[default]==2.10.0"
    uv pip install git+https://github.com/ozeliger/pyairports.git pycountry
    uv pip install wandb
    uv pip install -e .
    uv pip install "click<8.1.8"
fi

cd "$PROJECT_DIR"   

mkdir -p /scratch-ssd/$USER/logs
mkdir -p /scratch-ssd/$USER/checkpoints
mkdir -p /scratch-ssd/$USER/models

DATA_DIR="/scratch-ssd/$USER/simplelr_abel_level1to4"
mkdir -p "$DATA_DIR"

if [ ! -f "$DATA_DIR/train.parquet" ]; then
    wget -O "$DATA_DIR/train.parquet" https://huggingface.co/datasets/hkust-nlp/SimpleRL-Zoo-Data/resolve/main/simplelr_abel_level1to4/train.parquet
fi

if [ ! -f "$DATA_DIR/test.parquet" ]; then
    wget -O "$DATA_DIR/test.parquet" https://huggingface.co/datasets/hkust-nlp/SimpleRL-Zoo-Data/resolve/main/simplelr_abel_level1to4/test.parquet
fi

# Catch termination signals (like scancel) to gracefully stop Ray and free GPUs
trap "echo 'Caught termination signal, stopping Ray...'; ray stop; exit 0" EXIT SIGTERM SIGINT

ray start --head --node-ip-address 127.0.0.1 --num-gpus 8 --temp-dir=/scratch-ssd/$USER/ray_tmp
export HEAD_IP=127.0.0.1
export HEAD_PORT=6379

# use the right GPU connection.
# Overridable: `NCCL_P2P_DISABLE=0 ./train_qwen3B_summaries.sh` keeps NVLink P2P on,
# which is much faster for FSDP all-gathers on an NVSwitch node. Disabling P2P forces
# all collectives through host shared memory.
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export RAY_OVERRIDE_JOB_RUNTIME_ENV=1

./train_grpo_math_tune_ray_llama.sh --model_name Llama-3.2-3B-medium-14B --dataset_name simplelr_abel_level1to4 --max_response_length 2048 --train_batch_size 1024 --rollout_n 8 --kl_loss_coef 0.0001 --entropy_coeffient 0.001 --rollout_gpu_memory_util 0.75 --rollout_tp 1 --save_freq 10 --micro_rollout_batch_size 128 --ppo_micro_batch_size 8
