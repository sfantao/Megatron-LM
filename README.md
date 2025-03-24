# Reproducible 
This is a branch of Rocm-Megatron based on upstream's commit [dea104b](https://github.com/ROCm/Megatron-LM/commit/dea104b977b34f30b3b4a6d003352601b8721c92). Follow these steps to set up the environment and run the example script:

1. Clone the repository and checkout the reproducible branch:
    ```sh
    git clone https://github.com/LumiOpen/Megatron-LM.git
    cd Megatron-LM/
    git checkout reproducible
    ```

2. Download and extract the data and tokenizer files:
    ```sh
    mkdir -p reproducible
    wget -P reproducible https://a3s.fi/swift/v1/AUTH_3426c6e056824ed69977bdc49064971f/storage-bucket-project-462000353/reproducible.tar.gz
    tar -xvf reproducible/reproducible.tar.gz
    ```

3. Create a directory for logs:
    ```sh
    mkdir -p logs
    ```

4. Use the specified container and install the necessary packages:
    ```sh
    #WIP container
    CONTAINER=/scratch/project_462000394/containers/for-turkunlp-team/lumi/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.6.0-dockerhash-0fb1415058b3.sif

    #Extremely ugly one liner
    #Go inside the container, create a venv, and install newer versions of TE and FA
    singularity exec -B $PWD,/scratch/project_462000394/containers/for-turkunlp-team/ $CONTAINER bash -c "\$WITH_CONDA; python3 -m venv venv --system-site-packages; source venv/bin/activate; pip install -U tensorboard; pip install /scratch/project_462000394/containers/for-turkunlp-team/flash_attn-2.7.3-cp312-cp312-linux_x86_64.whl /scratch/project_462000394/containers/for-turkunlp-team/transformer_engine-1.11.0+e7a7f6d-cp312-cp312-linux_x86_64.whl"
    ```
5. Modify the [train.slurm](slurm_scripts/train.slurm) and [launch.sh](slurm_scripts/launch.sh) script to use the correct paths.

    ```sh
    sbatch slurm_scripts/train.slurm
    ```