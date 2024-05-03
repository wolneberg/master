#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=60G            
#SBATCH --job-name="SLR"
#SBATCH --output=MSASL/output/%x.out
#SBATCH --gres=gpu:1
# --constraint=gpu32g

#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=ingrimwo@stud.ntnu.no

#mem: total memory per node
#mem-per-cpu: memory per cpu-core

echo "We are running from this directory: $SLURM_SUBMIT_DIR"
echo "The name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

echo "--- nvidia-smi ---"
nvidia-smi

module purge
module load Python/3.10.4-GCCcore-11.3.0
module load cuDNN/8.8.0.121-CUDA-12.0.0

pip install -q -r requirements.txt
pip install -q tf-models-official==2.14
pip install -q --upgrade tensorflow==2.14
pip install -q --upgrade tensorflow.text==2.14
pip install -q git+https://github.com/tensorflow/docs

python3 train_msasl.py
