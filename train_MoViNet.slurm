#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=03:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100G            
#SBATCH --job-name="SLR"
#SBATCH --output=Models/MoViNet/Output/%x.out
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint="gpu80g"

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
module list
module load Python/3.10.4-GCCcore-11.3.0

# python -m venv venv/Test9
source venv/Test9/bin/activate


module load cuDNN/8.8.0.121-CUDA-12.0.0

pip install -q -r requirements.txt
pip install -q tf-models-official==2.14
pip install -q --upgrade tensorflow==2.14
pip install -q git+https://github.com/tensorflow/docs
# pip install git+https://github.com/okankop/vidaug
module list

python train_movinet.py -b 4 -l 0.001 -sd 0.1 -uf 45 -d 0.1 -v 'a4' -s 1 -e 2 -r 172

echo "Finished running train_movinet.py"

deactivate
# conda deactivate