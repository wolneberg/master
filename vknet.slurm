#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=10G
#SBATCH --job-name="vknet"
#SBATCH --output=testing.out
#SBATCH --gres=gpu:1

echo "We are running from this directory: $SLURM_SUBMIT_DIR"
echo "The name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

#usikkerområde start
curl --proto '=https' --tlsv1.2 -sSf -y https://sh.rustup.rs | sh

module purge 

#module load Python/3.10.4-GCCcore-11.3.0
module load Anaconda3/2022.10
module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0

#nvcc --version
#echo $CUDA_HOME $LD_LIBRARY_PATH $PATH

pip install -q -r NLASLR/requirements.txt
#pip instal lintel
conda install -c conda-forge lintel
#usikkerområde slutt

config_file='NLASLR/configs/rgb_frame64.yaml'
torchrun --nproc_per_node 8 --master_port 29999 NLASLR/training.py --config ${config_file} 

echo "Done running training.py"