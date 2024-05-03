#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=05:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=10G  
#SBATCH --job-name="SLR"
#SBATCH --output=MSASL/%x.out

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


module purge
module load Python/3.10.4-GCCcore-11.3.0

pip install -q moviepy
pip install -q pytube

python MSASL/trim.py