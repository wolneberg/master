#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=ie-idi
#SBATCH --time=1-10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=40G            
#SBATCH --job-name="movinet6"
#SBATCH --output=movinet6.out

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
module load Python/3.11.5-GCCcore-13.2.0

python3 -m venv .venv
source .venv/bin/activate

pip3 install -q -r requirements.txt
pip3 install -q tf-models-official

echo "wlasl-100: RMS optimizer initial lr=0.1, cp_callback, verbose=2, epochs=20, med shuffle av train dataset, freeze:false"

python3 Models/MoViNet/movinet.py

echo "Finished running movient.py"

deactivate
