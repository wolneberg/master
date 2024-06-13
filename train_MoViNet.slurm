#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=60G            
#SBATCH --job-name="SLR"
#SBATCH --output=Models/MoViNet/Output/%x.out
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu32g



#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=helenaml@stud.ntnu.no

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
# # module load  Python/3.11.3-GCCcore-12.3.0
# # module load Clang/16.0.6-GCCcore-12.3.0 

# python -m venv venv/Test9
source venv/Test9/bin/activate
# # pip uninstall keras
# # pip install 

module load cuDNN/8.8.0.121-CUDA-12.0.0

pip install -q -r requirements.txt
echo "install tf-modles-official" 
pip install -q tf-models-official==2.14
# pip install --upgrade protobuf
pip show tf-models-official
module list
# pip show tensorflow
pip install -q --upgrade tensorflow==2.14
pip show tensorflow
# pip show tensorflow-text
pip install -q --upgrade tensorflow.text==2.14
pip show tensorflow-text
# pip uninstall tf-nightly
# pip install --upgrade protobuf
pip install -q git+https://github.com/tensorflow/docs
# pip install git+https://github.com/okankop/vidaug

echo "movinet med gpu"
echo "wlasl-100: rmsprop, epochs=30, not freezed backbone, med shuffle av train dataset, over"
# python -v
echo 'batch: 4, learning rate, 0.01, dropout: 0.3'
python train_movinet.py -b 4 -l 0.01 -d 0 -v 'a0' -s 1 -e 2

echo "Finished running train_movinet.py"

deactivate
# conda deactivate