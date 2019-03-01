#!/bin/bash
SBATCH -A project00720
SBATCH -J voxel_mapper
SBATCH --mail-user=dam@ias.tu-darmstadt.de
SBATCH --mail-type=NONE
SBATCH -e /work/scratch/qd34dado/generalTsallisv2/generalTsallisv2_experiment.err.%A_%a.txt
SBATCH -o /work/scratch/qd34dado/generalTsallisv2/generalTsallisv2_experiment.out.%A_%a.txt
SBATCH --mem-per-cpu=1500
SBATCH -t 24:00:00
SBATCH -n 1
SBATCH -c 32
#SBATCH -C avx2

#load the required modules
module load python/2.7.12 cuda/9.0 intel 

#active your virtual environment
source ~/.venv2/bin/activate

cd ~/Workspace/SparsePC

echo "Not joint ${SLURM_JOB_ID} ${1} ${2} ${3}" >> /home/qd34dado/experiments.txt

python python2 trainer.py --logtostderr --batch_size=400 --env=${1} --validation_frequency=25 --tau=${3} --rollout=10 --critic_weight=1.0 --gamma=0.9 --clip_norm=10 --replay_buffer_freq=1 --objective=generaltsallisv2 --learning_rate=0.01 --tsallis=True --q=2.0 --k=1.0 --num_steps={2} --file_to_save='results/${1}_20_q_2_tau_${3}_learning_rate_0.01.txt'
