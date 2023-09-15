#!/bin/bash
#SBATCH -t 0-00:10000:00                    # time limit set to 1 minute
#SBATCH --mem=8G                         # reserve 1GB of memory
#SBATCH -J Tutorial                      # the job name
#SBATCH --mail-type=END,FAIL,TIME_LIMIT  # send notification emails
#SBATCH -n 1                             # use 5 tasks
#SBATCH --cpus-per-task=1                # use 1 thread per taks
#SBATCH -N 1                             # request slots on 1 node
#SBATCH --gres=gpu:RTX8000:1 --partition=dcv
#SBATCH --output=/home/dmittal/Desktop/Job_Outputs/test_%j_out.txt         # capture output
#SBATCH --error=/home/dmittal/Desktop/Job_Outputs/test_%j_err.txt          # and error streams


module load anaconda3/latest
. $ANACONDA_HOME/etc/profile.d/conda.sh

conda activate RUL_Prediction
python /home/dmittal/Desktop/RUL_Prediction_DFKI/FPC_Prediction/RUL_Using_LSTM_Conventional.py --processes $SLURM_NTASKS --threads $SLURM_CPUS_PER_TASK "$@"
