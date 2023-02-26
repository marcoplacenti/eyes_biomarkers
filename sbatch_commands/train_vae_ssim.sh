#! /bin/bash

# A name for the job
#SBATCH --job-name=train_VAE_ssim
 
# The queue you want to run on. This will determine instance type
#SBATCH --partition=gpu

#Request GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --constraint=p32xlarge
 
# send email at start and end of job
#SBATCH --mail-type=END
 
# to this email. must be NN email
#SBATCH --mail-user=mopc@novonordisk.com

#SBATCH --time=23:00:00

# writes errors to stderr file and output to stdout
# https://it.stonybrook.edu/help/kb/handling-job-output
#SBATCH -e sbatch_logs/VAE_ssim_errors_log.txt
#SBATCH -o sbatch_logs/VAE_ssim_output_log.txt
#SBATCH --open-mode=append
 
# Load the virtual environment
source env/bin/activate

# run datasets creation
python src/trainer.py --config configs/VAE_SSIM.yaml