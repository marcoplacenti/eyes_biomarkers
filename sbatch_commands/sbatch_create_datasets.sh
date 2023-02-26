#! /bin/bash

# A name for the job
#SBATCH --job-name=Generate-Dataset
 
# The queue you want to run on. This will determine instance type
#SBATCH --partition=gpu

#Request GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --constraint=p32xlarge
 
# send email at start and end of job
#SBATCH --mail-type=END

#SBATCH --time=2-0
 
# to this email. must be NN email
#SBATCH --mail-user=mopc@novonordisk.com

# writes errors to stderr file and output to stdout
# https://it.stonybrook.edu/help/kb/handling-job-output
#SBATCH -e prep_dataset_errors_log.txt
#SBATCH -o prep_dataset_output_log.txt
#SBATCH --open-mode=append
 
# Load the virtual environment
source env/bin/activate

# run datasets creation
python src/split_data.py