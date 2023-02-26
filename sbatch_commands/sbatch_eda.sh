#! /bin/bash

# A name for the job
#SBATCH --job-name=MOPC_EDA
 
# The queue you want to run on. This will determine instance type
#SBATCH --partition=gpu

#Request GPU
#SBATCH --mem=32G
#SBATCH --constraint=p32xlarge
 
# send email at start and end of job
#SBATCH --mail-type=END

#SBATCH --time=10:00:00
 
# to this email. must be NN email
#SBATCH --mail-user=mopc@novonordisk.com

# writes errors to stderr file and output to stdout
# https://it.stonybrook.edu/help/kb/handling-job-output
#SBATCH -e sbatch_logs/EDA.txt
#SBATCH -o sbatch_logs/EDA.txt
#SBATCH --open-mode=append
 
# Load the virtual environment
source env/bin/activate

# run datasets creation
#python src/data/data_analysis.py
python src/data/distrib_mean_and_std.py