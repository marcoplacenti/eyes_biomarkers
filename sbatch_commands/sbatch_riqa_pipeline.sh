#! /bin/bash

# A name for the job
#SBATCH --job-name=MOPC_riqa_pipeline
 
# The queue you want to run on. This will determine instance type
#SBATCH --partition=gpu

#Request GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --constraint=p32xlarge
 
# send email at start and end of job
#SBATCH --mail-type=END

#SBATCH --time=10:00:00
 
# to this email. must be NN email
#SBATCH --mail-user=mopc@novonordisk.com

# writes errors to stderr file and output to stdout
# https://it.stonybrook.edu/help/kb/handling-job-output
#SBATCH -e errors_log_21016.txt
#SBATCH -o output_log_21016.txt
#SBATCH --open-mode=append
 
# Load the virtual environment
source env/bin/activate

# run RIQA inference
python src/riqa.py