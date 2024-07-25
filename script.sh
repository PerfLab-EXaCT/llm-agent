#!/bin/sh
# Specify the allocation account to charge for the job.
#SBATCH -A ${chess}

# Set a limit on the total run time of the job. hh:mm:ss
#SBATCH -t 00:10:00

# Request one node for the job.
#SBATCH -N 1

# Request one task (process) for the job.
#SBATCH -n 1

# Set the name of the job, which will appear in the job scheduler.
#SBATCH -J llm-agent-project

# Specify the file for standard output, using the job ID as part of the filename.
#SBATCH -o "./%j.stdout.txt"

# Specify the file for standard error, using the job ID as part of the filename.
#SBATCH -e "./%j.stderr.txt"

# Specify the partition (queue) to submit the job to.
#SBATCH -p a100_shared

# Request one GPU for the job.
#SBATCH --gres=gpu:1

# Specify the email address to send notifications to.
#SBATCH --mail-user=${brian.chen@pnnl.gov}

# Specify when to send email notifications. 'ALL' sends emails on job begin, end, and failure.
#SBATCH --mail-type=ALL