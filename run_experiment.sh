#!/bin/bash

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: ./run_experiment.sh <mode> <problem_dir>"
    echo "Example: ./run_experiment.sh profile causal_conv1d_py"
    exit 1
fi

MODE=$1
PROBLEM=$2

# Load credentials from .env
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo ".env file not found"
    exit 1
fi

echo "Starting remote experiment: $MODE on $PROBLEM"

# Pass expect script via -c to ssh and run command
/usr/bin/expect -c '
set timeout -1
spawn ssh -o StrictHostKeyChecking=accept-new "$env(username)@$env(host)" "cd ~/Battlestars && git pull origin hackathon && cd helion && export HELION_AUTOTUNE_PRECOMPILE=spawn && python eval.py '"$MODE"' '"$PROBLEM"'/"
expect "*assword:*"
send "$env(password)\r"
expect eof
'

echo "Experiment finished. Pulling logs..."
./pull_logs.sh
