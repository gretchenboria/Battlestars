#!/bin/bash

# Load credentials from .env
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo ".env file not found"
    exit 1
fi

echo "Starting massive parallel remote execution for all problems (Benchmark + Profile)..."

# Pass expect script via -c to ssh and run command
/usr/bin/expect -c '
set timeout -1
spawn ssh -o StrictHostKeyChecking=accept-new "$env(username)@$env(host)" "cd ~/Battlestars && git pull origin hackathon && cd helion && source ~/helion_env/bin/activate && export HELION_AUTOTUNE_PRECOMPILE=spawn && for p in fp8_quant_py causal_conv1d_py gated_deltanet_chunk_fwd_h_py gated_deltanet_chunk_fwd_o_py gated_deltanet_recompute_w_u_py; do echo \"=============================\"; echo \"Running \$p\"; echo \"=============================\"; python eval.py benchmark \$p/; python eval.py profile \$p/; done"
expect "*assword:*"
send "$env(password)\r"
expect eof
'

echo "All experiments finished! Pulling all generated logs..."
./pull_logs.sh
