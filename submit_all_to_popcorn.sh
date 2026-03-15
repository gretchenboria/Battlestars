#!/bin/bash

# Load credentials from .env
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo ".env file not found"
    exit 1
fi

echo "Starting remote submission for all problems..."

export COMMAND="export PATH=\"\$HOME/.local/bin:\$PATH\" && cd ~/Battlestars && rm -rf helion/logs helion/eval.log && git fetch origin hackathon && git reset --hard origin/hackathon && cd helion && source ~/helion_env/bin/activate && export HELION_AUTOTUNE_PRECOMPILE=spawn && \
cd causal_conv1d_py && echo 'Submitting causal_conv1d_py' && popcorn submit --mode leaderboard submission.py --no-tui && cd .. && \
cd fp8_quant_py && echo 'Submitting fp8_quant_py' && popcorn submit --mode leaderboard submission.py --no-tui && cd .. && \
cd gated_deltanet_chunk_fwd_h_py && echo 'Submitting gated_deltanet_chunk_fwd_h_py' && popcorn submit --mode leaderboard submission.py --no-tui && cd .. && \
cd gated_deltanet_chunk_fwd_o_py && echo 'Submitting gated_deltanet_chunk_fwd_o_py' && popcorn submit --mode leaderboard submission.py --no-tui && cd .. && \
cd gated_deltanet_recompute_w_u_py && echo 'Submitting gated_deltanet_recompute_w_u_py' && popcorn submit --mode leaderboard submission.py --no-tui && cd .."

/usr/bin/expect -c '
set timeout -1
spawn ssh -o StrictHostKeyChecking=accept-new "$env(username)@$env(host)" "$env(COMMAND)"
expect "*assword:*"
send "$env(password)\r"
expect eof
'

echo "All submissions completed."
