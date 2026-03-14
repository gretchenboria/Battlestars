#!/bin/bash

# Load credentials from .env
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo ".env file not found"
    exit 1
fi

# Pass expect script via -c to leave stdin attached to your terminal
/usr/bin/expect -c '
set timeout 10
spawn ssh -o StrictHostKeyChecking=accept-new "$env(username)@$env(host)"
expect "*assword:*"
send "$env(password)\r"
interact
'
