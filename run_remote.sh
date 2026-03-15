#!/bin/bash

# Load credentials from .env
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo ".env file not found"
    exit 1
fi

COMMAND="$*"

# Pass expect script via -c
/usr/bin/expect -c '
set timeout 300
spawn ssh -o StrictHostKeyChecking=accept-new "$env(username)@$env(host)" '"$COMMAND"'
expect "*assword:*"
send "$env(password)\r"
expect eof
'
