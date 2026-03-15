#!/bin/bash

# Load credentials from .env
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo ".env file not found"
    exit 1
fi

echo "Pulling logs directory from remote server..."

# Create local logs dir just in case
mkdir -p helion/logs

# Pass expect script via -c to scp the files
/usr/bin/expect -c '
set timeout 120
spawn scp -r -o StrictHostKeyChecking=accept-new "$env(username)@$env(host):~/Battlestars/helion/logs/*" helion/logs/
expect "*assword:*"
send "$env(password)\r"
expect eof
'

echo "Done! Logs are now in helion/logs/"
