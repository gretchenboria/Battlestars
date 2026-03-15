#!/bin/bash

# Load credentials from .env
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo ".env file not found"
    exit 1
fi

echo "Pulling .popcorn.yaml from remote server..."

# Pass expect script via -c to scp the file
/usr/bin/expect -c '
set timeout 30
spawn scp -o StrictHostKeyChecking=accept-new "$env(username)@$env(host):~/.popcorn.yaml" ~/.popcorn.yaml
expect "*assword:*"
send "$env(password)\r"
expect eof
'

echo "Done! You can now use popcorn submit locally."
