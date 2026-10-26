#!/bin/bash

set -e

echo "REMOTE: $REMOTE"
echo "PATH_TO_DESTINATION: $PATH_TO_DESTINATION"
echo "PORT: ${PORT:-22}"
echo "SSHPASS_FILE: $SSHPASS_FILE"


for file in *.tar.gz; do
    filename=${file%.tar.gz}
    if [ -n "$SSHPASS_FILE" ]; then
        sshpass_cmd="sshpass -f $SSHPASS_FILE"
    else
        sshpass_cmd=""
    fi

    ${sshpass_cmd} ssh -p ${PORT:-22} $REMOTE "mkdir -p $PATH_TO_DESTINATION"
    ${sshpass_cmd} scp -P ${PORT:-22} $file $REMOTE:$PATH_TO_DESTINATION
    ${sshpass_cmd} ssh -p ${PORT:-22} $REMOTE "mkdir $PATH_TO_DESTINATION/$filename && tar -zxvf $PATH_TO_DESTINATION/$file -C $PATH_TO_DESTINATION/$filename --strip-components=1"
    ${sshpass_cmd} ssh -p ${PORT:-22} $REMOTE "cd $PATH_TO_DESTINATION/$filename/ && git init && git add . && git commit -m \"init\""
    echo "your code have been transfered and uppacked to: $PATH_TO_DESTINATION/$filename/"
done;