#!/bin/bash

tmux new -s "tg-bot-prod" -d " cd /home/jovyan/tg_bot/prod ; source venv/bin/activate ; python3.9 tracker_bot.py"
tmux new -s "s3-synchronize" -d " cd /home/jovyan/abeletsky/s3-synchronize ; AWS_ACCESS_KEY_ID='aws_access_key_id' AWS_SECRET_ACCESS_KEY='aws_secret_access_key' python s3-synchronize.py /home/jovyan/gigachat_checkpoints gigachat_checkpoints"