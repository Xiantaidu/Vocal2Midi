#!/bin/zsh

cd "$(dirname "$0")"
export PATH="/opt/homebrew/bin:/usr/local/bin:/opt/local/bin:$PATH"
if [[ ! -x .venv/bin/python ]]; then
  echo "Missing .venv. Create it with: /opt/homebrew/bin/python3.12 -m venv .venv"
  echo "Then install dependencies with: .venv/bin/python -m pip install -r requirements.txt"
  read -k 1 "?Press any key to close..."
  exit 1
fi

if [[ -x /opt/homebrew/bin/ffmpeg ]]; then
  export V2M_FFMPEG=/opt/homebrew/bin/ffmpeg
elif [[ -x /usr/local/bin/ffmpeg ]]; then
  export V2M_FFMPEG=/usr/local/bin/ffmpeg
fi

exec .venv/bin/python app_fluent.py
