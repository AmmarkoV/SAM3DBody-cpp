#!/bin/bash
# update.sh — pull the AmmarServer dependency and this package.

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

if [ -d dependencies/AmmarServer/.git ]; then
  echo "Updating AmmarServer.."
  cd dependencies/AmmarServer
  git pull origin master
  cd "$DIR"
else
  echo "AmmarServer not found — run ./initialize.sh first."
fi

# Pull this package too (no-op if not a standalone git checkout).
git pull 2>/dev/null

exit 0
