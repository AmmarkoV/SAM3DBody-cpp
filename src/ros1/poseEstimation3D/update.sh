#!/bin/bash
# update.sh — refresh the AmmClient dependency and re-vendor it.

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

if [ -d dependencies/AmmarServer/.git ]; then
  echo "Updating AmmarServer.."
  cd dependencies/AmmarServer
  git pull origin master
  cd "$DIR"
fi

# Pull this package too (no-op if not a standalone git checkout).
git pull 2>/dev/null

# Re-copy the AmmClient sources so the vendored, self-contained copy stays fresh.
./initialize.sh

exit 0
