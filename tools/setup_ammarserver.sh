#!/usr/bin/env bash
# tools/setup_ammarserver.sh — one-shot setup for the networked client/server
# (`sam_3dbody_net`) build.  See CLIENTSERVER.md.
#
# WHAT IT DOES
# ------------
#   1. Ensures ./AmmarServer is present.  If it is MISSING, git-clones
#      https://github.com/AmmarkoV/AmmarServer into it.  If it already exists —
#      including a *symlink* to a working checkout (the dev convention) — it is
#      left untouched so local/upstream work is never clobbered (pass --pull to
#      `git pull` an existing non-symlink checkout).
#   2. Builds the two shared libraries sam_3dbody_net links against:
#         AmmarServer/src/AmmServerlib/libAmmarServer.so   (target AmmarServerDynamic)
#         AmmarServer/src/AmmClient/libAmmClient.so        (target AmmClientDynamic)
#      The shared libs are used (not the .a) because the static AmmarServer lib
#      pulls in sibling libs (Hashmap, InputParser_C) that the .so bundles for us.
#
# Idempotent: re-running skips the build when both libs are already present
# (pass --force to rebuild).
#
# USAGE
#   tools/setup_ammarserver.sh            # clone-if-missing + build the libs
#   tools/setup_ammarserver.sh --pull     # also `git pull` an existing checkout
#   tools/setup_ammarserver.sh --force     # rebuild even if the libs are present
#
# After this, (re-)run CMake on the main project: the sam_3dbody_net target is
# added automatically once these libs are detected.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AMM="$ROOT/AmmarServer"
REPO="https://github.com/AmmarkoV/AmmarServer"

SERVER_LIB="$AMM/src/AmmServerlib/libAmmarServer.so"
CLIENT_LIB="$AMM/src/AmmClient/libAmmClient.so"

PULL=0
FORCE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --pull)    PULL=1;  shift ;;
        --force)   FORCE=1; shift ;;
        -h|--help) grep '^#' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

# ── 1. Ensure the source tree is present ─────────────────────────────────────
if [ -e "$AMM" ]; then
    if [ -L "$AMM" ]; then
        echo "=== AmmarServer is a symlink → $(readlink "$AMM") (dev checkout, left as-is) ==="
    else
        echo "=== AmmarServer checkout present at $AMM ==="
        if [ "$PULL" -eq 1 ]; then
            echo "  git pull…"
            git -C "$AMM" pull --ff-only
        fi
    fi
else
    echo "=== Cloning AmmarServer into $AMM ==="
    git clone --depth 1 "$REPO" "$AMM"
fi

# Resolve through the symlink so the build artifacts land in the real tree.
AMM_REAL="$(cd "$AMM" && pwd -P)"

# ── 2. Build the two shared libs ─────────────────────────────────────────────
if [ -f "$SERVER_LIB" ] && [ -f "$CLIENT_LIB" ] && [ "$FORCE" -eq 0 ]; then
    echo "=== libAmmarServer.so + libAmmClient.so already built — skipping (use --force to rebuild) ==="
else
    echo "=== Building AmmarServer shared libs ==="
    BUILD="$AMM_REAL/build"
    mkdir -p "$BUILD"
    cmake -S "$AMM_REAL" -B "$BUILD" >/dev/null
    # Only the two dynamic libs we link against; their sibling deps (Hashmap,
    # InputParser_C) are built automatically as target dependencies.  This skips
    # compiling the unrelated AmmarServer service executables.
    cmake --build "$BUILD" --target AmmarServerDynamic AmmClientDynamic -j"$(nproc)"
fi

# ── 3. Verify ────────────────────────────────────────────────────────────────
ok=1
for lib in "$SERVER_LIB" "$CLIENT_LIB"; do
    if [ -f "$lib" ]; then echo "  OK  $lib"; else echo "  MISSING  $lib" >&2; ok=0; fi
done
[ "$ok" -eq 1 ] || { echo "AmmarServer libs missing after build — check the build log above." >&2; exit 1; }

echo
echo "=== AmmarServer setup complete ==="
echo "Re-run CMake on the main project to enable sam_3dbody_net:"
echo "    cmake -S \"$ROOT\" -B \"$ROOT/build\" && cmake --build \"$ROOT/build\" --target sam_3dbody_net"
