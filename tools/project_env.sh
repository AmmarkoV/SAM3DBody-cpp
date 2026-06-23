# tools/project_env.sh — sourceable helper (do NOT exec; it must export into your shell)
#
# Activates the project's Python venv (repo-root venv/, created by
# scripts/setup.sh) so wrapper scripts and Python frontends pick up the right
# interpreter and packages.  Sourcing it is always safe: it is a no-op when the
# venv has not been created yet, so scripts that source it keep working on a
# bare checkout.
#
# Usage (from any wrapper script):
#   source "$(dirname "${BASH_SOURCE[0]}")/../tools/project_env.sh"
#
# Companion to tools/trt_env.sh, which puts the bundled TensorRT runtime libs on
# LD_LIBRARY_PATH.  Source both; order does not matter.

# Resolve repo root from THIS file's location (independent of the caller's CWD).
_proj_root="$( cd "$( dirname "${BASH_SOURCE[0]:-$0}" )/.." && pwd )"

# Activate the venv only if it exists and we aren't already inside it.
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$_proj_root/venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$_proj_root/venv/bin/activate"
fi

unset _proj_root
