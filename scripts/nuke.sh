#!/usr/bin/env bash
# Thin wrapper so `make nuke` works on POSIX. The logic is in nuke.py.
exec python "$(dirname "$0")/nuke.py" "$@"
