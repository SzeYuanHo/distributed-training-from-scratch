#!/usr/bin/env bash
# Thin wrapper so `make kill` works on POSIX. The logic is in kill_stragglers.py.
exec python "$(dirname "$0")/kill_stragglers.py" "$@"
