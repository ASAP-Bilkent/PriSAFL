#!/bin/sh
# Retry a command up to RETRY_MAX times (default 5). Handles download corruption / CDN hiccups
set -e
n=0
max="${RETRY_MAX:-5}"
until "$@"; do
    n=$((n + 1))
    if [ "$n" -ge "$max" ]; then
        echo "retry: '$*' still failing after $max attempts" >&2
        exit 1
    fi
    echo ">> retry $n/$max after failure: $*" >&2
    sleep 5
done
