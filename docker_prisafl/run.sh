#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-prisafl}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# CPUSET unset -> all host cores
cpuset_flag=()
[[ -n "${CPUSET:-}" ]] && cpuset_flag=(--cpuset-cpus="$CPUSET")

gpu_flag=()
docker_cmd=(docker)
host_flag=()

if [[ "${GPU:-0}" == "1" || "${SYS:-0}" == "1" ]]; then
    host_flag=(-H unix:///var/run/docker.sock)
    [[ "${GPU:-0}" == "1" ]] && gpu_flag=(--gpus all)
    docker "${host_flag[@]}" info >/dev/null 2>&1 || docker_cmd=(sudo docker)
fi

# interactive only on a tty
tty_flag=(-i)
[[ -t 0 ]] && tty_flag=(-it)

cmd="${*:-bash}"

exec "${docker_cmd[@]}" "${host_flag[@]}" run --rm "${tty_flag[@]}" \
    "${gpu_flag[@]}" \
    --user "$(id -u):$(id -g)" -e HOME=/tmp \
    "${cpuset_flag[@]}" \
    -v "$REPO":/work \
    -w /work \
    "$IMAGE" "$cmd"
