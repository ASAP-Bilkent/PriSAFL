#!/usr/bin/env bash
# Build the Pri-SAFL reproduction image (OpenFHE compiled from source).
# Builds on the system dockerd by default; DESKTOP=1 -> Docker Desktop. See README.
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="${IMAGE:-prisafl}"

# default -> native (max opt); GENERIC=1 -> portable x86-64 build. See README.
if [[ "${GENERIC:-0}" == "1" ]]; then
    NATIVEOPT=OFF; MARCH=x86-64
else
    NATIVEOPT=ON;  MARCH=native
fi

docker_cmd=(docker)
host_flag=()
if [[ "${DESKTOP:-0}" != "1" ]]; then
    host_flag=(-H unix:///var/run/docker.sock)
    docker "${host_flag[@]}" info >/dev/null 2>&1 || docker_cmd=(sudo docker)
fi

echo ">> building $IMAGE (NATIVEOPT=$NATIVEOPT march=$MARCH) on ${DESKTOP:+Docker Desktop}${DESKTOP:-system dockerd} (~30 min, OpenFHE compile)"
"${docker_cmd[@]}" "${host_flag[@]}" build \
    --build-arg OPENFHE_MARCH="$MARCH" \
    --build-arg WITH_NATIVEOPT="$NATIVEOPT" \
    -t "$IMAGE" \
    -f Dockerfile \
    .
echo ">> done: $IMAGE"
