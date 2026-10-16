#!/usr/bin/env bash
# One-time host setup so `docker run --gpus all` can reach the GPU.
#
#   sudo ./install-nvidia-toolkit.sh
#
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "run with sudo: sudo $0" >&2
    exit 1
fi

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update
apt-get install -y nvidia-container-toolkit

nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

echo
echo ">> verifying GPU visibility inside a container..."
docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L || {
    echo "!! GPU not visible — check 'nvidia-smi' on host and docker daemon logs" >&2
    exit 1
}
echo ">> nvidia-container-toolkit OK. 'GPU=1 ./run.sh ...' now works."
