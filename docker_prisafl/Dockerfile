# Pri-SAFL reproduction image
FROM ubuntu:24.04

ARG OPENFHE_MARCH=native
ENV DEBIAN_FRONTEND=noninteractive

# retry wrapper: re-run a download on failure (transient CDN/network corruption).
COPY --chmod=755 context/retry.sh /usr/local/bin/retry

# --- system build deps ---
RUN retry apt-get -o Acquire::Retries=3 update \
    && retry apt-get -o Acquire::Retries=3 install -y --no-install-recommends \
        build-essential cmake git wget ca-certificates pkg-config \
        libomp-dev autoconf \
    && rm -rf /var/lib/apt/lists/*

# --- miniforge (conda-forge channel) ---
ARG MINIFORGE_VERSION=25.11.0-1
RUN retry sh -c 'rm -rf /opt/conda /tmp/miniforge.sh \
        && wget -q --tries=5 --retry-connrefused -O /tmp/miniforge.sh \
            "https://github.com/conda-forge/miniforge/releases/download/'"${MINIFORGE_VERSION}"'/Miniforge3-Linux-x86_64.sh" \
        && bash /tmp/miniforge.sh -b -p /opt/conda' \
    && rm -f /tmp/miniforge.sh
ENV PATH=/opt/conda/bin:$PATH

# --- conda env (prisafl): py3.11.4 ---
RUN conda create -y -n prisafl python=3.11.4 && conda clean -afy

SHELL ["conda", "run", "--no-capture-output", "-n", "prisafl", "/bin/bash", "-c"]

# --- python deps (torch cu128 first so survival libs reuse it) ---
COPY context/requirements-torch.txt context/requirements.txt /tmp/
RUN retry pip install --no-cache-dir --retries 5 --timeout 60 -r /tmp/requirements-torch.txt
RUN retry pip install --no-cache-dir --retries 5 --timeout 60 -r /tmp/requirements.txt

RUN retry apt-get -o Acquire::Retries=3 update \
    && retry apt-get -o Acquire::Retries=3 install -y --no-install-recommends \
        lsb-release wget software-properties-common gnupg \
    && retry wget -q --tries=5 --retry-connrefused -O /tmp/llvm.sh https://apt.llvm.org/llvm.sh \
    && chmod +x /tmp/llvm.sh && retry /tmp/llvm.sh 20 \
    && retry apt-get -o Acquire::Retries=3 install -y --no-install-recommends libomp-20-dev \
    && rm -f /tmp/llvm.sh && rm -rf /var/lib/apt/lists/* \
    && PKG="$(python -c 'import site;print(site.getsitepackages()[0])')" \
    && mkdir -p "$PKG/pycox/datasets/data" \
    && chmod -R a+rwX "$PKG/pycox/datasets"

# --- OpenFHE C++ 1.3.0 from source ---
ARG OPENFHE_DEV_TAG=v1.3.0
ARG OPENFHE_PY_TAG=v1.3.0.0
# WITH_NATIVEOPT=ON -> -march=native tuning. GENERIC=1 ./build.sh sets OFF for a
# portable build where native is unstable (see README).
ARG WITH_NATIVEOPT=ON
RUN git clone --branch ${OPENFHE_DEV_TAG} --depth 1 \
        --recurse-submodules --shallow-submodules \
        https://github.com/openfheorg/openfhe-development /opt/src/openfhe-development
RUN cmake -S /opt/src/openfhe-development -B /opt/src/openfhe-development/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=clang-20 \
        -DCMAKE_CXX_COMPILER=clang++-20 \
        -DWITH_NATIVEOPT=${WITH_NATIVEOPT} \
        -DNATIVE_SIZE=64 \
        -DMATHBACKEND=4 \
        -DWITH_OPENMP=ON \
        -DBUILD_UNITTESTS=OFF -DBUILD_BENCHMARKS=OFF -DBUILD_EXAMPLES=OFF \
        -DCMAKE_CXX_FLAGS="-march=${OPENFHE_MARCH}" \
    && cmake --build /opt/src/openfhe-development/build -j"$(nproc)" \
    && cmake --install /opt/src/openfhe-development/build \
    && ldconfig

# --- OpenFHE python binding (py3.11) ---
# CMake mis-detects site-packages (installs to env prefix root), so we cp the
# built .so into real site-packages ourselves.
RUN git clone --branch ${OPENFHE_PY_TAG} --depth 1 --recurse-submodules \
        https://github.com/openfheorg/openfhe-python /opt/src/openfhe-python
RUN cmake -S /opt/src/openfhe-python -B /opt/src/openfhe-python/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=clang-20 \
        -DCMAKE_CXX_COMPILER=clang++-20 \
        -DPYTHON_EXECUTABLE_PATH="$(which python)" \
        -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
    && cmake --build /opt/src/openfhe-python/build -j"$(nproc)" \
    && cp /opt/src/openfhe-python/build/openfhe*.so \
          "$(python -c 'import site; print(site.getsitepackages()[0])')/" \
    && ldconfig \
    && python -c "import openfhe; print('openfhe import OK:', hasattr(openfhe,'GenCryptoContext'))"

ENV LD_LIBRARY_PATH=/usr/local/lib
ENV PYTHONUNBUFFERED=1

# repo mounted here at runtime
WORKDIR /work

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "prisafl", "/bin/bash", "-lc"]
CMD ["bash"]
