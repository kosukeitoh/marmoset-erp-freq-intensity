FROM python:3.12-slim

# Minimal deps needed to fetch and run the Miniforge installer
RUN apt-get update && apt-get install -y wget bzip2 git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Miniforge (conda-forge default channel, supports x86_64 + arm64)
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
ENV MINIFORGE_VERSION=24.11.3-2

# Detect arch at build time and install the matching Miniforge binary
RUN ARCH=$(uname -m) && \
    case "$ARCH" in \
        x86_64)  MINIFORGE_ARCH=Linux-x86_64 ;; \
        aarch64) MINIFORGE_ARCH=Linux-aarch64 ;; \
        *) echo "Unsupported architecture: $ARCH" && exit 1 ;; \
    esac && \
    wget --quiet "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-${MINIFORGE_ARCH}.sh" \
         -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p $CONDA_DIR && \
    rm /tmp/miniforge.sh

RUN conda update -n base -y conda && \
    conda --version

# Prefer conda-forge packages over defaults to avoid mixed-channel conflicts
RUN conda config --set channel_priority strict

# MNE-Python environment with EEG/BIDS tooling and Jupyter
RUN conda create -n mne_env -y python=3.12 mne-base mne-bids jupyterlab numpy scipy matplotlib scikit-learn && \
    conda clean -afy

# autoreject and ipympl are not on conda-forge; install via pip into the env
RUN /opt/conda/envs/mne_env/bin/pip install --no-cache-dir autoreject ipympl pingouin

# Make mne_env the active Python for all subsequent layers and at runtime
ENV PATH=/opt/conda/envs/mne_env/bin:$PATH
