# TAS-AI Docker Image
#
# Build:
#   docker build -t tasai .
#
# Run dashboard:
#   docker run -p 8050:8050 tasai tasai-dashboard --host 0.0.0.0
#
# Run example:
#   docker run tasai python -m tasai.examples.example_parameter_determination
#
# Interactive shell:
#   docker run -it tasai bash

FROM mambaorg/micromamba:1.5-jammy

# Set working directory
WORKDIR /app

# Copy the entire application
COPY --chown=$MAMBA_USER:$MAMBA_USER . /app/

# Install git for development
USER root
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
USER $MAMBA_USER

# Create environment and install package
RUN micromamba create -n tasai -f environment.yml -y && \
    micromamba clean --all --yes && \
    micromamba run -n tasai pip install -e . --no-deps

# PySpinW bundle is optional; skip if not available in the archive.
# COPY --chown=$MAMBA_USER:$MAMBA_USER pyspinw_pkg/ /app/pyspinw_pkg/
# RUN micromamba run -n tasai pip install /app/pyspinw_pkg/ --no-deps
RUN echo "Building analytical-only image. PySpinW skipped."

# Set the default environment
ENV ENV_NAME=tasai

# Expose dashboard port
EXPOSE 8050

# Use micromamba run as entrypoint
ENTRYPOINT ["micromamba", "run", "-n", "tasai"]

# Default command
CMD ["python", "-c", "import tasai; print('TAS-AI ready!')"]
