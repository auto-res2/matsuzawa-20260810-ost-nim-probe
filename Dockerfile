# Execution environment for the shared OST / PLINDER feasibility probe.
#
# Every instruction below carries its purpose in a comment on purpose: this
# image is not built verbatim but regenerated from intent, and instructions
# whose reason is legible survive that regeneration while unexplained ones
# get dropped.
#
# Note what is deliberately NOT here: OpenStructure, conda/micromamba, and
# the PLINDER data. The probe exists to find out whether the copies already
# installed on the cluster's shared filesystem can be used from inside this
# container, so installing our own would destroy the thing being measured.

# Plain Python base: the probe runs no GPU or numerical code, so a CUDA base
# image would only add size and build risk.
FROM python:3.11-slim

# Unbuffered output so the probe's findings appear in the run log as they
# happen rather than being lost in a buffer if the job is cut short.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# uv is the dependency installer the run contract invokes (`uv run ...`).
# Installed via pip rather than copied from another image, because a
# cross-image COPY has failed at the build stage on this cluster before.
RUN pip install --no-cache-dir uv

WORKDIR /workspace

# Dependency manifests first, so the install layer is cached and only
# rebuilt when the dependencies themselves change.
COPY pyproject.toml ./
COPY uv.lock ./

# --frozen installs exactly what uv.lock pins, with no resolution freedom.
# It is deliberately NOT followed by a bare `uv sync` fallback: a fallback
# would silently paper over a missing or stale lock file, which is how an
# unintended version reaches the target architecture unnoticed.
RUN uv sync --frozen

# The probe source and its Hydra config.
COPY . .

# The results directory the run contract writes reports into.
RUN mkdir -p .research/results

CMD ["bash"]
