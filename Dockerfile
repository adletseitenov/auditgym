# The solution is the Python standard library only: nothing is installed, nothing is fetched at build
# or run time, and the check runs with `--network none`.
# Base image pinned by digest (the multi-architecture index of python:3.11-slim, Python 3.11.18, taken
# 2026-11-19 with `docker pull python:3.11-slim && docker inspect --format '{{index .RepoDigests 0}}'
# python:3.11-slim`). To move to a newer image, repeat that command and change the digest here.
FROM python:3.11-slim@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY . /app
# Default: the fast check (timings: README, section "Проверка одной командой"). The full one:
#   docker run --rm --network none auditgym python -m auditgym check --full
CMD ["python", "-m", "auditgym", "check"]
