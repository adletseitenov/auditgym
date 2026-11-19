# No packages are installed: the solution uses the Python standard library only.
# Before hand-in, pin the base image by digest (python:3.11-slim@sha256:...).
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY . /app
CMD ["python", "-m", "auditgym", "check"]
