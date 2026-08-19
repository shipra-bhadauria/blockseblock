# ═══════════════════════════════════════════════════════════════════════════════
# Dockerfile — AI Engineering Bootcamp (BlockseBlock) · Feature 12: Ship It
#
# YOUR TASK: Fill in the three TODO sections below.
#   TODO 1 — choose a base image for the builder stage
#   TODO 2 — create a non-root user and own the app directories
#   TODO 3 — write the CMD that starts the server
#
# The rest of the file is complete. You're adding the three pieces that turn
# a working script into a production-grade container.
# ═══════════════════════════════════════════════════════════════════════════════

# ── Stage 1: dependency builder ───────────────────────────────────────────────
# TODO 1: Replace ??? with the correct base image.
#   Hint: we want Python 3.12, slim variant, as the builder stage.
#   Pattern: python:<version>-<variant> AS <stage-name>
FROM python:3.12-slim AS builder

WORKDIR /build

COPY blockseblock-ai-bootcamp/requirements.txt .

RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY shared/ ./shared/
COPY ui/ ./ui/
COPY app/ .

RUN mkdir -p /data/chroma /data/uploads

# TODO 2: Add a non-root user for security.
#   a) Create a user named "appuser" with a home directory.
#      Hint: useradd --create-home <username>
#   b) Give appuser ownership of /app and /data.
#      Hint: chown -R <username> /app /data
#   c) Switch to that user.
#      Hint: USER <username>
#
# Why? If your app is exploited, an attacker running as appuser cannot
# modify system files or escape to other containers.
#
# YOUR CODE HERE
RUN useradd --create-home appuser && \
    chown -R appuser /app /data
USER appuser

EXPOSE 8000

# TODO 3: Write the CMD that starts the server.
#   Use uvicorn to serve main:app on host 0.0.0.0, port 8000, with 1 worker.
#   Hint: CMD ["uvicorn", "main:app", ...]
#
# YOUR CODE HERE
CMD ["uvicorn", "main:app", "--host","0.0.0.0", "--port", "8000", "--workers", "1"]
