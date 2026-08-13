FROM python:3.11-slim-bookworm

ARG APP_UID=1000
ARG APP_GID=1000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --no-compile -r requirements.txt

RUN groupadd --gid "${APP_GID}" app \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home --no-log-init app \
    && mkdir -p /app/outputs \
    && chown -R app:app /app

COPY --chown=app:app src ./src

USER 1000:1000

VOLUME ["/app/outputs"]

CMD ["python", "-m", "src.run_pipeline", "--output-dir", "/app/outputs"]