FROM python:3.11-slim

RUN pip install --no-cache-dir uv

RUN useradd -m appuser

WORKDIR /app

# Give ownership of the directory itself
RUN chown appuser:appuser /app

COPY --chown=appuser:appuser . .

USER appuser

RUN uv sync --frozen

EXPOSE 7860
ENV GRADIO_SERVER_NAME=0.0.0.0

CMD ["uv", "run", "appHFBucket.py"]
