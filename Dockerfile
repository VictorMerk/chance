FROM python:3.12-slim

ENV PIP_ROOT_USER_ACTION=ignore PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

# CPU-only torch index keeps the image small.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu .

# `docker run image --max-iters 100` trains with these defaults.
ENTRYPOINT ["gpt-from-scratch-train"]
CMD []
