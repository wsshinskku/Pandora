FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
COPY examples ./examples
RUN python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN python -m pip install .
ENTRYPOINT ["python", "-m", "pandora"]
CMD ["run", "--config", "configs/smoke.yaml", "--output", "runs/docker"]
