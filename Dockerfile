# Hosted MCP server: Streamable HTTP at /mcp, plus the upload page and report pages.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml .
COPY prompts ./prompts
COPY src ./src
RUN pip install --no-deps -e .

EXPOSE 8000
CMD ["python", "-m", "forensic_rag.mcp_server", "--http"]
