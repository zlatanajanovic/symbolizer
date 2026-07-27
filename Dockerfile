FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        swi-prolog \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY pddlgym ./pddlgym
COPY symbolizer ./symbolizer
COPY experiments ./experiments
COPY examples ./examples
COPY requirements.txt setup.py README.md config.yaml .env_template ./

RUN pip install --upgrade pip setuptools wheel \
    && pip install -e ./pddlgym \
    && pip install -e .

CMD ["bash"]
