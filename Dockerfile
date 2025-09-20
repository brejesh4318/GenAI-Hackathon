FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    curl \
    wget \
    git \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.12 manually
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y python3.12 python3.12-venv python3.12-dev python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.12 -m venv $VIRTUAL_ENV

# Upgrade pip and install poetry
RUN pip install --upgrade pip && pip install gunicorn  poetry

# Disable Poetry's virtualenv
RUN poetry config virtualenvs.create false

# Set working directory
WORKDIR /opt

# Copy dependency files and install
COPY pyproject.toml poetry.lock* /opt/
RUN poetry install --no-interaction --no-ansi
RUN pip install pandas
# Copy application files
COPY . /opt

CMD ["gunicorn", "app.main:app", "--workers", "1", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8080"]
