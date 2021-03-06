FROM python:3.8-slim

WORKDIR /usr/src/app

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        icu-devtools \
        libicu-dev \
        build-essential \
        pkg-config \
        git \
        gfortran \
        libopenblas-dev \
        liblapack-dev \
        && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install -e .

ENTRYPOINT ["followthemoney-predict"]
