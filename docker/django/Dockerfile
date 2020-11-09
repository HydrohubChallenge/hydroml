FROM python:3.8
LABEL authors="Roberto Santos, Tadashi Igarashi"

ENV PYTHONUNBUFFERED 1

EXPOSE 8000

RUN addgroup -q django && adduser -q --ingroup django --disabled-password django

RUN apt-get update && \
    apt-get install -y \
        python3-gdal \
        libgeos++-dev \
        libproj-dev \
        openjdk-11-jdk \
        vim \
        htop \
        wait-for-it

RUN mkdir /api

WORKDIR /api

COPY requirements.txt /requirements.txt

RUN pip install -r /requirements.txt

COPY entrypoint.sh /entrypoint.sh

