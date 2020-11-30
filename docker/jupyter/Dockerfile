FROM jupyter/tensorflow-notebook

USER root

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get -y install \
        libeccodes0 \
        libgeos-dev \
        libproj-dev \
        proj-bin \
        cdo \
        libgdal-dev \
        gdal-bin \
        ffmpeg

USER jovyan

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

RUN conda config --add channels conda-forge && \
    conda config --set channel_priority strict

#RUN CPLUS_INCLUDE_PATH=/usr/include/gdal C_INCLUDE_PATH=/usr/include/gdal \
#    conda -c conda-forge install gdal==2.1.3

RUN conda install -c conda-forge gdal cartopy lxml

#COPY requirements.txt /tmp/requirements.txt
#RUN pip install -r /tmp/requirements.txt

#RUN conda install --yes --file /tmp/requirements.txt && \
#    fix-permissions $CONDA_DIR && \
#    fix-permissions /home/$NB_USER

RUN pip install \
    sentinelsat \
    pydap \
    nasadap \
    h5py \
    cmocean \
    pendulum \
    iso8601

RUN fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER
