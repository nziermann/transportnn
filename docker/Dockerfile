FROM ufoym/deepo:keras-py36

MAINTAINER Nils Ziermann <docker@nilsziermann.de>

#Install conda
RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion && \
    apt-get clean

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.7.12-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy


#Install packages from packagelist
COPY environment.yml ./
RUN /opt/conda/bin/conda env create --file environment.yml

# Activate env
RUN /opt/conda/bin/conda init bash
RUN echo "conda activate transportnn" >> ~/.bashrc

#Disable hdf5 file locking
ENV HDF5_USE_FILE_LOCKING=FALSE
ENV PATH=/opt/conda/envs/transportnn/bin:$PATH

ENV PYTHONPATH=/paperspace:$PYTHONPATH

USER root
