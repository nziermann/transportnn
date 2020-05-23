FROM ufoym/deepo:keras-py36 as dev

MAINTAINER Nils Ziermann <docker@nilsziermann.de>

USER root

#Disable hdf5 file locking
ENV HDF5_USE_FILE_LOCKING=FALSE
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/miniconda.sh
RUN bash /root/miniconda.sh -b -p /root/miniconda
RUN /root/miniconda/bin/conda install netCDF4 numpy pip keras matplotlib tensorflow-gpu
RUN /root/miniconda/bin/pip install gcloud gsutil
# If problems occur remove tensorflow-gpu from conda install and move to pip install
RUN pip install gsutil

FROM dev as complete
COPY /src /application
RUN py3clean /application

WORKDIR /application
ENV PYTHONPATH=/application:$PYTHONPATH
# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

RUN mkdir -p /tmp/artifacts

ENTRYPOINT ["/root/miniconda/bin/python3", "src/models/main.py"]