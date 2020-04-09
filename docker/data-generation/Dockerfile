FROM ubuntu:bionic

RUN apt-get update \
 && apt-get install -yqq build-essential gfortran python wget curl git

WORKDIR /petsc
RUN wget http://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-3.7.4.tar.gz
RUN tar -zxf petsc-3.7.4.tar.gz

WORKDIR /petsc/petsc-3.7.4
RUN ./configure --with-cc=gcc --with-cxx=g++ --with-fc=gfortran --download-mpich --download-fblaslapack
RUN make all test

ENV PETSC_DIR=/petsc/petsc-3.7.4
ENV PETSC_ARCH=arch-linux2-c-debug

RUN yes "yes" | python -c "$(curl -fsSL https://github.com/metos3d/metos3d/raw/master/install.py)"

#Generate zero model
WORKDIR /root/.metos3d/metos3d
RUN ./metos3d simpack ZERO clean
RUN ./metos3d simpack ZERO
RUN ./metos3d simpack N-DOP clean
RUN ./metos3d simpack N-DOP
COPY ./docker/data-generation/files /files
COPY ./docker/data-generation/scripts /scripts

COPY ./tools /tools
COPY ./tools/petsc2nc.py /root/.metos3d/metos3d/petsc2nc.py

RUN apt-get update \
 && apt-get install -yqq python3 python3-pip

RUN pip3 install --upgrade netCDF4 numpy gcloud gsutil mako pyyaml

RUN mkdir -p /tmp/metos3d
RUN mkdir -p /tmp/metos3d/input
RUN mkdir -p /tmp/metos3d/output
RUN mkdir -p /tmp/metos3d/nc

CMD /scripts/validation-data.sh
