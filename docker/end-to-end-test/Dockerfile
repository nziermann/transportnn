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

COPY ./docker/end-to-end-test/scripts /scripts

#Generate zero model
WORKDIR /root/.metos3d/metos3d
RUN ./metos3d simpack ZERO clean
RUN ./metos3d simpack ZERO
RUN ./metos3d simpack N-DOP clean
RUN ./metos3d simpack N-DOP
COPY ./docker/end-to-end-test/files /files

COPY ./tools /tools
COPY ./tools/petsc2nc.py /root/.metos3d/metos3d/petsc2nc.py

RUN apt-get update \
 && apt-get install -yqq python3 python3-pip

RUN pip3 install --upgrade netCDF4 numpy gcloud gsutil mako pyyaml

RUN mkdir -p /tmp/metos3d
RUN mkdir -p /tmp/metos3d/input
RUN mkdir -p /tmp/metos3d/output
RUN mkdir -p /tmp/metos3d/nc


#CMD [ \
#    "python3", \
#    "/tools/generate_testdata.py", \
#    "--input-dir", "/tmp/metos3d/input/", \
#    "--input-file", "DUMMY.petsc", \
#    "--output-dir", "/root/.metos3d/metos3d/work/", \
#    "--output-file", "DUMMY.petsc", \
#    "--nc-output-dir", "/tmp/metos3d/nc/", \
#    "--nc-output-file", "DUMMY.nc", \
#    "--metos-location", "/root/.metos3d/metos3d/", \
#    "--metos-model-simpack", "/root/.metos3d/metos3d/metos3d-simpack-ZERO.exe", \
#    "--metos-petsc2nc-conf", "/files/DUMMY.petsc2nc.conf.yaml", \
#    "--template-config-file", "/files/template.ZERO.option.txt", \
#    "--config-file-location", "/tmp/metos3d/adapted.ZERO.option.txt", \
#    "--normalized-volume-file", "/root/.metos3d/data/data/TMM/2.8/Geometry/normalizedVolumes.petsc", \
#    "--grid-file", "/root/.metos3d/data/data/mitgcm-128x64-grid-file.nc", \
#    "--grid-file", "/root/.metos3d/data/data/mitgcm-128x64-grid-file.nc", \
#    "--spinup-count", "3000", \
#    "--number-of-runs", "1", \
#    "--length-of-run", "240", \
#    "--output-file-prefix", "$03d-", \
#    "--normalized-value", "2.17", \
#    "-- spinup-model-simpack", "/root/.metos3d/metos3d/metos3d-simpack-N-DOP.exe", \
#    "--spinup-template-config-file-location", "/files/spinup_template.N-DOP.option.txt", \
#    "--spinup-config-file-location", "/tmp/metos3d/spinup_adapted.N-DOP.option.txt"
#]

CMD [ \
    "python3", \
    "/tools/generate_testdata.py", \
    "--input-dir", "/root/.metos3d/metos3d/work/", \
    "--input-file", "N.petsc", \
    "--output-dir", "/root/.metos3d/metos3d/work/", \
    "--output-file", "DUMMY.petsc", \
    "--nc-output-dir", "/tmp/metos3d/nc/", \
    "--nc-output-file", "DUMMY.nc", \
    "--metos-location", "/root/.metos3d/metos3d/", \
    "--metos-model-simpack", "/root/.metos3d/metos3d/metos3d-simpack-ZERO.exe", \
    "--metos-petsc2nc-conf", "/files/DUMMY.petsc2nc.conf.yaml", \
    "--template-config-file", "/files/template.ZERO.option.txt", \
    "--config-file-location", "/tmp/metos3d/adapted.ZERO.option.txt", \
    "--normalized-volume-file", "/root/.metos3d/data/data/TMM/2.8/Geometry/normalizedVolumes.petsc", \
    "--grid-file", "/root/.metos3d/data/data/mitgcm-128x64-grid-file.nc", \
    "--grid-file", "/root/.metos3d/data/data/mitgcm-128x64-grid-file.nc", \
    "--spinup-count", "2880", \
    "--number-of-runs", "1", \
    "--length-of-run", "240", \
    "--output-file-prefix", "$03d-", \
    "--normalized-value", "2.17", \
    "--spinup-model-simpack", "/root/.metos3d/metos3d/metos3d-simpack-N-DOP.exe", \
    "--spinup-template-config-file-location", "/files/spinup_template.N-DOP.option.txt", \
    "--spinup-config-file-location", "/tmp/metos3d/spinup_adapted.N-DOP.option.txt" \
]

#--input-dir /tmp/metos3d/input/ --input-file DUMMY.petsc --output-dir /root/.metos3d/metos3d/work/ --output-file DUMMY.petsc --nc-output-dir /tmp/metos3d/nc/ --nc-output-file DUMMY.nc --metos-location /root/.metos3d/metos3d/ --metos-model-simpack /root/.metos3d/metos3d/metos3d-simpack-ZERO.exe --metos-petsc2nc-conf /files/DUMMY.petsc2nc.conf.yaml --template-config-file /files/template.ZERO.option.txt --config-file-location /tmp/metos3d/adapted.ZERO.option.txt --normalized-volume-file /root/.metos3d/data/data/TMM/2.8/Geometry/normalizedVolumes.petsc --grid-file /root/.metos3d/data/data/mitgcm-128x64-grid-file.nc --spinup-count 1 --number-of-runs 1 --length-of-run 240 --output-file-prefix $03d --normalized-value 2.17