FROM ufoym/deepo:keras-py36

MAINTAINER Nils Ziermann <docker@nilsziermann.de>

USER root

#Disable hdf5 file locking
ENV HDF5_USE_FILE_LOCKING=FALSE
RUN pip install --upgrade netCDF4 numpy gcloud gsutil
RUN pip uninstall --yes h5py

COPY /src /application
WORKDIR /application
ENV PYTHONPATH=/application:$PYTHONPATH
# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

ENTRYPOINT ["python", "src/models/cnn.py"]