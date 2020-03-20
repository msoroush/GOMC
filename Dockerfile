# GCC support can be specified at major, minor, or micro version
# (e.g. 8, 8.2 or 8.2.0).
# See https://hub.docker.com/r/library/gcc/ for all supported GCC
# tags from Docker Hub.
# See https://docs.docker.com/samples/library/gcc/ for more on how to use this image
FROM nvidia/cuda:latest

# These commands copy your files into the specified directory in the image
# and set that as the working location
COPY . /usr/src/GOMC
WORKDIR /usr/src/GOMC

# Install necessary packages
RUN apt-get -y update
RUN apt-get -y install wget
RUN apt-get -y install cmake
RUN apt-get -y install unzip

# Compiles GOMC
RUN ./metamake.sh

LABEL Name=gomc Version=0.0.1
