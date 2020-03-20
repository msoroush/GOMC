FROM nvidia/cuda:latest

# These commands copy your files into the specified directory in the image
# and set that as the working location
COPY . /opt/GOMC
WORKDIR /opt/GOMC

# Install necessary packages
RUN apt-get -y update
RUN apt-get -y install wget
RUN apt-get -y install cmake
RUN apt-get -y install unzip

# Compiles GOMC
RUN ./metamake.sh

# Copy binaries to /usr/bin/
RUN cp bin/GOMC_?PU_* /usr/bin/
RUN cd /opt

# Lable name
LABEL Name=gomc Version=0.0.1
