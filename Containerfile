# Build from Red Hat Universal Base Image
FROM redhat/ubi9-minimal:latest

# Update packages and install required dependencies
RUN microdnf -y upgrade
RUN microdnf -y install python3.12

# Since we do not need root anymore, switch to normal user
RUN useradd -m builder
USER builder
WORKDIR /home/builder
