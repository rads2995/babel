# Build from Docker's official Debian image
FROM docker.io/debian:bookworm-slim

# Update packages and install required dependencies
RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg

# Since we do not need root anymore, switch to normal user
RUN useradd -m builder
USER builder
WORKDIR /home/builder
