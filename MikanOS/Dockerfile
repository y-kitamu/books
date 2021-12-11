FROM ubuntu:latest

ENV DEBIAN_FRONTEND noninteractive

ARG user
RUN useradd -m ${user} &&\
    usermod -aG sudo ${user}

# ssh (https://docs.docker.com/engine/examples/running_ssh_service/)
RUN apt-get update && apt-get upgrade -y && apt-get install openssh-server -y
RUN mkdir /var/run/sshd
RUN sed -ri 's/^#?PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config

# ansible
RUN apt-get update && apt-get install sudo ansible -y

# tools
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y build-essential llvm-7-dev lld-7 clang-7 nasm acpica-tools uuid-dev\
    qemu-system-x86 qemu-utils xauth unzip qemu-system-gui

RUN apt-get update && apt-get install -y dosfstools

# display
ARG display
RUN echo "export DISPLAY=${display}" >> /etc/profile
