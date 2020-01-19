FROM ubuntu:bionic

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  cmake \
  ninja-build \
  git \
  libssl-dev \
  ca-certificates

RUN mkdir /build && cd /build && git clone https://github.com/opencv/opencv.git && \
  cd opencv && mkdir build && cd build && cmake .. -GNinja && \
  ninja && ninja install

RUN apt-get install -y libboost-all-dev wget clang clang-format

ENV CC /usr/bin/clang

ENV CXX /usr/bin/clang++

RUN apt-get install -y cppcheck

