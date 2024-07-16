# hash:sha256:5c12807ab1121b1b53c8dab94f1a45f5eb492eb97fae30e160dc56a0a05fe650
FROM registry.codeocean.com/codeocean/ubuntu:20.04.2

RUN apt-get update -y
RUN apt-get install -y make
RUN apt-get install -y gcc zlib1g-dev bzip2 libbz2-dev libreadline8 libreadline-dev sqlite3 libsqlite3-dev openssl libssl-dev build-essential liblzma-dev
RUN apt-get install -y git curl wget
RUN apt-get install -y vim
RUN apt-get install -y sudo
RUN apt-get install -y libffi-dev

RUN apt-get install -y lsb-release gnupg
RUN apt-get install -y python3.8 python3-pip