FROM python:3.6
WORKDIR ./home/akshay/PycharmProjects/gais/Face_Recognition_GAIS/
COPY requirements.txt requirements.txt
COPY templates templates
RUN pip3 install -r requirements.txt
RUN apt-get -y update && apt-get install -y --fix-missing \
    build-essential \
    cmake \
    gfortran \
    git \
    wget \
    curl \
    graphicsmagick \
    libgraphicsmagick1-dev \
    libatlas-base-dev \
    libavcodec-dev \
    libavformat-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    pkg-config \
    python3-dev \
    python3-numpy \
    software-properties-common \
    zip \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*
RUN apt-get update
EXPOSE 8080
EXPOSE 4747
RUN apt-get install ffmpeg libsm6 libxext6  -y
COPY  . .
CMD ["python3","Runner.py"]
