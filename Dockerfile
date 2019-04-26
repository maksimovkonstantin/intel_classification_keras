FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

ENV DEBIAN_FRONTEND=noninteractive

ARG user_name
ARG user_id

RUN apt-get update
RUN apt-get install software-properties-common -y
RUN add-apt-repository ppa:jonathonf/python-3.6
RUN apt-get update
RUN apt-get install -y build-essential python3.6 python3.6-dev python3-pip python3.6-venv python-opencv
RUN apt-get install libturbojpeg
RUN python3.6 -m pip install pip --upgrade
RUN python3.6 -m pip install wheel

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.6 0

RUN pip install albumentations image-classifiers keras-metrics scikit-learn
RUN pip install tensorflow-gpu==1.12.0 scikit-image opencv-python matplotlib tqdm keras pandas
RUN pip install jpeg4py

RUN pip install torch torchvision

RUN mkdir /home/$user_name
RUN useradd -u $user_id $user_name
RUN chown $user_name:$user_name /home/$user_name
RUN usermod -aG sudo $user_name

USER  $user_name
WORKDIR /project