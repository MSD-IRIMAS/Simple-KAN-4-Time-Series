FROM pytorch/pytorch:latest

ARG USER_ID=1000
ARG GROUP_ID=1000

RUN groupadd -r -g $GROUP_ID myuser && useradd -r -u $USER_ID -g myuser -m -d /home/myuser myuser
ENV SHELL /bin/bash

RUN mkdir -p /home/myuser/code && chown -R myuser:myuser /home/myuser/code

WORKDIR /home/myuser/code

RUN apt update
RUN pip install --upgrade pip
RUN pip install numpy==1.24.4
RUN pip install pandas==2.0.3
RUN pip install scikit-learn==1.1.3
RUN pip install matplotlib==3.6.2
RUN pip install aeon==0.8.1
RUN pip install hydra-core --upgrade
RUN pip install omegaconf
RUN pip install black==23.11.0
RUN pip install pykan
RUN pip install setuptools==65.5.0
RUN pip install sympy==1.11.1
RUN pip install tqdm==4.66.2