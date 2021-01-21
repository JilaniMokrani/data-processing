FROM python:3.6
USER root
LABEL MAINTAINER="Jilani Mokrani" 
RUN apt update -y
RUN apt install -y libsndfile1 python3-pip
RUN pip3 install librosa seaborn pycm==0.8.1
ENV INPUT_DATA_PATH=/data
ENV OUTPUT_DATA_PATH=/root/output
COPY main.py .
CMD python3 main.py && bash
