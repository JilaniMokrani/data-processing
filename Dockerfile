FROM ubuntu
LABEL MAINTAINER="Jilani Mokrani" 
RUN apt update -y
RUN apt upgrade -y
RUN apt install -y libsndfile1
RUN apt install -y python3-pip
RUN pip3 install librosa seaborn pycm==0.8.1
COPY main.py .
ENV ROOT_DATA_PATH=/data
CMD python3 main.py