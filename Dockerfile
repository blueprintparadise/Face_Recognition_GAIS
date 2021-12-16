FROM python:3.6
WORKDIR ./home/akshay/PycharmProjects/gais/Face_Recognition_GAIS/
COPY requirements.txt requirements.txt
COPY templates templates
RUN pip3 install -r requirements.txt
RUN apt-get update

RUN apt-get install ffmpeg libsm6 libxext6  -y
COPY  . .

CMD ["python3","Runner.py"]
