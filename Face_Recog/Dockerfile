# Docker 
FROM  python:3.8.10

ADD Runner.py .

COPY requirements.txt requirements.txt

RUN pip3 install -r ./requirements.txt

CMD ['python','./Runner.py']

