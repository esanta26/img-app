# syntax=docker/dockerfile:1

FROM python:3.7-buster

WORKDIR /img-app

RUN python3 -m pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .

CMD ["python3", "src/Api.py"]