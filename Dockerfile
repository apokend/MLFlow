# FROM continuumio/miniconda3
#
# #WORKDIR /usr/local/deploy-app
#
# COPY . .
#
# RUN pip install --upgrade pip
#
# ADD requirements.txt requirements.txt
#
# RUN pip install -r requirements.txt  -f https://download.pytorch.org/whl/torch_stable.html
#
# RUN pip install mlflow sqlalchemy==1.3.0 psycopg2-binary flask
#
# EXPOSE 5001


FROM python:3.7

WORKDIR /deploy-app

COPY . .

RUN pip install --upgrade pip

RUN pip install mlflow sqlalchemy==1.3.0 psycopg2-binary flask

RUN pip install -r requirements.txt  -f https://download.pytorch.org/whl/torch_stable.html

EXPOSE 5001
