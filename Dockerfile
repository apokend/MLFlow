FROM continuumio/miniconda3

WORKDIR /deploy-app

RUN pip install --upgrade pip

ADD requirements.txt requirements.txt

RUN pip install -r requirements.txt  -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install mlflow sqlalchemy==1.3.0 psycopg2-binary flask

EXPOSE 5001
