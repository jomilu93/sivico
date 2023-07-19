FROM python:3.10.6-buster

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY sivico sivico
COPY setup.py setup.py
COPY data data
COPY tfidf_model tfidf_model
RUN pip install .

COPY Makefile Makefile

CMD uvicorn sivico.api.fast:app --host 0.0.0.0 --port $PORT