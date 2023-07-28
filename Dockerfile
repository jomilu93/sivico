FROM python:3.10.6-buster

COPY requirements_prod.txt requirements.txt
RUN pip install -r requirements.txt

COPY sivico sivico
COPY setup.py setup.py
RUN pip install .

# should remove these once we're able to fetch data directly from GCP
COPY data data
COPY tfidf_model tfidf_model

COPY Makefile Makefile

CMD uvicorn sivico.api.fast:app --host 0.0.0.0 --port $PORT