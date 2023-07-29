.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y sivico || :
	@pip install -e .

run_tfidf_preprocess:
	python -c 'from sivico.interface.main import tfidf_preprocess; tfidf_preprocess()'

run_tfidf_preprocess_local:
	python -c 'from sivico.interface.main_local import tfidf_preprocess; tfidf_preprocess()'

run_tfidf_batch_preprocess:
	python -c 'from sivico.interface.main import tfidf_batch_preprocess; tfidf_batch_preprocess()'

run_tfidf_batch_preprocess_local:
	python -c 'from sivico.interface.main_local import tfidf_batch_preprocess; tfidf_batch_preprocess()'

run_tfidf_vectorization:
	python -c 'from sivico.interface.main import vectorize_tfidf; vectorize_tfidf()'

run_tfidf_vectorization_local:
	python -c 'from sivico.interface.main_local import vectorize_tfidf; vectorize_tfidf()'

run_beto_preprocess:
	python -c 'from sivico.interface.main import beto_preprocess; beto_preprocess()'

run_beto_preprocess_local:
	python -c 'from sivico.interface.main_local import beto_preprocess; beto_preprocess()'

run_beto_embedding:
	python -c 'from sivico.interface.main import beto_batch_embeddings; beto_batch_embeddings()'

run_beto_embedding_local:
	python -c 'from sivico.interface.main_local import beto_batch_embeddings; beto_batch_embeddings()'

run_data_extraction:
	python -c 'from sivico.text_input_and_summarization.data import get_senator_initiative_data; get_senator_initiative_data()'

run_data_summarization:
	python -c 'from sivico.text_input_and_summarization.summarizer import summarize_beto; summarize_beto()'

get_data_from_bq:
	python -c 'from sivico.text_input_and_summarization.data import get_data_from_bq; get_data_from_bq()'

run_api:
	uvicorn sivico.api.fast:app --reload