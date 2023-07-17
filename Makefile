.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y sivico || :
	@pip install -e .

run_tfidf_preprocess:
	python -c 'from sivico.interface.main_local import preprocess; preprocess()'

# run_data_extraction:
# 	python -c 'from sivico.ml_logic.data import get_senator_initiative_data; get_senator_initiative_data()'

# run_api:
# 	uvicorn sivico.api.fast:app --reload