.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y taxifare || :
	@pip install -e .

run_data_extraction:
	python -c 'from sivico.ml_logic.data import get_senator_initiative_data; get_senator_initiative_data()'

run_api:
	uvicorn sivico.api.fast:app --reload