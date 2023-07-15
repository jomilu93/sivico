import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

from sivico.params import *
from sivico.ml_logic.data import get_data_with_cache, clean_data, load_data_to_bq
from sivico.ml_logic.model import initialize_model, compile_model, train_model, evaluate_model
from sivico.ml_logic.preprocessor import preprocess_features
from sivico.ml_logic.registry import load_model, save_model, save_results

load_data_to_bq(
        data_processed_with_timestamp,
        gcp_project=GCP_PROJECT,
        bq_dataset=BQ_DATASET,
        table=f'processed_senators',
        truncate=True
    )