import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

from sivico.params import *
from sivico.ml_logic.preprocessor import summarize_bert

summarize_bert()

print("✅ data downloaded, translated and pushed to BigQuery \n")

