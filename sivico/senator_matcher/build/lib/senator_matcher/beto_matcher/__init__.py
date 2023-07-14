import os
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch

# Load the BETO model and tokenizer
TOKENIZER = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')
BETO_MODEL = BertModel.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')
