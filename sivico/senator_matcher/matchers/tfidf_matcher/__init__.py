import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
import re
import stanza
import string
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Downloads necessary resources
nltk.download('stopwords')
nltk.download('punkt')
stanza.download('es')
