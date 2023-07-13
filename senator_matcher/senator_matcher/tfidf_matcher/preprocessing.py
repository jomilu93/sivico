import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import stanza

# Set up tools for text processing
stop_words = set(stopwords.words('spanish'))
nlp = stanza.Pipeline('es')

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()

    # Remove special characters
    text = re.sub(r'\[.*?\]', '', text)  # remove enclosed text i.e. [este texto esta entre llaves]
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # remove punctuation
    text = re.sub(r'\w*\d\w*', '', text)  # remove alphanumeric characters

    # Tokenization and filtering stop words
    text = word_tokenize(text)
    text = [word for word in text if word not in stop_words]

    # Lemmatization
    doc = nlp(' '.join(text))
    lemmas = []

    for sentence in doc.sentences:
        for word in sentence.words:
            lemmas.append(word.lemma)

    text = ' '.join(lemmas)

    return text
