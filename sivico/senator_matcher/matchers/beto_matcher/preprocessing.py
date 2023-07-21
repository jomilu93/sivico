import re
import ast

def preprocess_text_for_beto(text):
    initiatives_list = ast.literal_eval(text)
    preprocessed_initiatives_list = [preprocess_initiative_string(initiative) for initiative in initiatives_list]
    text = '. '.join(preprocessed_initiatives_list)

    return text or None # storing empty strings causes issues when embedding

def preprocess_initiative_string(text):
    text = re.sub(r'\[.*?\]', '', text)  # remove enclosed text i.e. [este texto esta entre llaves]
    text = text.strip()

    return text
