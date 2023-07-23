from dateutil.parser import parse

from sivico.params import *
from sivico.text_input_and_summarization.data import get_senator_initiative_data, load_data_to_bq

import torch
import nltk
from transformers import BertTokenizer, BertModel
from nltk.tokenize import sent_tokenize
from googletrans import Translator
from deep_translator import GoogleTranslator
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def summarize_bert():
    senators = get_data_from_bq(pre-processed_senators)

    print("✅ senator data ready to translate and summarize. \n")

    #Split the concatenated initiatives for each senator into bites smaller than 5000 to be able to translate.
    senators["initiatives_summary_dummy_split"] = ""
    n = 4999
    for i, row in senators.iterrows():
        if not row["initiative_list"] == []:
            initiatives_split = [row["initiatives_summary_dummy"][i:i+n] for i in range(0, len(row["initiatives_summary_dummy"]), n)]
            senators.at[i, "initiatives_summary_dummy_split"] = initiatives_split
        else:
            senators.at[i, "initiatives_summary_dummy_split"] = []

    #Translate the concatenated, split string of initiatives per senator, store in new column.

    senators["initiatives_summary_dummy_split_en"] = ""

    for i, row in senators.iterrows():
        print(f"Working on row {i} of {len(senators)}")
        if len(row["initiatives_summary_dummy_split"]) >= 1:
            en_initiatives = GoogleTranslator(source='es', target='en').translate_batch(row["initiatives_summary_dummy_split"])
            senators.at[i, "initiatives_summary_dummy_split_en"] = en_initiatives
            print(f"Summary {i} translated successfully. Sample: {en_initiatives[0][:20]}")
        else:
            print(f"Senator number {i}, senator {row['senadores']} has no initiatives to translate.")
            continue

    #Join all split, translated initiatives into one long english string per senator.
    senators["initiatives_summary_dummy_en"] = senators["initiatives_summary_dummy_split_en"].apply(lambda x: "".join(x))

    #Import BERT model
    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    #Create summaries in english, spliting into batches of 150 sentences at a time.
    senators["initiative_summary_en"] = ""
    for index, row in senators.iterrows():
        if len(row["initiatives_summary_dummy_en"]) >= 1:
            #split text into sentences
            sentences = sent_tokenize(row["initiatives_summary_dummy_en"])
            #split sentences into batches
            n = 150
            sentences_split = [sentences[i:i+n] for i in range(0, len(sentences), n)]
            summary = []

            #translate by batch, then rejoin and add to df
            for sen_split in sentences_split:
                #tokenize sentences
                tokenized_sentences = [tokenizer.encode(sent, add_special_tokens=True) for sent in sentences]

                #define max length
                max_len = 0
                for i in tokenized_sentences:
                    if len(i) > max_len:
                        max_len = len(i)

                #padding sentences
                padded_sentences = []
                for i in tokenized_sentences:
                    while len(i) < max_len:
                        i.append(0)
                    padded_sentences.append(i)

                #creating tensors from padded sentences to feed into model
                input_ids = torch.tensor(padded_sentences)

                #feed tensors into model
                with torch.no_grad():
                    last_hidden_states = model(input_ids)[0]

                #Create sentence embedding, returning to np array from tensor format
                sentence_embeddings = []
                for i in range(len(sen_split)):
                    sentence_embeddings.append(torch.mean(last_hidden_states[i], dim=0).numpy())

                # Compute the similarity matrix
                similarity_matrix = cosine_similarity(sentence_embeddings)

                # Generate the summary through sentence scores and setting the summary size for most similar sentences.
                num_sentences = round(len(sen_split)*.25)
                summary_sentences = []
                for i in range(num_sentences):
                    sentence_scores = list(enumerate(similarity_matrix[i]))
                sentence_scores = sorted(sentence_scores, key=lambda x: x[1], reverse=True)
                for i in range(num_sentences):
                    summary_sentences.append(sen_split[sentence_scores[i][0]])
                sub_summary = ' '.join(summary_sentences)
                summary.append(sub_summary)

            summary = " ".join([str(item) for item in summary])

            senators.at[index, "initiative_summary_en"] = summary
        else:
            continue

    #Split the english summaries for each senator into bites smaller than 5000 to be able to translate back to spanish.
    senators["initiatives_summary_en_split"] = ""

    n = 4999

    for i, row in senators.iterrows():
        if not row["initiative_list"] == []:
            summary_split_es = [row["initiative_summary_en"][i:i+n] for i in range(0, len(row["initiative_summary_en"]), n)]
            senators.at[i, "initiatives_summary_en_split"] = summary_split_es
        else:
            senators.at[i, "initiatives_summary_en_split"] = []

    #Translate the split english summaires back to spanish, split string of initiatives per senator, store in new column.

    senators["initiatives_summary_es_split"] = ""

    for i, row in senators.iterrows():
        print(f"Working on row {i} of {len(senators)}")
        if len(row["initiatives_summary_dummy_split"]) >= 1:
            es_initiatives = GoogleTranslator(source='en', target='es').translate_batch(row["initiatives_summary_en_split"])
            senators.at[i, "initiatives_summary_es_split"] = es_initiatives
            print(f"Summary {i} translated successfully. Sample: {es_initiatives[0][:20]}")
        else:
            print(f"Senator number {i}, senator {row['senadores']} has no initiatives to translate.")
            continue

    #Merge translated batches into one
    senators["initiative_summary_es"] = senators["initiatives_summary_es_split"].apply(lambda x: "".join(x))
    
    #clean data to ensure successful load to big query
    senators = senators.fillna("").drop(["Unnamed: 0.1", "Unnamed: 0", "Unnamed: 0.2"], axis=1)

    #load processed senator data to big query
    load_data_to_bq(
        senators,
        gcp_project=GCP_PROJECT,
        bq_dataset=BQ_DATASET,
        table=f'summarized_senators',
        truncate=True
    )