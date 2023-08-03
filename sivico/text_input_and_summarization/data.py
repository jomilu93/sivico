import pandas as pd
from google.cloud import bigquery
from colorama import Fore, Style
from pathlib import Path
import torch
import nltk
from transformers import BertTokenizer, BertModel
from nltk.tokenize import sent_tokenize
from deep_translator import GoogleTranslator
from sklearn.metrics.pairwise import cosine_similarity
import requests
import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
from lxml import etree
import re
from selenium import webdriver
from sivico.params import *

def get_senator_initiative_data():
    def get_senators():
        senators_url = 'https://www.senado.gob.mx/65/datosAbiertos/senadoresDatosAb.json'
        senators_json = requests.get(senators_url).json()
        senators = pd.DataFrame.from_dict(senators_json)
        senators = senators.rename(columns={"idSenador": "senator_id"})
        return senators

    senators = get_senators()

    #Creating a field that includes first and last names to join with initiatives+proposals table.
    senators["senadores"] = senators["Nombre"].str.strip()+" "+senators["Apellidos"].str.strip()

    def get_senator_attendance():
        senators = get_senators()
        senator_ids = senators["senator_id"].tolist()

        senator_attendance = pd.DataFrame()
        senator_attendance["senator_id"] = ""
        senator_attendance["session_date"] = ""
        senator_attendance["attendance_record"] = ""

        counter = 0
        for sen in senator_ids:
            url = f'https://www.senado.gob.mx/65/asistencias/{sen}#info'
            html = requests.get(url)
            content = BeautifulSoup(html.text, 'html.parser')
            content_x = etree.HTML(str(content))
            dates = content_x.xpath('//*[@id="imPage"]/div[7]/div[2]/div/div[2]/section/div/div/table/tbody//a')
            att_records = content_x.xpath('//*[@id="imPage"]/div[7]/div[2]/div/div[2]/section/div/div/table/tbody//strong')
            for i in range(len(dates)):
                senator_attendance.at[i+counter, 'senator_id'] = sen
                senator_attendance.at[i+counter, 'session_date'] = dates[i].text
                senator_attendance.at[i+counter, 'attendance_record'] = att_records[i].text
            counter += len(dates)

        senator_attendance["attendance_score"] = senator_attendance["attendance_record"].copy()
        senator_attendance["attendance_score"] = senator_attendance["attendance_score"].map(lambda x: 1 if x == "Asistencia" else 0)
        senator_attendance = pd.merge(senator_attendance, senators[['senator_id','Fraccion', 'Estado', 'Apellidos', 'Nombre', 'tipoEleccion']], on='senator_id', how='left')

        senator_attendance["full_name"] = senator_attendance['Nombre'] + " " + senator_attendance['Apellidos']

        senator_attendance = senator_attendance.groupby(['senator_id', 'full_name', 'Fraccion', 'Estado', 'tipoEleccion'], as_index=False)[['attendance_score']].mean()

        return senator_attendance

    senator_attendance = get_senator_attendance()

    #add senator attendance to original senator df
    senators = senators.merge(senator_attendance[["senator_id", "attendance_score"]], how="left", on="senator_id")

    def get_initiatives():
        """fucntion that extracts initiatives from Senate JSON."""

        init_64_url = 'https://www.senado.gob.mx/65/datosAbiertos/iniciativa_64.json'
        init_65_url = 'https://www.senado.gob.mx/65/datosAbiertos/iniciativa_65.json'

        init_64_json = requests.get(init_64_url).json()
        init_65_json = requests.get(init_65_url).json()

        init_64 = pd.DataFrame.from_dict(init_64_json)
        init_65 = pd.DataFrame.from_dict(init_65_json)

        initiatives = pd.concat([init_64, init_65])

        initiatives['fecha_presentacion'] = pd.to_datetime(initiatives['fecha_presentacion'],errors='coerce')
        initiatives['fecha_aprobacion'] = pd.to_datetime(initiatives['fecha_aprobacion'],errors='coerce')

        initiatives = initiatives.set_index('id')

        return initiatives

    def get_proposals():
        """function that extracts proposals from Senate JSON."""

        prop_64_url = 'https://www.senado.gob.mx/65/datosAbiertos/proposicion_64.json'
        prop_65_url = 'https://www.senado.gob.mx/65/datosAbiertos/proposicion_65.json'

        prop_64_json = requests.get(prop_64_url).json()
        prop_65_json = requests.get(prop_65_url).json()

        prop_64 = pd.DataFrame.from_dict(prop_64_json)
        prop_65 = pd.DataFrame.from_dict(prop_65_json)

        proposals = pd.concat([prop_64, prop_65])

        proposals['fecha_presentacion'] = pd.to_datetime(proposals['fecha_presentacion'],errors='coerce')
        proposals['fecha_aprobacion'] = pd.to_datetime(proposals['fecha_aprobacion'],errors='coerce')

        proposals = proposals.set_index('id')

        return proposals

    #Create concatenated df that includes initiatives and proposals.
    initiatives = get_initiatives()
    proposals = get_proposals()
    inipros = pd.concat([initiatives, proposals])

    #creates a 1:1 relationship between initiative/proposal and senator (in case where more than 1 senator proposes).
    inipros["senadores"] = inipros["senadores"].apply(lambda x:x.strip().split("<br>"))

    for i, row in inipros.iterrows():
        senator_ids = []
        for senator in row["senadores"]:
            strt_pos = senator.find('(')
            senator = senator[:strt_pos-1].strip()
            senator_ids.append(senator)
        inipros.at[i, "senadores"] = senator_ids[:-1]

    inipros = inipros.explode("senadores")

    #Manually change names in inipros so they match senator names from senator table.

    inipros.loc[inipros["senadores"] == "Geovanna del Carmen Bañuelos de La Torre", "senadores"] = "Geovanna Bañuelos"
    inipros.loc[inipros["senadores"] == "Noé Fernando Castañón Ramírez", "senadores"] = "Noé Castañón"
    inipros.loc[inipros["senadores"] == "José Clemente Castañeda Hoeflich", "senadores"] = "Clemente Castañeda Hoeflich"
    inipros.loc[inipros["senadores"] == "Juan Manuel Zepeda Hernández", "senadores"] = "Juan Zepeda"
    inipros.loc[inipros["senadores"] == "Patricia Mercado Castro", "senadores"] = "Patricia Mercado"
    inipros.loc[inipros["senadores"] == "Dante Delgado Rannauro", "senadores"] = "Dante Delgado"
    inipros.loc[inipros["senadores"] == "Bertha Xóchitl Gálvez Ruiz", "senadores"] = "Xóchitl Gálvez Ruiz"
    inipros.loc[inipros["senadores"] == "Lilly Téllez García", "senadores"] = "Lilly Téllez"
    inipros.loc[inipros["senadores"] == "Raúl Bolaños Cacho Cué", "senadores"] = "Raúl Bolaños-Cacho Cué"
    inipros.loc[inipros["senadores"] == "Elvia Marcela Mora Arellano", "senadores"] = "Marcela Mora"
    inipros.loc[inipros["senadores"] == "Minerva Citlalli Hernández Mora", "senadores"] = "M. Citlalli Hernández Mora"

    #Inner join on senator names to ensure only initiatives that match senator ids from table remain.
    inipros = inipros.merge(senators[["senadores", "senator_id"]], how='inner', on='senadores')

    #Create a clean list of commissions to which each initiative belongs

    inipros["comisiones"] = inipros["comisiones"].apply(lambda x:x.split("<br>")[1].strip()
                                                        if "Puntos Constitucionales" in x and not x.split("<br>")[1] == ""
                                                        else x.split("<br>")[0].strip())

    inipros["comisiones"] = inipros["comisiones"].apply(lambda x:x.split(", ") if "Comisión" in x else x)

    inipros = inipros.explode("comisiones")

    inipros["comisiones"] = inipros["comisiones"].apply(lambda x:x.replace("(Coordinadora)", ""))
    inipros["comisiones"] = inipros["comisiones"].apply(lambda x:x.replace("Tercera Comisión:", ""))
    inipros["comisiones"] = inipros["comisiones"].apply(lambda x:x.replace("Segunda Comisión:", ""))
    inipros["comisiones"] = inipros["comisiones"].apply(lambda x:x.replace("Primera Comisión:", "").strip())
    inipros["comisiones"] = inipros["comisiones"].apply(lambda x:x.replace("(Comisiones Unidas)", "").strip())

    inipros["comisiones"] = inipros["comisiones"].apply(lambda x:x.split(" y ") if "Defensa Nacional y Educación Pública" in x else x)

    inipros = inipros.explode("comisiones")

    inipros["comisiones"] = inipros["comisiones"].apply(lambda x:"Relaciones Exteriores".strip() if "Relaciones Exteriores" in x else x)
    inipros["comisiones"] = inipros["comisiones"].apply(lambda x:"Educación".strip() if "Educación" in x else x)
    inipros["comisiones"] = inipros["comisiones"].apply(lambda x:"Justicia".strip() if "Justicia" in x else x)
    inipros["comisiones"] = inipros["comisiones"].apply(lambda x:"Seguridad Social".strip() if "Social" in x else x)
    inipros["comisiones"] = inipros["comisiones"].apply(lambda x:"Agricultura, Ganadería, Pesca y Desarrollo Rural".strip() if "Agr" in x else x)
    inipros["comisiones"] = inipros["comisiones"].apply(lambda x:"Energía".strip() if "Recursos Hidráulicos" in x else x)
    inipros["comisiones"] = inipros["comisiones"].apply(lambda x:"Estudios Legislativos".strip() if "Legislativos" in x else x)

    inipros["comisiones"] = inipros["comisiones"].apply(lambda x:x.replace(" ", "_").strip())
    inipros["comisiones"] = inipros["comisiones"].apply(lambda x:x.replace(",", "_").strip())

    #Return overall initiative list back to senator df
    senators["initiative_list"] = ""

    #Function that creates a list of initiative syntheses and then adds to senator database.
    for i, row in senators.iterrows():
        initiatives = []
        relevant_inipros = inipros[inipros["senator_id"] == str(row["senator_id"])]["sintesis"]
        [initiatives.append(initiative.replace('\r\n\r\n', ' ')) for initiative in relevant_inipros]
        senators.at[i, "initiative_list"] = list(set(initiatives))

    #Creates dummy summary of a all initiatives, to be replaced by BERT or BETO summaries.
    senators["initiatives_summary_dummy"] = senators["initiative_list"].apply(lambda x: "".join(x))

    #Create column for each commission
    commissions = inipros["comisiones"].unique()
    for commission in commissions:
        senators[f"{commission}_initiative_list"] = ""

    #Fill columns with relevant initiatves per sentator and commission
    for i, row in senators.iterrows():
        for commission in commissions:
            initiatives = []
            relevant_inipros = inipros[(inipros["senator_id"] == str(row["senator_id"])) & (inipros["comisiones"]==str(commission))]["sintesis"]
            [initiatives.append(initiative.replace('\r\n\r\n', ' ')) for initiative in relevant_inipros]
            senators.at[i, f"{commission}_initiative_list"] = list(set(initiatives))

    print("✅ get_senator_initiative_data_done \n")

    #load processed senator data to big query
    load_data_to_bq(
        senators,
        gcp_project=GCP_PROJECT,
        bq_dataset=BQ_DATASET,
        table=f'pre_processed_senators',
        truncate=True
    )

    return senators

def get_data_from_bq(table_name):

    query = f"""
        SELECT *
        FROM {GCP_PROJECT}.{BQ_DATASET}.{table_name}
    """

    gcp_project = os.environ.get("GCP_PROJECT")

    print(Fore.BLUE + "\nLoad data from BigQuery server..." + Style.RESET_ALL)
    client = bigquery.Client(project=gcp_project)
    query_job = client.query(query)
    result = query_job.result()
    df = result.to_dataframe()

    print(f"✅ Data loaded, with shape {df.shape}")

    return df

def load_data_to_bq(
        data: pd.DataFrame,
        gcp_project:str,
        bq_dataset:str,
        table: str,
        truncate: bool
    ):
    """
    - Save the DataFrame to BigQuery
    - Empty the table beforehand if `truncate` is True, append otherwise
    """

    assert isinstance(data, pd.DataFrame)
    full_table_name = f"{gcp_project}.{bq_dataset}.{table}"
    print(Fore.BLUE + f"\nSave data to BigQuery @ {full_table_name}...:" + Style.RESET_ALL)

    # Load data onto full_table_name

    # 🎯 HINT for "*** TypeError: expected bytes, int found":
    # After preprocessing the data, your original column names are gone (print it to check),
    # so ensure that your column names are *strings* that start with either
    # a *letter* or an *underscore*, as BQ does not accept anything else

    # $CHA_BEGIN
    # TODO: simplify this solution if possible, but students may very well choose another way to do it
    # We don't test directly against their own BQ tables, but only the result of their query
    data.columns = [f"_{column}" if not str(column)[0].isalpha() and not str(column)[0] == "_" else str(column) for column in data.columns]

    client = bigquery.Client()

    # Define write mode and schema
    write_mode = "WRITE_TRUNCATE" if truncate else "WRITE_APPEND"
    job_config = bigquery.LoadJobConfig(write_disposition=write_mode)

    print(f"\n{'Write' if truncate else 'Append'} {full_table_name} ({data.shape[0]} rows)")

    # Load data
    job = client.load_table_from_dataframe(data, full_table_name, job_config=job_config)
    result = job.result()  # wait for the job to complete

    print(f"✅ Data saved to bigquery, with shape {data.shape}")
