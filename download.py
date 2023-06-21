"""
Functions to collect data.
"""

import pandas as pd
import numpy as np
import os

from pysus.online_data import SIM, CNES, IBGE, parquets_to_dataframe

import util

download_states = util.available_states
download_years = util.available_years

def get_SIM(states: list = download_states, years: list = download_years) -> pd.DataFrame:
    """
    Get preprocessed SIM data.
    """
    df_SIM: pd.DataFrame()
    try:
        df_SIM = pd.read_csv('./data/preprocessed_SIM.csv')
        print("get_SIM: preprocessed data found in cache.")
    except FileNotFoundError:
        print("get_SIM: preprocessed data not found in cache, working on it...")
        df_SIM_raw = get_db_raw('SIM', states=states, years=years)
        SIM_selection = ['DTOBITO', 'HORAOBITO', 'CAUSABAS', 'LOCOCOR', 'CODMUNRES', 'IDADE', 'SEXO', 'RACACOR', 'ESC', 'ESTCIV']
        df_SIM = df_SIM_raw[SIM_selection]
        df_SIM = df_SIM.rename(columns={'CODMUNRES': 'CODMUN'})
        # Fix: remove white spaces from data
        for col in df_SIM:
            df_SIM[col] = df_SIM[col].astype(str).apply(str.strip)
        # Select only suicide deaths
        df_SIM['CAUSABAS'] = df_SIM['CAUSABAS'].str[:3]
        df_SIM = df_SIM.loc[df_SIM['CAUSABAS'].isin(util.dict_methods.keys())]
        # Decode features
        df_SIM['IDADE'] = df_SIM['IDADE'].apply(util.decode_age)
        df_SIM['DTOBITO'] = df_SIM['DTOBITO'].apply(util.decode_date)
        translate_SIM(df_SIM)
        df_SIM['CODMUN'] = df_SIM['CODMUN'].astype(int)

        ### Feature Extraction ###
        # 'DTOBITO' -> 'ano_obito', 'dia_obito', 'mes_obito', 'fim_semana', 'feriado', 'estacao_ano'
        df_SIM['year'] = df_SIM['DTOBITO'].apply(util.get_year)
        df_SIM['month'] = df_SIM['DTOBITO'].apply(util.get_month)
        df_SIM['day'] = df_SIM['DTOBITO'].apply(util.get_day)
        df_SIM['season'] = df_SIM['DTOBITO'].apply(util.get_season)
        df_SIM['weekday'] = df_SIM['DTOBITO'].apply(util.get_weekday)
        df_SIM['holiday'] = df_SIM['DTOBITO'].apply(util.is_holiday, args=[1])
        # Get municipality data and calculate suicide rates
        df_muni = get_municipality()
        suicide_rates = []
        drop_list = []
        for year in years:
            # Get number of deaths per municipality and year
            suicides_year = df_SIM.loc[df_SIM['year'] == year]
            suicides_year = suicides_year['CODMUN'].value_counts().reset_index()
            suicides_year.columns = ['CODMUN', f'suicides_{year}']
            df_muni = df_muni.merge(suicides_year, how='left', on='CODMUN')
            df_muni[f'suicides_{year}'] = df_muni[f'suicides_{year}'].fillna(0)
            df_muni[f'suicide_rate_{year}'] = df_muni[f'suicides_{year}'] / df_muni['pop_muni'].astype(int) * 100000
            df_muni = df_muni.drop(columns=f'suicides_{year}')
            drop_list.append(f'suicide_rate_{year}')
            suicide_rates.append(f'suicide_rate_{year}')
        df_muni['average_suicide_rate'] = df_muni[suicide_rates].mean(axis=1)
        df_muni = df_muni.drop(columns=drop_list)
        # Merge with municipality dataframe on municipality code
        # 'CODMUN' -> 'state', 'name_muni', 'pop_muni', 'average_suicide_rate', 'facility_rate'
        df_SIM['state'] = df_SIM['CODMUN'].apply(util.get_state)
        df_SIM = df_SIM.merge(df_muni, how='left', on='CODMUN')
        df_SIM = df_SIM.drop(columns='num_facilities')
        # 'IDADE' -> 'age_group'
        grupos = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        df_SIM['age_group'] = pd.cut(x=df_SIM['IDADE'], bins=grupos)
        # 'CAUSABAS' -> 'method'
        df_SIM['method'] = df_SIM['CAUSABAS'].apply(util.get_suicide_method)
        # 'HORAOBITO' -> 'periodo_dia'
        df_SIM['day_period'] = df_SIM['HORAOBITO'].apply(util.get_period)
        # Fix dtypes and save as .csv
        df_SIM = util.fix_numerical_dtypes(df_SIM)
        df_SIM.to_csv('./data/preprocessed_SIM.csv', index=False)
    return df_SIM

def get_CNES(states: list = download_states, years: list = download_years) -> pd.DataFrame:
    """
    Transforming CNES
    """
    df_CNES = pd.DataFrame()
    try:
        df_CNES = pd.read_csv('./data/preprocessed_CNES.csv')
        print("get_CNES: preprocessed data found in cache.")
    except:
        print("get_CNES: preprocessed data not found in cache, working on it...")
        df_CNES_raw = get_db_raw('CNES', states=states, years=years)
        CNES_selection = ['CNES', 'COMPETEN', 'CODUFMUN', 'COD_CEP', 'NATUREZA', 'VINC_SUS', 'TP_UNID', 'SERAP02P', 'SERAP02T']
        df_CNES = df_CNES_raw[CNES_selection]
        # Select only facilities that provide psychotherapy support (SADT) or Social Service
        df_CNES = df_CNES.loc[(df_CNES['TP_UNID'] == '39') | (df_CNES['SERAP02P'] == '1') | (df_CNES['SERAP02T'] == '1')]
        # Extracting feature 'year' from 'COMPETEN'
        df_CNES['year'] = df_CNES['COMPETEN'].str[:4] 
        # Make it readable
        df_CNES['NATUREZA'] = df_CNES['NATUREZA'].astype('object').map({
                                '01': 'Publica', # MS
                                '02': 'Outra', # Outros órgãos
                                '03': 'Publica', # Autarquia
                                '04': 'Publica', # Fundação pública
                                '05': 'Publica', # Empresa pública
                                '06': 'Publica', # Organização pública
                                '07': 'Privada', # Empresa privada
                                '08': 'Privada', # Fundação privada
                                '09': 'Outra', # Cooperativa
                                '10': 'Privada', # Serviço Social autônomo
                                '11': 'Publica', # Entidade beneficente
                                '12': 'Outra', # Economia mista
                                '13': 'Outra' # Sindicato
                            })
        # Fix dtypes and save as .csv
        df_CNES = util.fix_numerical_dtypes(df_CNES)
        df_CNES.to_csv('./data/preprocessed_CNES.csv', index=False)
    return df_CNES

def get_municipality() -> pd.DataFrame:
    """
    Get preprocessed municipality data.
    """
    df_muni: pd.DataFrame()
    try:
        df_muni = pd.read_csv('./data/municipality_data.csv')
        print("get_municipality: preprocessed data found in cache.")
    except FileNotFoundError:
        print("get_municipality: preprocessed data not found in cache, working on it...")
        df_muni_raw = get_municipality_raw()
        # Select municipality code, name and population
        df_muni = df_muni_raw[['D1C', 'D1N', 'V']].rename(columns={'D1C': 'CODMUN', 'D1N': 'name_muni', 'V': 'pop_muni'})
        df_muni['CODMUN'] = df_muni['CODMUN'].astype(str).str[:-1].astype(int) # Remove verification digit
        df_muni['name_muni'] = df_muni['name_muni'].str.rsplit(' ', 2).str[0] # Remove state from municipality name
        # Get number of healthcare facilities from CNES
        df_CNES = get_CNES()
        facilities_muni = df_CNES['CODUFMUN'].value_counts().reset_index().astype(int)
        facilities_muni.columns = ['CODMUN', 'num_facilities']
        df_muni = df_muni.merge(facilities_muni, how='left', on='CODMUN')
        df_muni['num_facilities'] = df_muni['num_facilities'].fillna(0)
        df_muni['facility_rate'] = df_muni['num_facilities'] / df_muni['pop_muni'] * 1000
        #df_muni['mental_healthcare'] = df_muni['facility_rate'].apply(util.healthcare)
        # Fix dtypes and save as .csv
        df_muni = util.fix_numerical_dtypes(df_muni)
        df_muni.to_csv('./data/municipality_data.csv', index=False)
    return df_muni

def get_db_raw(database: str, states: list = download_states, years: list = download_years) -> pd.DataFrame:
    """
    Read cached file containing the database. If no file is found, download it.
    """
    raw_df = pd.DataFrame()
    try:
        raw_df = pd.read_parquet(f'./data/rawdata/{database}.parquet')
        print(f"get_db_raw: raw {database} data found in cache.")
    except FileNotFoundError:
        print(f"get_db_raw: raw {database} data not found in cache, downloading from DATASUS...")
        raw_df = download_db(database, states, years)
    return raw_df

def get_municipality_raw() -> pd.DataFrame:
    """
    Download municipality data from IBGE (code, name, population).
    """
    df_muni = pd.DataFrame()
    try:
        df_muni = pd.read_csv('./data/rawdata/municipality_raw.csv')
        print("get_municipality_raw: raw municipality data found in cache.")
    except FileNotFoundError:
        print("get_municipality_raw: raw municipality data not found in cache, downloading from IBGE...")
        df_muni = IBGE.get_sidra_table(table_id=1505, territorial_level=6, variables=93, 
                                      classification=12017, categories=0, headers='n')
        # Save as .csv
        df_muni.to_csv('./data/rawdata/municipality_raw.csv', index=False)
    return df_muni

def download_db(database: str, states: list = download_states, years: list = download_years) -> pd.DataFrame:
    """
    Download parquets with PySUS and concatenate them all into a single dataframe.
    """
    parquets = []
    match database:
        case "SIM": parquets = SIM.download(states=states, years=years)
        case "CNES": parquets = CNES.download(group='ST', states=states, years=years, months=1)
        case _: raise ValueError("download.get_database: available databases are SIM and CNES\n")
    print(f"download_db: {database} parquet files downloaded.")
    df_list = []
    for path in parquets:
        df_list.append(parquets_to_dataframe(path))
    raw_df = pd.concat(df_list, ignore_index=True)
    raw_df.to_parquet(f'./data/rawdata/{database}.parquet')
    print(f"download_db: {database} data concatenated and stored.")
    return raw_df

def parquets_to_df(data_dir: str) -> pd.DataFrame:
    """
    Read all parquet files inside a directory into one pandas dataframe.
    """
    df_list = []
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        # If item is a parquet file, read and append to list
        if os.path.isfile(item_path) and item.endswith(".parquet"):
            df = pd.read_parquet(item_path)
            df_list.append(df)
        # If item is a directory, look for parquet files inside it
        elif os.path.isdir(item_path):
            df = parquets_to_df(item_path)
            df_list.append(df)
    concatenated_df = pd.concat(df_list, ignore_index=True)
    return concatenated_df


def get_ICD() -> pd.DataFrame:
    """
    Get ICD10 table, filter for suicide codes and rename key ('CID10') to merge with SIM.
    This was meant to get code descriptions but they're written in a very weird format, with all words abbreviated.
    """
    df_CID = SIM.get_CID10_table()
    df_CID = df_CID.loc[df_CID['CID10'].isin(util.icd_suicide_codes)]
    df_CID = df_CID[['CID10', 'DESCR']].rename(columns={'CID10': 'CAUSABAS', 'DESCR': 'method'})
    return df_CID

def get_CBO() -> pd.DataFrame:
    df_CBO = SIM.get_ocupations()
    df_CBO = df_CBO.rename(columns={'CODIGO': 'OCUP', 'DESCRICAO': 'occupation'})
    return df_CBO

def translate_SIM(df_SIM: pd.DataFrame) -> None:
    """
    Translate SIM attribute values to natural language.
    """
    df_SIM['LOCOCOR'] = df_SIM['LOCOCOR'].map({
                            '1': "Estabelecimento de saude",
                            '2': "Estabelecimento de saude",
                            '3': "Domicilio",
                            '4': "Via publica",
                            '5': "Outro"
                        })
    df_SIM['SEXO'] = df_SIM['SEXO'].map({
                            '1': "Masculino",
                            '2': "Feminino"
                        })
    df_SIM['RACACOR'] = df_SIM['RACACOR'].map({
                            '1': "Branca",
                            '2': "Preta",
                            '3': "Amarela",
                            '4': "Parda",
                            '5': "Indigena"
                        })
    # Note: 'ESC2010' is only available after 2011, hence rendered useless 
    df_SIM['ESC'] = df_SIM['ESC'].map({
                            '1': "Sem escolaridade",
                            '2': "Fundamental I",
                            '3': "Fundamental II",
                            '4': "Médio",
                            '5': "Superior"
                        })
    df_SIM['ESTCIV'] = df_SIM['ESTCIV'].map({
                            '1': "Solteiro",
                            '2': "Casado",
                            '3': "Viuvo",
                            '4': "Divorciado",
                            '5': "Uniao estavel"
                        })
    