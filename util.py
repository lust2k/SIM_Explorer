"""
Utility functions.
"""

import pandas as pd
import numpy as np

import holidays
import calendar
from datetime import date, timedelta
#from pysus.online_data import SIM
from pysus.preprocessing import decoders

available_states = ['PR', 'SC', 'RS']
available_years = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]

dict_states = {
    '11': 'RO', '12': 'AC', '13': 'AM', '14': 'RR', '15': 'PA', '16': 'AP', '17': 'TO',
    '21': 'MA', '22': 'PI', '23': 'CE', '24': 'RN', '25': 'PB', '26': 'PE', '27': 'AL', '28': 'SE', '29': 'BA',
    '31': 'MG', '32': 'ES', '33': 'RJ', '34': 'SP',
    '41': 'PR', '42': 'SC', '43': 'RS',
    '50': 'MS', '51': 'MT', '52': 'GO', '53': 'DF'
}

# Suicide codes (ICD10)
#    X60 to X69 -> self-poisoning
#    X70 to X84 -> self-inflicted injury
# Grouping causes and rewriting descriptions...
dict_methods = {
    'X60' : "Drogas ou medicamentos", # Analgesicos nao-opiaceos
    'X61' : "Drogas ou medicamentos", # Anticonvulsionantes e psicotropicos
    'X62' : "Drogas ou medicamentos", # Narcoticos e psicodislepticos
    'X63' : "Drogas ou medicamentos", # Outras substancias farmacologicas de acao sobre o sist. nervoso
    'X64' : "Drogas ou medicamentos", # Outras drogas e substancias nao especificadas
    'X65' : "Drogas ou medicamentos", # Alcool
    'X66' : "Outras substancias", # Solventes organicos e seus vapores
    'X67' : "Outras substancias", # Outros gases e vapores
    'X68' : "Outras substancias", # Pesticidas
    'X69' : "Outras substancias", # Substancias nao identificadas
    'X70' : "Estrangulamento",
    'X71' : "Afogamento",
    'X72' : "Arma de fogo", # Arma de fogo de mão (pistolas e revólveres)
    'X73' : "Arma de fogo", # Arma de fogo de calibre maior (espingardas, carabinas)
    'X74' : "Arma de fogo", # Arma de fogo nao especificada
    'X75' : "Outros meios", # Dispositivos explosivos
    'X76' : "Fogo, fumaça, gases ou objetos quentes", # Fumaca e fogo
    'X77' : "Fogo, fumaça, gases ou objetos quentes", # Vapor de agua, gases ou objetos quentes 
    'X78' : "Arma branca", # Objeto cortante ou penetrante
    'X79' : "Arma branca", # Objeto contundente
    'X80' : "Impacto", # Precipitacao de lugar elevado
    'X81' : "Impacto", # Permanencia diante de objeto em movimento
    'X82' : "Impacto", # Impacto de um veículo motorizado
    'X83' : "Outros meios",
    'X84' : "Outros meios" # Nao especificado
}

# Seasons
Y = 2000 # Any leap year
seasons = [("Summer", (date(Y,  1,  1),  date(Y,  3, 20))),
           ("Autumn", (date(Y,  3, 21),  date(Y,  6, 20))),
           ("Winter", (date(Y,  6, 21),  date(Y,  9, 22))),
           ("Spring", (date(Y,  9, 23),  date(Y, 12, 20))),
           ("Summer", (date(Y, 12, 21),  date(Y, 12, 31)))]

br_holidays = holidays.country_holidays('BR', years=available_years)

def decode_age(age: str) -> int:
    return decoders.decodifica_idade_SIM(age, 'Y')

def decode_date(date_str: str) -> date:
    return decoders.decodifica_data_SIM(date_str)

def get_state(codmun: int | str) -> str:
    return dict_states.get(str(codmun)[:2])

def get_suicide_method(icd: str) -> str:
    return dict_methods.get(icd)
    
def get_season(data: date) -> str:
    data = data.replace(year=Y)
    for season, (start, end) in seasons:
        if start <= data <= end:
            return season

def get_period(hora: int) -> str:
    # Valores alfanuméricos ou nulos
    if hora.isnumeric() == False:
        return np.nan
    # Madrugada (00:00 - 05:59)
    elif 0 <= int(hora) < 600:
        return "Night"
    # Manhã (06:00 - 11:59)
    elif 600 <= int(hora) < 1200:
        return "Morning"
    # Tarde (12:00 - 17:59)
    elif 1200 <= int(hora) < 1800:
        return "Afternoon"
    # Noite (18:00 - 23:59)
    elif 1800 <= int(hora) < 2400:
        return "Evening"
    # Valores numéricos inválidos (ex. 9999)
    return np.nan

def get_weekday(data: date) -> str:
    weekday = calendar.weekday(data.year, data.month, data.day) # Segunda = 0, ...
    if weekday == 0:
        return "Monday"
    elif weekday == 1:
        return "Tuesday"
    elif weekday == 2:
        return "Wednesday"
    elif weekday == 3:
        return "Thursday"
    elif weekday == 4:
        return "Friday"
    elif weekday == 5:
        return "Saturday"
    elif weekday == 6:
        return "Sunday"
    return np.nan

def get_year(data: date) -> int:
    return data.year

def get_month(data: date) -> int:
    return data.month

def get_day(data: date) -> int:
    return data.day

def is_holiday(data: date, interval: int) -> bool:
    """
    Verify if date is a holiday or close to one.
    """
    delta = timedelta(days=interval)
    for holiday in br_holidays:
        if holiday-delta <= data <= holiday+delta:
            return True
    return False

def healthcare(facility_rate: float) -> str:
    if facility_rate == 0: return "None"
    elif facility_rate >= 5: return "High"
    elif facility_rate >= 2: return "Moderate"
    else: return "Low"

def impute_column(column: pd.Series) -> pd.Series:
    """
    Impute missing data in a column. Median for numerical features, mode for categorical.
    """
    missing_data = column.isna().sum()

    if missing_data / len(column) < 0.3: # 0.03
        # Impute numerical column with median
        if np.issubdtype(column.dtype, np.number):
            column = column.fillna(column.median())
        # Impute categorical column with mode
        else: # column.dtype == 'object'
            column = column.fillna(column.mode()[0])
        
    # TO DO: Impute columns with more than X% missing data using KNN?
    else:
        if np.issubdtype(column.dtype, np.number):
            pass
        else: # column.dtype == 'object'
            pass
    return column

def impute_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing data in all columns of a dataframe.
    """
    for column_name in df.columns:
        df[column_name] = impute_column(df[column_name])
    return df

def column_nulls(df: pd.DataFrame) -> list:
    """
    Get a list of the columns that have missing data in a dataframe.
    """
    columns = []
    for col in df.columns:
        missing_data = df[col].isna().sum()
        if missing_data > 0:
            columns.append(col)
            print(f"{col}: {missing_data}")
    return columns

def fix_numerical_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix numerical column dtypes.
    """
    numerical_features = df.select_dtypes(include=[np.number]).columns
    for feature in numerical_features:
        df[feature] = pd.to_numeric(df[feature], errors='coerce')
    return df
