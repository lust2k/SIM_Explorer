"""
Web app: data dictionary.
"""

import streamlit as st
import pandas as pd

def data_dict():
    columns = ['Variable name', 'Type', 'Description']
    data_dict = [
        ('DTOBITO', 'datetime', "Date of death"),
        ('HORAOBITO', 'string', "Time of death"),
        ('CAUSABAS', 'string', "Cause of death codes as defined by ICD"),
        ('LOCOCOR', 'string', "Place of death"),
        ('CODMUN', 'int', "Municipality of residence codes as defined by IBGE"),
        ('IDADE', 'int', "Age"),
        ('SEXO', 'string', "Sex"),
        ('RACACOR', 'string', "Race/color as classified by IBGE"),
        ('ESC', 'string', "Highest level of education"),
        ('ESTCIV', 'string', "Civil status"),
        ('age_group', 'category', "Age group"),
        ('method', 'string', "Suicide method"),
        ('name_muni', 'string', "Municipality of residence's name"),
        ('pop_muni', 'int', "Municipality of residence's population"),
        ('num_facilities', 'int', "Number of healthcare facilities with mental health support in the municipality"),
        ('year', 'int', "Year of death"),
        ('month', 'int', "Month of death"),
        ('day', 'int', "Day of death"),
        ('season', 'string', "Season of the year of death"),
        ('weekday', 'string', "Weekday of death"),
        ('holiday', 'bool', "Death happened on a holiday?"),
        ('period', 'string', "Day period of death")
    ]
    df_dict = pd.DataFrame(data_dict, columns=columns)
    df_dict.to_csv('./data/data_dict.csv', index=False)
    return df_dict

st.write(
    """
    ### Data Dictionary
    <div style='text-align:justify;'>
    The original databases used in this project are provided by DATASUS (Unified Health System's IT Department).
    SIM (Mortality Information System) data includes cause of death as classified by ICD and the victims' demographics;
    CNES (National Registry of Healthcare Facilities) contains a wide variety of data on all healthcare facilities in the country, regardless of their legal nature or whether they integrate SUS.
    More information on these and other databases is available <a href="https://datasus.saude.gov.br/">here</a>.
    <br><br>
    After selecting and extracting features, cleaning, transforming and linking the data, the dataset contains the following attributes:
    </div>
    """, 
    unsafe_allow_html=True
)

data_dict = data_dict()

st.table(data_dict)