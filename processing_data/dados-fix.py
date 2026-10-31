# public dados preparation script
# source column names and input filename are generic

import os
import pandas as pd

input_file = 'DATA/dados/input.csv'
output_file = 'DATA/dados/selected.csv'
dados = pd.read_csv(input_file, sep=';')

# convert the required source values
for col in ['moderate_minutes', 'vigorous_minutes', 'valid_days', 'vo2_measure', 'sleep_total', 'participant_code']:
    if col in dados.columns and dados[col].dtype == object:
        dados[col] = dados[col].str.replace(',', '.', regex=False)
    if col != 'participant_code':
        dados[col] = pd.to_numeric(dados[col], errors='coerce')

# prepare ids and calculate additional variables
dados = dados.rename(columns={'participant_code': 'id'})
dados['id'] = 'DADOS-' + dados['id'].astype(int).astype(str)
dados['moderate_act'] = (dados['moderate_minutes'] / dados['valid_days'] / 60).round(2)
dados['vigorous_act'] = (dados['vigorous_minutes'] / dados['valid_days'] / 60).round(2)
dados['vo2_max'] = dados['vo2_measure'].round(2)
dados['sleep'] = (dados['sleep_total'] / dados['valid_days']).round(2)
selected = dados[['id', 'moderate_act', 'vigorous_act', 'vo2_max', 'sleep']]
os.makedirs(os.path.dirname(output_file), exist_ok=True)
selected.to_csv(output_file, index=False)
print('Saved selected variables to', output_file)
