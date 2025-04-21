import pandas as pd
import numpy as np

import warnings

warnings.filterwarnings('ignore')

read_file_path = '/home/siyi/PycharmProjects/AutoML-ForecastingPipeline/data/external_data/raw/US_total_energy.csv'
save_file_path = '/home/siyi/PycharmProjects/AutoML-ForecastingPipeline/data/external_data/processed/us_energy_data_processed.csv'


energy_df = pd.read_csv(read_file_path)

df1 = energy_df.pivot(index='YYYYMM',columns='Description',values='Value').reset_index()
df1.columns = df1.columns.str.replace(' ', '_')
df1.sort_values(by='YYYYMM',ascending=True,inplace=True)

df2 = df1[~df1['YYYYMM'].astype(str).str.endswith('13')]
df2.sort_values(by='YYYYMM',ascending=True,inplace=True)

for col in df2.columns:
    if col != 'YYYYMM':
        df2[f'{col}_3_month_rolling_avg'] = df2[col].rolling(3).mean()
        df2[f'{col}_6_month_rolling_avg'] = df2[col].rolling(6).mean()
        df2[f'{col}_12_month_rolling_avg'] = df2[col].rolling(12).mean()
        df2[f'{col}_24_month_rolling_avg'] = df2[col].rolling(24).mean()

df2.to_csv(save_file_path,index=False)