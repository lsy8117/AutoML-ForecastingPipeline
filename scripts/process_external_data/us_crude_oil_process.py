import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

read_file_path = '/home/siyi/PycharmProjects/AutoML-ForecastingPipeline/data/external_data/raw/1._U.S._Energy_Markets_Summary.csv'
save_file_path = '/home/siyi/PycharmProjects/AutoML-ForecastingPipeline/data/external_data/processed/us_energy_market_summary_processed.csv'

energy_df = pd.read_csv(read_file_path)

df1 = energy_df.drop(columns=['Unnamed: 1','map','linechart','units','source key'], inplace=False)

df1 = df1.rename(columns={'remove':'item'}).set_index('item')
df2 = df1.T
df2.reset_index(inplace=True)

df2.rename(columns={'index':'month-year'}, inplace=True)
df2['year_month'] = pd.to_datetime(df2['month-year'], format='%b-%y')
df2.sort_values(by='year_month', inplace=True)

df2.replace('--', np.nan, inplace=True)

for col in df2.columns:
    if col != 'month-year' and col != 'year_month':
        df2[col] = df2[col].astype(float)
        df2[f'{col}_3_month_rolling_avg'] = df2[col].rolling(3).mean()
        df2[f'{col}_6_month_rolling_avg'] = df2[col].rolling(6).mean()
        df2[f'{col}_12_month_rolling_avg'] = df2[col].rolling(12).mean()
        df2[f'{col}_24_month_rolling_avg'] = df2[col].rolling(24).mean()

df2.to_csv(save_file_path, index=False)