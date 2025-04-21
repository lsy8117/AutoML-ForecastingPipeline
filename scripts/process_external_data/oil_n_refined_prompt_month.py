import pandas as pd
import numpy as np
import warnings


warnings.filterwarnings('ignore')

folder_path = '/home/siyi/PycharmProjects/AutoML-ForecastingPipeline/data/external_data/raw'
save_folder = '/home/siyi/PycharmProjects/AutoML-ForecastingPipeline/data/external_data/processed'

file_path = f'{folder_path}/oil_refined_product_prompt_month.xlsx'
save_file_path = f'{save_folder}/oil_refined_product_prompt_month_processed.xlsx'

pm_df = pd.read_excel(file_path)

pm_df['Date'] = pd.to_datetime(pm_df['Date'], format='%Y-%m-%d')
pm_df = pm_df.sort_values(by=['Date'],ascending=True)

pm_df.columns = pm_df.columns.str.replace(' ', '_')

pm_data_cols = [item for item in pm_df.columns if item != 'Date']

for col in pm_data_cols:
    pm_df[f"{col} 60 Day MA"] = pm_df[col].rolling(window=60).mean()
    pm_df[f"{col} 90 Day MA"] = pm_df[col].rolling(window=90).mean()
    pm_df[f"{col} 360 Day MA"] = pm_df[col].rolling(window=360).mean()

pm_df.to_csv(save_file_path, index=False)