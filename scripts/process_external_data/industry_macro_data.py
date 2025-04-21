import pandas as pd
import numpy as np
import warnings


warnings.filterwarnings('ignore')


#### Change raw file name and final output name ####
'''
This file could be used to process some data files downloaded from EIA website.
Data files that has been processed by this script are:
    PET_PRI_SPT_S1_D.xls
    NG_PRI_FUT_S1_D.xls
    PET_SUM_SND_D_NUS_MBBL_M_cur.xls
    psw18vwall.xls
    pswrgvwall.xls

Other similar format files could also use this script to process.
Change the file names and excel sheet names before processing.
'''

folder_path = '/home/siyi/PycharmProjects/AutoML-ForecastingPipeline/data/external_data/raw/PET_PRI_SPT_S1_D.xls'

file_name = 'PET_PRI_SPT_S1_D.xls'
sheet_names = ['Data 1', 'Data 2', 'Data 3','Data 4', 'Data 5', 'Data 6', 'Data 7','Data 8']
save_folder_path = '/home/siyi/PycharmProjects/AutoML-ForecastingPipeline/data/external_data/processed'
save_file_name = 'monthly_spot_prices_processed.csv'
####################################################



file_path = folder_path + '/' + file_name
save_file_path = save_folder_path + '/' + save_file_name

df_list = []
for sheet in sheet_names:
    curr_df = pd.read_excel(file_path, sheet_name=sheet, skiprows=2)
    curr_df['Date'] = pd.to_datetime(curr_df['Date'], format='%Y-%m-%d')
    curr_df.columns = curr_df.columns.str.replace(' ', '_')

    curr_df['year_month'] = curr_df['Date'].dt.to_period('M')

    col_list = [item for item in curr_df.columns if item not in ['Date', 'year_month']]
    curr_monthly_avg = curr_df.groupby('year_month')[col_list].mean().reset_index()

    for col in col_list:
        curr_monthly_avg[f"{col}_3_month_MA"] = curr_monthly_avg[col].rolling(window=3).mean()
        curr_monthly_avg[f"{col}_6_month_MA"] = curr_monthly_avg[col].rolling(window=6).mean()
        curr_monthly_avg[f"{col}_12_month_MA"] = curr_monthly_avg[col].rolling(window=12).mean()
        curr_monthly_avg[f"{col}_24_month_MA"] = curr_monthly_avg[col].rolling(window=24).mean()
        curr_monthly_avg[f"{col}_lag_1_month"] = curr_monthly_avg[col].shift(1)
        curr_monthly_avg[f"{col}_lag_2_month"] = curr_monthly_avg[col].shift(2)
        curr_monthly_avg[f"{col}_lag_3_month"] = curr_monthly_avg[col].shift(3)
        curr_monthly_avg[f"{col}_lag_4_month"] = curr_monthly_avg[col].shift(4)
        curr_monthly_avg[f"{col}_lag_5_month"] = curr_monthly_avg[col].shift(5)
        curr_monthly_avg[f"{col}_lag_6_month"] = curr_monthly_avg[col].shift(6)
        curr_monthly_avg[f"{col}_lag_12_month"] = curr_monthly_avg[col].shift(12)
        curr_monthly_avg[f"{col}_lag_24_month"] = curr_monthly_avg[col].shift(24)


    df_list.append(curr_monthly_avg)

merge_column = 'year_month'  # Replace 'key' with the actual column you want to merge on

# Initialize with the first DataFrame
merged_df = df_list[0]

# Merge iteratively on the specified column(s)
for df in df_list[1:]:
    merged_df = pd.merge(merged_df, df, on=merge_column,
                         how='outer')  # You can change the 'how' parameter as per your needs


merged_df.to_csv(save_file_path, index=False)