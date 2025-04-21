import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

#### Change file name if needed ####
file_path = '/home/siyi/PycharmProjects/AutoML-ForecastingPipeline/data/external_data/raw/rigs_count.xlsx'
save_file_path = '/home/siyi/PycharmProjects/AutoML-ForecastingPipeline/data/external_data/processed/rigs_count.csv'
####################################


wekkly_df = pd.read_excel(file_path, sheet_name='weekly')
col_list = list(range(1970,2017))

melted_df = pd.melt(
    wekkly_df,
    id_vars=['Weeks'],
    value_vars=col_list,
    var_name='Year',
    value_name='Rigs_Count'
)

melted_df.head()
#%%
melted_df['Date'] = pd.to_datetime(melted_df['Year'].astype(str) + melted_df['Weeks'].astype(int).astype(str)+'1', format='%Y%W%w')

melted_df['year_month'] = melted_df['Date'].dt.to_period('M')

melted_df = melted_df.replace(" ", np.nan)

# Drop rows where column 'B' has NaN
melted_df = melted_df.dropna(subset=['Rigs_Count']).reset_index(drop=True)

melted_df['Rigs_Count'] = melted_df['Rigs_Count'].astype(float)
monthly_avg = melted_df.groupby(['year_month'])['Rigs_Count'].mean().reset_index()

monthly_avg = monthly_avg[monthly_avg['year_month'] < '2013-01']


monthly_df = pd.read_excel(file_path, sheet_name='monthly')

monthly_df['Date'] = pd.to_datetime(monthly_df['Year'].astype(str) + monthly_df['Month'].astype(int).astype(str), format='%Y%m')

monthly_df['year_month'] = monthly_df['Date'].dt.to_period('M')

monthly_df.drop(columns=['Year', 'Month'], inplace=True)

monthly_df.drop(columns=['Date'], inplace=True)

monthly_df.rename(columns={'Sum of Rig Count Value':'Rigs_Count'}, inplace=True)

concat_df = pd.concat([monthly_df, monthly_avg], ignore_index=True)

concat_df.sort_values(by=['year_month'], ascending=True, inplace=True)

concat_df[f'Rigs_Count_3_month_MA'] = concat_df['Rigs_Count'].rolling(window=3).mean()
concat_df[f'Rigs_Count_6_month_MA'] = concat_df['Rigs_Count'].rolling(window=6).mean()
concat_df[f'Rigs_Count_12_month_MA'] = concat_df['Rigs_Count'].rolling(window=12).mean()
concat_df[f'Rigs_Count_24_month_MA'] = concat_df['Rigs_Count'].rolling(window=24).mean()


concat_df.to_csv(save_file_path, index=False)