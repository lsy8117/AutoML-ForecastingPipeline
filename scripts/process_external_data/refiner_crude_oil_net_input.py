import pandas as pd
import numpy as np
import warnings


warnings.filterwarnings('ignore')

folder_path = '/home/siyi/PycharmProjects/AutoML-ForecastingPipeline/data/external_data/raw'
save_folder = '/home/siyi/PycharmProjects/AutoML-ForecastingPipeline/data/external_data/processed'

ru_file_path = f'{folder_path}/US_refinery_utilization.xlsx'
ni_file_path = f'{folder_path}/US_refiner_net_input_crude_oil.xlsx'
gip_file_path = f'{folder_path}/US_gross_input_refineries.xlsx'
odc_file_path = f'{folder_path}/US_operable_oil_distillation_capacity.xlsx'

save_file_path = f'{save_folder}/monthly_refinery_data_processed.csv'



################ Refinery Utilization ############
ru_df = pd.read_excel(ru_file_path)

ru_df['Date'] = pd.to_datetime(ru_df['Date'], format='%Y-%m-%d')
ru_df.columns = ru_df.columns.str.replace(' ', '_')

ru_df['year_month'] = ru_df['Date'].dt.to_period('M')
ru_monthly_avg = ru_df.groupby('year_month')['Weekly_U.S._Percent_Utilization_of_Refinery_Operable_Capacity_(Percent)'].mean().reset_index()

ru_monthly_avg = ru_monthly_avg.sort_values(by=['year_month'],ascending=True)

ru_monthly_avg.rename(columns={'Weekly_U.S._Percent_Utilization_of_Refinery_Operable_Capacity_(Percent)':'monthly_refinery_utilization_rate'}, inplace=True)

col = 'monthly_refinery_utilization_rate'
ru_monthly_avg[f"{col}_3_month_MA"] = ru_monthly_avg[col].rolling(window=3).mean()
ru_monthly_avg[f"{col}_6_month_MA"] = ru_monthly_avg[col].rolling(window=6).mean()
ru_monthly_avg[f"{col}_12_month_MA"] = ru_monthly_avg[col].rolling(window=12).mean()
ru_monthly_avg[f"{col}_24_month_MA"] = ru_monthly_avg[col].rolling(window=24).mean()

################ Net input data ############
ni_df = pd.read_excel(ni_file_path)

ni_df['Date'] = pd.to_datetime(ni_df['Date'], format='%Y-%m-%d')
ni_df.columns = ni_df.columns.str.replace(' ', '_')

ni_df['year_month'] = ni_df['Date'].dt.to_period('M')
ni_monthly_avg = ni_df.groupby('year_month')['Weekly_U.S._Refiner_Net_Input_of_Crude_Oil__(Thousand_Barrels_per_Day)'].mean().reset_index()

ni_monthly_avg.rename(columns={'Weekly_U.S._Refiner_Net_Input_of_Crude_Oil__(Thousand_Barrels_per_Day)':'monthly_refiner_crude_oil_net_input'}, inplace=True)

col = 'monthly_refiner_crude_oil_net_input'
ni_monthly_avg[f"{col}_3_month_MA"] = ni_monthly_avg[col].rolling(window=3).mean()
ni_monthly_avg[f"{col}_6_month_MA"] = ni_monthly_avg[col].rolling(window=6).mean()
ni_monthly_avg[f"{col}_12_month_MA"] = ni_monthly_avg[col].rolling(window=12).mean()
ni_monthly_avg[f"{col}_24_month_MA"] = ni_monthly_avg[col].rolling(window=24).mean()

############ Gross Input Refinery ##############
gip_df = pd.read_excel(gip_file_path)

gip_df['Date'] = pd.to_datetime(ni_df['Date'], format='%Y-%m-%d')
gip_df.columns = gip_df.columns.str.replace(' ', '_')

gip_df['year_month'] = gip_df['Date'].dt.to_period('M')
gip_monthly_avg = gip_df.groupby('year_month')['Weekly_U.S._Gross_Inputs_into_Refineries__(Thousand_Barrels_per_Day)'].mean().reset_index()

gip_monthly_avg.rename(columns={'Weekly_U.S._Gross_Inputs_into_Refineries__(Thousand_Barrels_per_Day)':'monthly_refiner_gross_input'}, inplace=True)

col = 'monthly_refiner_gross_input'
gip_monthly_avg[f"{col}_3_month_MA"] = gip_monthly_avg[col].rolling(window=3).mean()
gip_monthly_avg[f"{col}_6_month_MA"] = gip_monthly_avg[col].rolling(window=6).mean()
gip_monthly_avg[f"{col}_12_month_MA"] = gip_monthly_avg[col].rolling(window=12).mean()
gip_monthly_avg[f"{col}_24_month_MA"] = gip_monthly_avg[col].rolling(window=24).mean()


############ Operable Oil Distillation Capacity ##############
odc_df = pd.read_excel(odc_file_path)

odc_df['Date'] = pd.to_datetime(odc_df['Date'], format='%Y-%m-%d')
odc_df.columns = odc_df.columns.str.replace(' ', '_')

odc_df['year_month'] = odc_df['Date'].dt.to_period('M')
odc_monthly_avg = odc_df.groupby('year_month')['Weekly_U._S._Operable_Crude_Oil_Distillation_Capacity___(Thousand_Barrels_per_Calendar_Day)'].mean().reset_index()

odc_monthly_avg.rename(columns={'Weekly_U._S._Operable_Crude_Oil_Distillation_Capacity___(Thousand_Barrels_per_Calendar_Day)':'monthly_oil_distillation_capacity'}, inplace=True)

col = 'monthly_oil_distillation_capacity'
odc_monthly_avg[f"{col}_3_month_MA"] = odc_monthly_avg[col].rolling(window=3).mean()
odc_monthly_avg[f"{col}_6_month_MA"] = odc_monthly_avg[col].rolling(window=6).mean()
odc_monthly_avg[f"{col}_12_month_MA"] = odc_monthly_avg[col].rolling(window=12).mean()
odc_monthly_avg[f"{col}_24_month_MA"] = odc_monthly_avg[col].rolling(window=24).mean()

#######################################################################################
############ Combine Data ##############
df_merge = pd.merge(ru_monthly_avg, ni_monthly_avg, on='year_month', how='right')
df_merge = df_merge.merge(gip_monthly_avg, on='year_month', how='left')
df_merge = df_merge.merge(odc_monthly_avg, on='year_month', how='left')

df_merge.to_csv(save_file_path, index=False)