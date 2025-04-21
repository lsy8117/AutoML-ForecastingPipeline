import numpy as np

from data_preprocessing import *
# from quarter_data_preprocess import *
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


raw_data_path = '/home/siyi/PycharmProjects/AutoML-ForecastingPipeline/data/bbg_data/sample_oil_firm_8_tables.csv'
final_data_path = '/home/siyi/PycharmProjects/AutoML-ForecastingPipeline/data/bbg_data/processed_data.csv'



raw_df = pd.read_csv(raw_data_path)
print(raw_df.head())

quarter_al_df = primary_quarterly_data_prep(raw_df)

## Adding industry information
industry_mappings = read_industry_code_mapping()
df_company_info, industry_cols = read_company_info()
df1 = merge_with_company_info(quarter_al_df, industry_cols, industry_mappings, df_company_info)

## Union process with custom aggregate
df2 = choose_quarter_data_for_comp(df1)

create_lag_list = []

# y_cols = ['SALES_REV_TURN','CF_CASH_FROM_OPER','EBITDA']
y_cols = ['SALES_REV_TURN','CF_CASH_FROM_OPER','EBITDA','ARD_CAPITAL_EXPENDITURES']
lagged_y = []

for y in y_cols:
    lagged_y.append(y+'_lag1')

for id, sub_df in df2.groupby('ID_BB_UNIQUE'):
    sub_df = sub_df.sort_values(['Year','Quarter'], ascending=True)
    for y in y_cols:
        sub_df[y + '_lag1'] = sub_df[y].shift(-1)
        # lagged_y.append(y + '_lag1')
        sub_df[y + '_last_year_same_quarter'] = sub_df[y].shift(4)
    sub_df['ANNOUNCEMENT_DT_lag1'] = sub_df['ANNOUNCEMENT_DT'].shift(-1)
    sub_df['LATEST_PERIOD_END_DT_FULL_RECORD_lag1'] = sub_df['LATEST_PERIOD_END_DT_FULL_RECORD'].shift(-1)
    sub_df['Year_lag1'] = sub_df['Year'].shift(-1)
    sub_df['Quarter_lag1'] = sub_df['Quarter'].shift(-1)
    sub_df = sub_df.drop(sub_df.index[-1])
    create_lag_list.append(sub_df)
df3 = pd.concat(create_lag_list, ignore_index=True)

df3.replace(0, np.nan, inplace=True)
df3.replace("NULL", np.nan, inplace=True)

df4 = df3[df3['Year']>= 1998]

############# Adding Latest announcement data from competitors ############

# df_X = pd.concat([train_X_df, test_X_df], ignore_index=True)
# df_y = pd.concat([train_y_df, test_y_df], ignore_index=True)
# df5 = df_X.merge(df_y, on=['Year','Quarter','ID_BB_UNIQUE'], how='left')


## If the outlier data problem has been resolved, this can be removed ##
outlier = df4['ARD_CAPITAL_EXPENDITURES'].max()
df4 = df4[df4['ARD_CAPITAL_EXPENDITURES'] != outlier]
df4 = df4[df4['ARD_CAPITAL_EXPENDITURES_lag1'] != outlier]

df4['ANNOUNCEMENT_DT_lag1'] = pd.to_datetime(df4['ANNOUNCEMENT_DT_lag1'], format='%Y-%m-%d')
df4.sort_values(by=['ANNOUNCEMENT_DT_lag1'],inplace=True)

list_of_cols = ['ANNOUNCEMENT_DT_lag1','ID_BB_UNIQUE']
list_of_cols.extend(lagged_y)
df_y = df4[list_of_cols]
df_cvx = df_y[df_y['ID_BB_UNIQUE'] == 'EQ0010031500001000']
df_cvx.columns = [f"{col}_cvx" for col in df_cvx.columns]
df_cvx.drop(columns=['ID_BB_UNIQUE_cvx'],inplace=True)
df_xom = df_y[df_y['ID_BB_UNIQUE'] == 'EQ0010054600001000']
df_xom.columns = [f"{col}_xom" for col in df_xom.columns]
df_xom.drop(columns=['ID_BB_UNIQUE_xom'],inplace=True)
df_cop = df_y[df_y['ID_BB_UNIQUE'] == 'EQ0010117400001000']
df_cop.columns = [f"{col}_cop" for col in df_cop.columns]
df_cop.drop(columns=['ID_BB_UNIQUE_cop'],inplace=True)

df4.sort_values(by=['ANNOUNCEMENT_DT_lag1'],inplace=True)

df4 = pd.merge_asof(df4, df_cvx, left_on='ANNOUNCEMENT_DT_lag1', right_on='ANNOUNCEMENT_DT_lag1_cvx', allow_exact_matches=False, direction='backward')
df4 = pd.merge_asof(df4, df_xom, left_on='ANNOUNCEMENT_DT_lag1', right_on='ANNOUNCEMENT_DT_lag1_xom', allow_exact_matches=False, direction='backward')
df4 = pd.merge_asof(df4, df_cop, left_on='ANNOUNCEMENT_DT_lag1', right_on='ANNOUNCEMENT_DT_lag1_cop', allow_exact_matches=False, direction='backward')

# print(trial_df[['ANNOUNCEMENT_DT','SALES_REV_TURN','SALES_REV_TURN_lag1','ANNOUNCEMENT_DT_lag1']])
cols_to_drop = ['ANNOUNCEMENT_DT_lag1_cvx','ANNOUNCEMENT_DT_lag1_xom','ANNOUNCEMENT_DT_lag1_cop']
df4.drop(columns=cols_to_drop, inplace=True)

df4.to_csv(final_data_path, index=False)