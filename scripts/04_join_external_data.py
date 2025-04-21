import numpy as np

from data_preprocessing import *
# from quarter_data_preprocess import *
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

train_test_data_path = '/home/siyi/PycharmProjects/AutoML-ForecastingPipeline/data/train_test_data'
external_data_path = '/home/siyi/PycharmProjects/AutoML-ForecastingPipeline/data/external_data'

################ Join Imputed train test datasets with external data ###############
##### (1) Read the imputed BBG data
train_X_df = pd.read_csv(f'{train_test_data_path}/train_X_df.csv')
test_X_df = pd.read_csv(f'{train_test_data_path}/test_X_df.csv')
X_df = pd.concat([train_X_df, test_X_df])

X_df['LATEST_PERIOD_END_DT_FULL_RECORD'] = pd.to_datetime(X_df['LATEST_PERIOD_END_DT_FULL_RECORD'], format='%Y-%m-%d')

X_df['LATEST_PERIOD_END_DT_FULL_RECORD_lag1'] = pd.to_datetime(X_df['LATEST_PERIOD_END_DT_FULL_RECORD_lag1'], format='%Y-%m-%d')

X_df['YearMonth'] = X_df['LATEST_PERIOD_END_DT_FULL_RECORD'].dt.to_period('M')

##### Read the external datasets and merge with BBG data
## Yahoo Finance data
yfinance_df = pd.read_csv(f'{external_data_path}/yahoo_finance_oil_related_data_with_MA.csv')

yfinance_df['Date'] = pd.to_datetime(yfinance_df['Date'], format='%Y-%m-%d')

yfinance_df[f"S_P_500_Index_return"] = ((yfinance_df['S&P 500 Index'] - yfinance_df['S&P 500 Index'].shift(252)) / yfinance_df['S&P 500 Index'].shift(252))
yfinance_df['risk_free_rate'] = yfinance_df['U.S._10_Year_Treasury_Yield']/100
yfinance_df.replace([np.inf, -np.inf], np.nan, inplace=True)

X_df.sort_values(by='LATEST_PERIOD_END_DT_FULL_RECORD', ascending=True, inplace=True)

X_df = pd.merge_asof(X_df,yfinance_df, left_on='LATEST_PERIOD_END_DT_FULL_RECORD', right_on='Date',direction='backward')

### Create D1 Estimate ###

df_list = []
for id, sub_df in X_df.groupby('ID_BB_UNIQUE'):
    sub_df = sub_df.sort_values(['Year','Quarter'], ascending=True)
    beta = 1.2
    if id == 'EQ0010031500001000':
        beta = 0.75
    elif id == 'EQ0010117400001000':
        beta = 0.85
    elif id == 'EQ0010054600001000':
        beta = 0.7
    sub_df['required_return'] = sub_df['risk_free_rate'] - beta*(sub_df['S_P_500_Index_return']-sub_df['risk_free_rate'])
    sub_df['growth_of_equity'] = (1-(sub_df['DVD_PAYOUT_RATIO']/100))*sub_df['GEO_GROW_ROE']
    sub_df['dividend_growth_rate'] = (sub_df['ARD_DVD_PER_SH'] - sub_df['ARD_DVD_PER_SH'].shift(3)) / sub_df['ARD_DVD_PER_SH'].shift(1)
    if id == 'EQ0010031500001000': ## cvx
        sub_df['D1'] = sub_df['Chevron'] * (sub_df['required_return'] - sub_df['dividend_growth_rate'])
    elif id == 'EQ0010117400001000': ## cop
        sub_df['D1'] = sub_df['ConocoPhillips'] * (sub_df['required_return'] - sub_df['dividend_growth_rate'])
    elif id == 'EQ0010054600001000': ## xom
        sub_df['D1'] = sub_df['ExxonMobil'] * (sub_df['required_return'] - sub_df['dividend_growth_rate'])

    sub_df['total_D1'] = sub_df['D1']*sub_df['BS_SH_OUT']
    sub_df['estimated_NI_wD1'] = sub_df['total_D1']/(sub_df['DVD_PAYOUT_RATIO']/100)
    df_list.append(sub_df)
X_df = pd.concat(df_list, ignore_index=True)

## EIA data, US energy market summary
us_energy_df = pd.read_csv(f'{external_data_path}/us_energy_market_summary_processed.csv')

us_energy_df['year_month'] = pd.to_datetime(us_energy_df['year_month'], format='%Y-%m-%d')
us_energy_df['YearMonth'] = us_energy_df['year_month'].dt.to_period('M')
us_energy_df.drop(columns=['year_month'],inplace=True)# '1997-01', '1998-02', ...


X_df = X_df.merge(us_energy_df, on='YearMonth', how='left')

####### Refinery Data #######
refinery_df = pd.read_csv(f'{external_data_path}/monthly_refinery_data_processed.csv')
refinery_df['year_month'] = pd.to_datetime(refinery_df['year_month'], format='%Y-%m')
refinery_df['YearMonth'] = refinery_df['year_month'].dt.to_period('M')
refinery_df.drop(columns=['year_month'],inplace=True)

X_df = X_df.merge(refinery_df, on='YearMonth', how='left')

####### Prompt Month Data #######
pm_df = pd.read_csv(f'{external_data_path}/oil_refined_product_prompt_month_processed.csv')
pm_df['Date'] = pd.to_datetime(pm_df['Date'], format='%Y-%m-%d')


X_df = X_df.merge(pm_df, left_on='LATEST_PERIOD_END_DT_FULL_RECORD', right_on='Date',how='left')

####### EIA US Crude Oil Summary #######
summary_df = pd.read_csv(f'{external_data_path}/US_crude_oil_data_summary_processed.csv')
summary_df['year_month'] = pd.to_datetime(summary_df['year_month'], format='%Y-%m')
summary_df['YearMonth'] = summary_df['year_month'].dt.to_period('M')
summary_df.drop(columns=['year_month'],inplace=True)

X_df = X_df.merge(summary_df, on='YearMonth', how='left')

####### Monthly Spot Prices Data #######
sp_df = pd.read_csv(f'{external_data_path}/monthly_spot_prices_processed.csv')
sp_df['year_month'] = pd.to_datetime(sp_df['year_month'], format='%Y-%m')
sp_df['YearMonth'] = sp_df['year_month'].dt.to_period('M')
sp_df.drop(columns=['year_month'],inplace=True)

X_df = X_df.merge(sp_df, on='YearMonth', how='left')

####### Monthly Diesel Prices Data #######
diesel_df = pd.read_csv(f'{external_data_path}/monthly_diesel_prices_processed.csv')
diesel_df['year_month'] = pd.to_datetime(diesel_df['year_month'], format='%Y-%m')
diesel_df['YearMonth'] = diesel_df['year_month'].dt.to_period('M')
diesel_df.drop(columns=['year_month'],inplace=True)

X_df = X_df.merge(diesel_df, on='YearMonth', how='left')

####### Monthly Gasoline Prices Data #######
gasoline_df = pd.read_csv(f'{external_data_path}/monthly_gasoline_prices_processed.csv')
gasoline_df['year_month'] = pd.to_datetime(gasoline_df['year_month'], format='%Y-%m')
gasoline_df['YearMonth'] = gasoline_df['year_month'].dt.to_period('M')
gasoline_df.drop(columns=['year_month'],inplace=True)

X_df = X_df.merge(gasoline_df, on='YearMonth', how='left')

####### Monthly Supply and Disposition Data #######
sd_df = pd.read_csv(f'{external_data_path}/monthly_supply_disposition_processed.csv')
sd_df['year_month'] = pd.to_datetime(sd_df['year_month'], format='%Y-%m')
sd_df['YearMonth'] = sd_df['year_month'].dt.to_period('M')
sd_df.drop(columns=['year_month'],inplace=True)

X_df = X_df.merge(sd_df, on='YearMonth', how='left')

####### Monthly Rigs Count #######
rc_df = pd.read_csv(f'{external_data_path}/rigs_count.csv')
rc_df['year_month'] = pd.to_datetime(rc_df['year_month'], format='%Y-%m')
rc_df['YearMonth'] = rc_df['year_month'].dt.to_period('M')
rc_df.drop(columns=['year_month'],inplace=True)

X_df = X_df.merge(rc_df, on='YearMonth', how='left')

####### Company operation data #######
op_df = pd.read_csv(f'{external_data_path}/company_variables_quarter.csv')
op_df = op_df[~op_df['period_ending_date'].astype(str).str.endswith('.1')]
op_df['period_ending_date'] = pd.to_datetime(op_df['period_ending_date'], format='%d/%m/%Y')
# op_df.drop(columns=['ID_BB_UNIQUE'],inplace=True)

X_df = X_df.merge(op_df, left_on=['LATEST_PERIOD_END_DT_FULL_RECORD','ID_BB_UNIQUE'], right_on=['period_ending_date','ID_BB_UNIQUE'], how='left')

####### Analysts estimated data #######
estimate_df = pd.read_csv(f'{external_data_path}/company_estimates_data.csv')
estimate_df = estimate_df[~estimate_df['period_ending_date'].astype(str).str.endswith('.1')]
estimate_df['period_ending_date'] = pd.to_datetime(estimate_df['period_ending_date'], format='%d/%m/%Y')
# op_df.drop(columns=['ID_BB_UNIQUE'],inplace=True)

X_df = X_df.merge(estimate_df, left_on=['LATEST_PERIOD_END_DT_FULL_RECORD_lag1','ID_BB_UNIQUE'], right_on=['period_ending_date','ID_BB_UNIQUE'], how='left')

X_df['CEst_Capital'] = -X_df['CEst_Capital'] ## Convert capital expenditure from negative to positive numbers

####### S&P estimated data #######
sp_estimate_df = pd.read_csv(f'{external_data_path}/processed_s&p_estimates.csv')

X_df = X_df.merge(sp_estimate_df, left_on=['Year_lag1','Quarter_lag1','ID_BB_UNIQUE'], right_on=['s_p_estimate_Year','s_p_estimate_Quarter','s_p_estimate_ID_BB_UNIQUE'], how='left')

train_X_df = X_df[X_df['Year']<2018]
test_X_df = X_df[X_df['Year']>=2018]

train_X_df.columns = train_X_df.columns.str.replace(' ', '_')
test_X_df.columns = test_X_df.columns.str.replace(' ', '_')

train_X_df.replace('--', np.nan, inplace=True)
test_X_df.replace('--', np.nan, inplace=True)

train_X_df.columns = train_X_df.columns.str.replace(r'[^a-zA-Z0-9_.()]', '_', regex=True)
test_X_df.columns = test_X_df.columns.str.replace(r'[^a-zA-Z0-9_.()]', '_', regex=True)

train_X_df = train_X_df.astype({col: float for col in train_X_df.columns if isinstance(train_X_df[col].dtype, np.dtypes.Float64DType)})
test_X_df = test_X_df.astype({col: float for col in test_X_df.columns if isinstance(test_X_df[col].dtype, np.dtypes.Float64DType)})