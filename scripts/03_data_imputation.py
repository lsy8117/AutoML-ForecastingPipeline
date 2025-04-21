import pandas as pd
import numpy as np
from data_preprocessing import mark_high_missing_columns, impute_missing_values, impute_for_test_set
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from feature_selection_methods import rolling_window_prediction, lightgbm_predict, lightgbm_tuned, calculate_quarterly_r2
import lightgbm as lgb
import matplotlib.pyplot as plt
import time

import warnings

warnings.filterwarnings('ignore')

cleaned_data_path = '/home/siyi/PycharmProjects/AutoML-ForecastingPipeline/data/bbg_data/processed_data.csv'
train_test_data_path = '/home/siyi/PycharmProjects/AutoML-ForecastingPipeline/data/train_test_data'

all_data = pd.read_csv(cleaned_data_path)

imputed_data = {}

y_cols = ['SALES_REV_TURN','CF_CASH_FROM_OPER','EBITDA','ARD_CAPITAL_EXPENDITURES']


lagged_y_cols = []
for y in y_cols:
    lagged_y_cols.append(y+'_lag1')

cat_cols = ['INDUSTRY_SECTOR_NUM_mapped', 'INDUSTRY_GROUP_NUM_mapped',
       'INDUSTRY_SUBGROUP_NUM_mapped', 'Industry_level_4_num_mapped',
       'Industry_level_5_num_mapped', 'Industry_level_6_num_mapped', 'ID_BB_UNIQUE', 'ID_BB_GLOBAL']


train_X_sets = []
train_y_sets = []
test_X_sets = []
test_y_sets = []


y_dataset_cols = lagged_y_cols.copy()
y_dataset_cols.extend(['Year','Quarter','ID_BB_UNIQUE'])


cols_not_impute = ['Year','Quarter','ANNOUNCEMENT_DT','ANNOUNCEMENT_DT_lag1','LATEST_PERIOD_END_DT_FULL_RECORD','REVISION_ID','ROW_NUMBER','FLAG','SECURITY_DESCRIPTION','RCODE','NFIELDS','EQY_FUND_CRNCY','LATEST_PERIOD_END_DT_FULL_RECORD_lag1','Year_lag1','Quarter_lag1']
cols_not_impute.extend(cat_cols)



######### Data Imputation Step ############
for comp, sub_df in all_data.groupby('ID_BB_UNIQUE'):
    print("Company: ", comp)
    print(sub_df.index.is_unique)
    # n_rows = len(sub_df)
    # n_train = int(n_rows * 0.7)
    # print("n_train: ", n_train)

    # Split the data into training and testing subsets
    # train_sub_df = sub_df.iloc[:n_train]
    # test_sub_df = sub_df.iloc[n_train:]
    # n_test = len(test_sub_df)
    train_sub_df = sub_df[sub_df['Year']<2018]
    test_sub_df = sub_df[sub_df['Year']>=2018]

    if len(train_sub_df) < 3:
        continue
    ## Impute for train set

    # Step 1: Drop columns with high missing values marked as np.nan
    df_processed = mark_high_missing_columns(train_sub_df, threshold=0.8)

    # Step 2: Handle missing values for each ticker
    df_processed = impute_missing_values(df_processed,cols_not_impute)

    train_df = df_processed.copy()
    x_df = train_df.drop(
        columns=lagged_y_cols)  ## dropping all lagged y variables from the features df, would not be available from the time of prediction

    train_X_sets.append(x_df)

    train_lagged_y = train_df[y_dataset_cols] # storing all the lagged y target for training set
    test_lagged_y = test_sub_df[y_dataset_cols] # storing all the lagged y target for test set

    train_y_sets.append(train_lagged_y)
    test_y_sets.append(test_lagged_y)

    print("x_df:")
    print(x_df.columns)

    ## Prepare test dataset
    ## ideally, data from time step k should be imputed using all data before k

    # test_df_processed = impute_missing_values(sub_df)

    ground_truth_df = test_sub_df.copy()
    test_sub_df2 = test_sub_df.copy()
    test_sub_df2 = test_sub_df2.drop(columns=lagged_y_cols)

    prep_df_for_test = pd.concat([x_df, test_sub_df2], ignore_index=True)
    test_df_processed = impute_for_test_set(prep_df_for_test, cols_not_impute)

    test_X_sets.append(test_df_processed[test_df_processed['Year']>=2018])

## Prepare dataset for feature selection
## Use current year X's with current year Y as target output
## For feature selection purpose, only use train_X_df, exclude columns that we don't want to involve in this process

train_X_df = pd.concat(train_X_sets, ignore_index=True)
train_y_df = pd.concat(train_y_sets, ignore_index=True)
test_X_df = pd.concat(test_X_sets, ignore_index=True)
test_y_df = pd.concat(test_y_sets, ignore_index=True)
test_X_df = test_X_df.sort_values(['Year','Quarter','ID_BB_UNIQUE'])
test_y_df = test_y_df.sort_values(['Year','Quarter','ID_BB_UNIQUE'])


train_X_df['ARD_CAPITAL_EXPENDITURES'] = -train_X_df['ARD_CAPITAL_EXPENDITURES']
test_X_df['ARD_CAPITAL_EXPENDITURES'] = -test_X_df['ARD_CAPITAL_EXPENDITURES']
train_y_df['ARD_CAPITAL_EXPENDITURES_lag1'] = -train_y_df['ARD_CAPITAL_EXPENDITURES_lag1']
test_y_df['ARD_CAPITAL_EXPENDITURES_lag1'] = -test_y_df['ARD_CAPITAL_EXPENDITURES_lag1']

train_X_df.to_csv(f'{train_test_data_path}/train_X_df.csv', index=False)
train_y_df.to_csv(f'{train_test_data_path}/train_y_df.csv', index=False)
test_X_df.to_csv(f'{train_test_data_path}/test_X_df.csv', index=False)
test_y_df.to_csv(f'{train_test_data_path}/test_y_df.csv', index=False)