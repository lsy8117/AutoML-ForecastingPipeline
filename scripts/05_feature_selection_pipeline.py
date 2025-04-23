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

train_test_data_path = '/home/siyi/PycharmProjects/AutoML-ForecastingPipeline/data/train_test_data'
test_result_path = '/home/siyi/PycharmProjects/AutoML-ForecastingPipeline/data/test_results'
evluating_data_name = 'w_v2'


## Use data with created features
train_X_df=pd.read_csv(f'{train_test_data_path}/train_X_df.csv')
train_y_df=pd.read_csv(f'{train_test_data_path}/train_y_df.csv')
test_X_df=pd.read_csv(f'{train_test_data_path}/test_X_df.csv')
test_y_df=pd.read_csv(f'{train_test_data_path}/test_y_df.csv')

##### Remove data rows affected by outliers #####
outlier = train_X_df['ARD_CAPITAL_EXPENDITURES'].max()
print('Outlier in capex: ', outlier)
X_cols = list(train_X_df.columns)
y_cols = list(train_y_df.columns)

train_df = train_X_df.merge(train_y_df, on=['Year','Quarter','ID_BB_UNIQUE'], how='left')
test_df = test_X_df.merge(test_y_df, on=['Year','Quarter','ID_BB_UNIQUE'], how='left')

# train_df = pd.concat([train_X_df, train_y_df], axis=1)
# test_df = pd.concat([test_X_df, test_y_df], axis=1)

train_df = train_df[~train_df.isin([outlier]).any(axis=1)]
test_df = test_df[~test_df.isin([outlier]).any(axis=1)]

train_X_df = train_df[X_cols]
train_y_df = train_df[y_cols]
test_X_df = test_df[X_cols]
test_y_df = test_df[y_cols]

print(train_X_df.dtypes[train_df.dtypes == 'object'])



## Exclude Year, categorical columns and other irrelevant columns
cols_excluded = ['Date_x','month_year','Date_y','period_ending_date_x','period_ending_date_y','s_p_estimate_ID_BB_UNIQUE','period_ending_date','Year','Date', 'YearMonth','LATEST_PERIOD_END_DT_FULL_RECORD','REVISION_ID','ROW_NUMBER','FLAG','SECURITY_DESCRIPTION','RCODE','NFIELDS','EQY_FUND_CRNCY','ANNOUNCEMENT_DT','FUNDAMENTAL_ENTRY_DT','ANNOUNCEMENT_DT_lag1','LATEST_PERIOD_END_DT_FULL_RECORD_lag1']

categorical_cols = ['INDUSTRY_SECTOR_NUM_mapped', 'INDUSTRY_GROUP_NUM_mapped',
       'INDUSTRY_SUBGROUP_NUM_mapped', 'Industry_level_4_num_mapped',
       'Industry_level_5_num_mapped', 'Industry_level_6_num_mapped','ID_BB_UNIQUE','ID_BB_GLOBAL']

y_cols = ['SALES_REV_TURN','CF_CASH_FROM_OPER','EBITDA','ARD_CAPITAL_EXPENDITURES']


################## Feature Selection Step ##################
## Iterate over the 4 Y target for feature selection process
start_time = time.time()
prediction_results = {}

for y in y_cols:
    ## Adding the current Y target to exclude from input feature list
    y_start = time.time()
    print("Selecting features for: ", y)

    cols_exclude_from_feature_selection = []
    for col in train_X_df.columns:
        if col in cols_excluded:
            cols_exclude_from_feature_selection.append(col)

    # cols_exclude_from_feature_selection = cols_excluded.copy()
    cols_exclude_from_feature_selection.append(y)

    input_features_df = train_X_df.drop(columns=cols_exclude_from_feature_selection, inplace=False)

    cat_cols = []
    for feature in categorical_cols:
        if feature in input_features_df.columns:
            input_features_df[feature] = input_features_df[feature].astype('category')
            cat_cols.append(feature)

    lgbm_params = {
        'boosting_type': 'gbdt',
        'objective': 'mape',
        'class_weight': None,
        'min_split_gain': 0.0,
        'min_child_weight': 0.001,
        'subsample': 1.0,
        'subsample_freq': 0,
        'colsample_bytree': 1.0,
        'reg_alpha': 0.005,
        # 'lambda_l1': 0.2,
        'reg_lambda': 0.0,
        'random_state': None,
        'n_jobs': -1,
        'verbose': -1,
        'learning_rate': 0.005,
        'n_estimators': 1000,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
    }

    y_label = y + '_lag1'
    print('Input features:',input_features_df.columns)
    train_data = lgb.Dataset(input_features_df, label=train_y_df[y_label], categorical_feature=cat_cols)
    model = lgb.train(lgbm_params, train_data, num_boost_round=20)

    importance = model.feature_importance(importance_type='gain')

    feature_names = model.feature_name()

    importance_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
    importance_df = importance_df.sort_values(by='importance', ascending=False)

    feature_num = 10
    max_feature_num = 30
    feature_num_step = 5
    recall = 2
    k = 0 ##
    previous_rmse = float('inf')
    best_rmse = float('inf')
    best_rmse_feature_num = float('inf')
    best_rmse_feature_set = []
    best_rmse_r2 = 0
    best_test_df_pred = None
    best_train_df_pred = None

    # ####### Start iterate around different numbers of feature selection ######
    while feature_num_step != 0 and feature_num > 1:
        print("number of features: ", feature_num)
        selected_features = importance_df[:feature_num]['feature'].tolist()


        features_for_model = selected_features.copy()
        features_for_model.append(y)

        print("Selected features:")
        print(features_for_model)

        cat_vars = []

        for feature in features_for_model:
            if feature in categorical_cols:
                cat_vars.append(feature)
                train_X_df[feature] = train_X_df[feature].astype('category')
                test_X_df[feature] = test_X_df[feature].astype('category')

        y_label = y + '_lag1'
        merge_train = train_X_df.merge(train_y_df, on=['Year','Quarter','ID_BB_UNIQUE'], how='left')
        merge_test = test_X_df.merge(test_y_df, on=['Year','Quarter','ID_BB_UNIQUE'], how='left')
        for feature in features_for_model:
            if feature in categorical_cols:
                cat_vars.append(feature)
                merge_train[feature] = merge_train[feature].astype('category')
                merge_test[feature] = merge_test[feature].astype('category')
        final_rmse, final_r2, test_df_pred, train_df_pred = lightgbm_predict(merge_train, merge_test, y_label, features_for_model, cat_vars)


        pd.set_option('display.max_rows', None)

        print(final_rmse)


        if feature_num_step > 0:
            if final_rmse < previous_rmse:
                if final_rmse < best_rmse:
                    best_rmse = final_rmse
                    best_rmse_feature_num = feature_num
                    best_rmse_feature_set = features_for_model
                    best_rmse_r2 = final_r2
                    best_test_df_pred = test_df_pred
                    best_train_df_pred = train_df_pred
                previous_rmse = final_rmse

            else:
                feature_num_step = -1
        else:
            if final_rmse < best_rmse:
                best_rmse = final_rmse
                best_rmse_feature_num = feature_num
                best_rmse_feature_set = features_for_model
                best_rmse_r2 = final_r2
                best_test_df_pred = test_df_pred
                best_train_df_pred = train_df_pred
            elif final_rmse > best_rmse and feature_num < best_rmse_feature_num:
                feature_num_step = 0
            elif final_rmse == best_rmse:
                best_rmse_feature_num = feature_num
                best_rmse_feature_set = features_for_model
                best_rmse_r2 = final_r2
                best_test_df_pred = test_df_pred
                best_train_df_pred = train_df_pred
        previous_rmse = final_rmse
        feature_num += feature_num_step


    print("Optimal number of features: ", best_rmse_feature_num)
    print("Optimal RMSE: ", best_rmse)
    print("R-squared: ", best_rmse_r2)
    print("Optimal features selected:")
    print(best_rmse_feature_set)
    y_end = time.time()
    y_time = y_end - y_start
    # pd.set_option('display.max_rows', None)
    # print(best_test_df_pred[['ID_BB_UNIQUE','Year','Quarter',y_label, f'pred_{y_label}']])
    curr_test_result = best_test_df_pred[['ID_BB_UNIQUE','Year','Quarter',y_label, f'pred_{y_label}']]
    curr_test_result.to_csv(f'{test_result_path}/{evluating_data_name}_test_result_{y_label}.csv', index=False)
    quarterly_r2 = calculate_quarterly_r2(curr_test_result, y_label, f'pred_{y_label}')
    quarterly_r2.to_csv(f'{test_result_path}/{evluating_data_name}_test_result_r2_{y_label}.csv', index=False)
    prediction_results[y] = {'num_features': best_rmse_feature_num,'RMSE': best_rmse, 'R-squared': best_rmse_r2, 'features_selected': best_rmse_feature_set}

    if y == 'SALES_REV_TURN':
        test_X_df[f'pred_{y_label}'] = best_test_df_pred[f'pred_{y_label}']
        train_X_df[f'pred_{y_label}'] = best_train_df_pred
    print(f"Time taken to process {y} is {y_time} seconds")
    print("===================================")

end_time = time.time()
execution_time = end_time - start_time
prediction_results_df = pd.DataFrame.from_dict(prediction_results, orient='index')
pd.options.display.float_format = '{:.4g}'.format
print(prediction_results_df[['num_features','RMSE','R-squared']])
prediction_results_df.to_csv(f'{test_result_path}/{evluating_data_name}_features_selected.csv', index=False)
print(f"Program execution time: {execution_time} seconds")

train_X_df.drop(columns=['pred_SALES_REV_TURN_lag1'],inplace=True)
test_X_df.drop(columns=['pred_SALES_REV_TURN_lag1'],inplace=True)