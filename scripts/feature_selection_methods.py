import pandas as pd
import numpy as np
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor


# import optuna


def rolling_window_prediction(train_X, train_y, test_X, test_y, cat_cols, selected_features, y_label):
    test_start = test_X['Year'].min()
    test_end = test_X['Year'].max()

    lgbm_params = {
        'boosting_type': 'gbdt',
        'objective': 'regression_l2',
        'class_weight': None,
        'min_split_gain': 0.0,
        'min_child_weight': 0.001,
        'subsample': 1.0,
        'subsample_freq': 0,
        'colsample_bytree': 1.0,
        'reg_alpha': 0.0,
        'reg_lambda': 0.0,
        'random_state': None,
        'n_jobs': -1,
        'verbose': -1
    }

    merge_train = train_X.merge(train_y, on=['Year', 'ID_BB_UNIQUE'], how='left')

    train_data = lgb.Dataset(merge_train[selected_features], label=merge_train[y_label], categorical_feature=cat_cols)
    model = lgb.train(lgbm_params, train_data, num_boost_round=5)

    pred_results = []

    for year in range(test_start, test_end):
        test_X_curr_year = test_X[test_X['Year'] == year]
        test_y_curr_year = test_y[test_y['Year'] == year]

        merged_df = test_X_curr_year.merge(test_y_curr_year, on=['Year', 'ID_BB_UNIQUE'], how='left')
        merged_df = merged_df.sort_values(by=['ID_BB_UNIQUE']).reset_index(drop=True)

        result = model.predict(data=merged_df[selected_features])

        curr_result = pd.DataFrame({"prediction": result, "ground_truth": merged_df[y_label]})

        pred_results.append(curr_result)

        if year != test_end - 1:
            train_X = pd.concat([train_X, test_X_curr_year])
            train_y = pd.concat([train_y, test_y_curr_year])

            for col in cat_cols:
                train_X[col] = train_X[col].astype('category')

            merge_train = train_X.merge(train_y, on=['Year', 'ID_BB_UNIQUE'], how='left')

            train_data = lgb.Dataset(merge_train[selected_features], label=merge_train[y_label], categorical_feature=cat_cols)
            model = lgb.train(lgbm_params, train_data, num_boost_round=5)

    result_df = pd.concat(pred_results)

    result_df_clean = result_df.dropna(subset=['ground_truth'])

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(result_df_clean['ground_truth'], result_df_clean['prediction']))
    r2 = r2_score(result_df_clean['ground_truth'], result_df_clean['prediction'])

    return result_df_clean, rmse, r2

def lasso_sequential_feature_selection(input_features_df, y_col, feature_num):
    lasso = Lasso(alpha=0.1)
    ## Try different numbers of features
    selector = SequentialFeatureSelector(lasso, n_features_to_select=feature_num, direction='forward')

    ## Fit selector model
    selector.fit(input_features_df, y_col)

    selected_features = selector.get_feature_names_out().tolist()
    return selected_features

def randomforest_tune(train_df, test_df, y_label, features_cols):
    param_grid = {'n_estimators': [50, 100, 200, 300, 500]}
    tscv = TimeSeriesSplit(n_splits=2)

    rf_model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(rf_model, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(train_df[features_cols], train_df[y_label])

    best_rf_model = grid_search.best_estimator_

    y_pred = best_rf_model.predict(test_df[features_cols])

    rmse = np.sqrt(mean_squared_error(test_df[y_label], y_pred))
    r2 = r2_score(test_df[y_label], y_pred)
    return rmse, r2

def lightgbm_tuned(train_df, test_df, y_label, features_cols):
    tscv = TimeSeriesSplit(n_splits=3)

    param_grid = {
        # 'n_estimators': np.arange(600, 1600, 100),
        # 'learning_rate': [0.001, 0.005, 0.01],
        'feature_fraction': [0.7, 0.8, 0.9],
        'bagging_fraction': [0.7, 0.8, 0.9],
        # 'bagging_freq': [3,4,5,6,7],
        # 'max_depth': [-1, 5, 10],
        # 'subsample': [0.7, 0.9, 1.0],
        # 'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        # 'reg_alpha': [0.1, 0.2, 0.3],
        # 'reg_lambda': [0.1, 0.2, 0.3],
    }

    # param_grid = {
    #     'num_leaves': np.arange(20, 100, 10),
    #     'max_depth': np.arange(3, 12, 1),
    #     # 'learning_rate': np.logspace(-3, 0, 5),
    #     'n_estimators': np.arange(700, 1500, 100),
    #     'min_child_samples': np.arange(20, 200, 20),
    #     # 'min_child_weight': np.linspace(0.001, 0.1, 5),
    #     'subsample': np.linspace(0.5, 1.0, 5),
    #     'colsample_bytree': np.linspace(0.5, 1.0, 5),
    #     'feature_fraction': np.linspace(0.5, 1.0, 5),
    #     'bagging_fraction': np.linspace(0.5, 1.0, 5),
    #     # 'bagging_freq': [1, 2, 5],  # Try different frequencies
    #     'reg_alpha': np.logspace(-3, 1, 5),
    #     'reg_lambda': np.logspace(-3, 1, 5)
    # }



    lgbm = lgb.LGBMRegressor(
        boosting_type='gbdt',  # Standard gradient boosting decision tree
        # device='gpu',
        verbose=-1,
        objective='regression',
        class_weight=None,
        min_split_gain=0.0,
        min_child_weight=0.001,
        subsample=1.0,
        subsample_freq=0,
        colsample_bytree=1.0,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        learning_rate=0.005,
        n_estimators=1500,
        bagging_frequency=5,
        # feature_fraction=0.9,
        # bagging_fraction=0.7,
    )

    grid_search = GridSearchCV(lgbm, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=1, verbose=1)
    grid_search.fit(train_df[features_cols], train_df[y_label])

    # random_search = RandomizedSearchCV(lgbm, param_grid, scoring='neg_mean_squared_error', n_jobs=1, verbose=0, cv=tscv, random_state=42)
    # random_search.fit(train_df[features_cols], train_df[y_label])

    # Best model after tuning
    best_model = grid_search.best_estimator_

    print("Best parameters by Random Search:")
    print(grid_search.best_params_)

    train_pred = best_model.predict(train_df[features_cols])
    valid_indices = train_df[y_label].notna()
    valid_train_y = train_df[y_label][valid_indices]
    valid_y_pred = train_pred[valid_indices]
    train_rmse = np.sqrt(mean_squared_error(valid_train_y, valid_y_pred))
    train_r2 = r2_score(valid_train_y, valid_y_pred)
    print('In sample rmse after grid search: ', train_rmse)
    print('In sample R2 after grid search: ', train_r2)

    result = best_model.predict(test_df[features_cols])

    mask = ~(np.isnan(result) | test_df[y_label].isnull())

    rmse = np.sqrt(mean_squared_error(test_df[y_label][mask], result[mask]))
    r2 = r2_score(test_df[y_label][mask], result[mask])
    print('Test set RMSE: ', rmse)
    print('Test set R2: ', r2)

    test_df[f'pred_{y_label}'] = result
    test_df[f'pred_{y_label}'] = test_df[f'pred_{y_label}'].astype(float)

    return rmse, r2, test_df

# def lightgbm_bayesiansearch(train_df, test_df, y_label, features_cols, cat_cols):
#     params = {
#         'num_leaves': trial.suggest_int('num_leaves', 20, 100),
#         'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
#         'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
#         'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 50)
#     }

def lightgbm_predict(train_df, test_df, y_label, features_cols, cat_cols):
    lgbm_params = {
        'boosting_type': 'gbdt',
        'objective': 'regression_l2',
        'class_weight': None,
        # 'min_split_gain': 0.0,
        'min_child_weight': 0.001,
        'subsample': 1.0,
        'subsample_freq': 0,
        'colsample_bytree': 1.0,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
        'learning_rate': 0.005,
        'n_estimators': 1500,
        'feature_fraction': 0.9,  # Use 80% of features
        'bagging_fraction': 0.7,  # Use 80% of data for each tree
        'bagging_freq': 5
    }

    train_data = lgb.Dataset(train_df[features_cols], label=train_df[y_label], categorical_feature=cat_cols)
    model = lgb.train(lgbm_params, train_data, num_boost_round=5)

    train_pred = model.predict(data=train_df[features_cols])
    print("train_df length: ", len(train_df))
    print("train data prediction length: ", len(train_pred))
    valid_indices = train_df[y_label].notna()
    valid_train_y = train_df[y_label][valid_indices]
    valid_y_pred = train_pred[valid_indices]
    rmse = np.sqrt(mean_squared_error(valid_train_y, valid_y_pred))
    r2 = r2_score(valid_train_y, valid_y_pred)
    print("In sample model RMSE with selected features:", rmse)
    print("In sample model R2: ", r2)

    result = model.predict(data=test_df[features_cols])

    mask = ~(np.isnan(result) | test_df[y_label].isnull())

    rmse = np.sqrt(mean_squared_error(test_df[y_label][mask], result[mask]))
    r2 = r2_score(test_df[y_label][mask], result[mask])

    print("Test set RMSE with selected features:", rmse)
    print("Test set model R2: ", r2)

    test_df[f'pred_{y_label}'] = result
    test_df[f'pred_{y_label}'] = test_df[f'pred_{y_label}'].astype(float)

    return rmse, r2, test_df, train_pred



def linear_reg_predict(train_df, test_df, y_label, features_cols):
    model = HistGradientBoostingRegressor()
    model.fit(train_df[features_cols], train_df[y_label])

    # Make predictions
    result = model.predict(test_df[features_cols])
    mask = ~(np.isnan(result) | test_df[y_label].isnull())

    rmse = np.sqrt(mean_squared_error(test_df[y_label][mask], result[mask]))
    r2 = r2_score(test_df[y_label][mask], result[mask])

    print("Test data RMSE:", rmse)
    print("Test data R2: ", r2)

    test_df[f'pred_{y_label}'] = result
    test_df[f'pred_{y_label}'] = test_df[f'pred_{y_label}'].astype(float)

    return rmse, r2, test_df


def lightgbm_predict_cv(train_df, test_df, y_label, features_cols, cat_cols):
    lgbm_params = {
        'boosting_type': 'gbdt',
        'objective': 'regression_l2',
        'class_weight': None,
        # 'min_split_gain': 0.0,
        'min_child_weight': 0.001,
        'subsample': 1.0,
        'subsample_freq': 0,
        'colsample_bytree': 1.0,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'n_jobs': -1,
        # 'verbose': 1,
        'learning_rate': 0.005,
        'n_estimators': 1500,
        'feature_fraction': 0.9,  # Use 80% of features
        'bagging_fraction': 0.7,  # Use 80% of data for each tree
        'bagging_freq': 5
    }

    train_data = lgb.Dataset(train_df[features_cols], label=train_df[y_label], categorical_feature=cat_cols)

    cv_results = lgb.cv(
        lgbm_params,
        train_data,
        num_boost_round=1500,
        nfold=5,
        stratified=False,
        shuffle=True,
        early_stopping_rounds=50,
        verbose_eval=100,
        seed=42
    )

    best_num_boost_round = len(cv_results['rmse-mean'])
    print("Best number of boosting rounds from CV:", best_num_boost_round)

    # Step 3: Train the final model with best boosting rounds
    model = lgb.train(lgbm_params, train_data, num_boost_round=best_num_boost_round)

    train_pred = model.predict(data=train_df[features_cols])
    print("train_df length: ", len(train_df))
    print("train data prediction length: ", len(train_pred))
    valid_indices = train_df[y_label].notna()
    valid_train_y = train_df[y_label][valid_indices]
    valid_y_pred = train_pred[valid_indices]
    rmse = np.sqrt(mean_squared_error(valid_train_y, valid_y_pred))
    r2 = r2_score(valid_train_y, valid_y_pred)
    print("In sample model RMSE with selected features:", rmse)
    print("In sample model R2: ", r2)

    result = model.predict(data=test_df[features_cols])

    mask = ~(np.isnan(result) | test_df[y_label].isnull())

    rmse = np.sqrt(mean_squared_error(test_df[y_label][mask], result[mask]))
    r2 = r2_score(test_df[y_label][mask], result[mask])

    print("Test set RMSE with selected features:", rmse)
    print("Test set model R2: ", r2)

    test_df[f'pred_{y_label}'] = result
    test_df[f'pred_{y_label}'] = test_df[f'pred_{y_label}'].astype(float)

    return rmse, r2, test_df, train_pred

def lightgbm_predict_cv2(train_df, test_df, y_label, features_cols, cat_cols):
    lgbm_params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'class_weight': None,
        # 'min_split_gain': 0.0,
        'min_child_weight': 0.001,
        'subsample': 1.0,
        'subsample_freq': 0,
        'colsample_bytree': 1.0,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
        'learning_rate': 0.01,
        'n_estimators': 500,
        'feature_fraction': 0.8,  # Use 80% of features
        'bagging_fraction': 0.8,  # Use 80% of data for each tree
        'bagging_freq': 5
    }

    # train_data = lgb.Dataset(train_df[features_cols], label=train_df[y_label], categorical_feature=cat_cols)
    train_data = train_df[features_cols]
    train_label = train_df[y_label]

    metrics_rmse = []
    metrics_r2 = []

    tscv = TimeSeriesSplit(n_splits=3)
    for train_index, test_index in tscv.split(train_data):
        X_train, X_test = train_data.iloc[train_index], train_data.iloc[test_index]
        y_train, y_test = train_label[train_index], train_label[test_index]

        # Create LightGBM datasets for training and testing
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        # Train a LightGBM model
        num_round = 100
        bst = lgb.train(lgbm_params, train_data, num_round)

        # Make predictions on the test set
        y_pred = bst.predict(X_test)

        valid_indices = y_test.notna()
        valid_y_label = y_test[valid_indices]
        valid_y_pred = y_pred[valid_indices]
        rmse = np.sqrt(mean_squared_error(valid_y_label, valid_y_pred))
        r2 = r2_score(valid_y_label, valid_y_pred)

        metrics_rmse.append(rmse)
        metrics_r2.append(r2)

    # Calculate the average accuracy across all folds
    average_rmse = np.mean(metrics_rmse)
    print(f'Average RMSE: {average_rmse:.4f}')
    average_r2 = np.mean(metrics_r2)
    print(f'Average R2: {average_r2:.4f}')

    model = lgb.train(lgbm_params, train_data, num_boost_round=5)

    train_pred = model.predict(data=train_df[features_cols])
    valid_indices = train_df[y_label].notna()
    valid_train_y = train_df[y_label][valid_indices]
    valid_y_pred = train_pred[valid_indices]
    rmse = np.sqrt(mean_squared_error(valid_train_y, valid_y_pred))
    r2 = r2_score(valid_train_y, valid_y_pred)
    print("In sample model RMSE with selected features:", rmse)
    print("In sample model R2: ", r2)

    result = model.predict(data=test_df[features_cols])

    mask = ~(np.isnan(result) | test_df[y_label].isnull())

    rmse = np.sqrt(mean_squared_error(test_df[y_label][mask], result[mask]))
    r2 = r2_score(test_df[y_label][mask], result[mask])
    return rmse, r2

def calculate_quarterly_r2(df, true_y_label, pred_y_label):
    r2_list = []
    for id, sub_df in df.groupby(['Year','Quarter']):
        # print(id)
        valid_indices = sub_df[true_y_label].notna()
        valid_true_y = sub_df[true_y_label][valid_indices]
        valid_y_pred = sub_df[pred_y_label][valid_indices]

        r2 = r2_score(valid_true_y, valid_y_pred)
        r2_list.append({'year':id[0], 'quarter':id[1], 'r2':r2})
        # print(r2)

    final_df = pd.DataFrame(r2_list)
    return final_df



## Testing code
# features_cols = ['ARD_CAPITAL_EXPENDITURES','CEst_Capital','Quarter']
#
# y_label = 'ARD_CAPITAL_EXPENDITURES_lag1'
# cat_vars = []
# for feature in features_cols:
#     if feature in categorical_cols:
#         cat_vars.append(feature)
# merge_train = train_X_df.merge(train_y_df, on=['Year','Quarter','ID_BB_UNIQUE'], how='left')
# merge_test = test_X_df.merge(test_y_df, on=['Year','Quarter','ID_BB_UNIQUE'], how='left')
# final_rmse, final_r2, test_df_pred, train_pred = lightgbm_predict(merge_train, merge_test, y_label, features_cols, cat_vars)
# test_df_pred[['Year','Quarter','ID_BB_UNIQUE',f'pred_{y_label}'，‘LATEST_PERIOD_END_DT_FULL_RECORD_lag1’]].to_csv(f'/home/siyi/data/oil_firm/bbg_full/compare_results/{y_label}_results.csv', index=False)