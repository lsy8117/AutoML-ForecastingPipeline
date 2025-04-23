import pandas as pd
from scripts.data_preprocessing import read_industry_code_mapping, create_industry_stats_for_year
import warnings

warnings.filterwarnings('ignore')

industry_mapping = read_industry_code_mapping()
print(industry_mapping)

process_folder_path = '/home/siyi/PycharmProjects/AutoML-ForecastingPipeline/data'
save_folder_path = '/home/siyi/PycharmProjects/AutoML-ForecastingPipeline/data'
process_file_name = 'train_X_df.csv'
save_file_name = 'train_X_df_w_industry_stats.csv'

process_file_path = f'{process_folder_path}/{process_file_name}' ## X_df that you want to process
save_file_path = f'{save_folder_path}/{save_file_name}'
df = pd.read_csv(process_file_path)
# industry_groupings = ['INDUSTRY_SECTOR_NUM_mapped', 'INDUSTRY_GROUP_NUM_mapped', 'INDUSTRY_SUBGROUP_NUM_mapped', 'Industry_level_4_num_mapped', 'Industry_level_5_num_mapped', 'Industry_level_6_num_mapped']
industry_groupings = ['Industry_level_4_num_mapped', 'Industry_level_5_num_mapped', 'Industry_level_6_num_mapped']


start_year = df['Year'].min()
end_year = df['Year'].max()

############ Get the feature columns ##############

## Define the list of columns that you want to calcualte for industry statistics for ##
feature_cols = list(df.columns)
cat_cols = ['INDUSTRY_SECTOR_NUM_mapped', 'INDUSTRY_GROUP_NUM_mapped',
       'INDUSTRY_SUBGROUP_NUM_mapped', 'Industry_level_4_num_mapped',
       'Industry_level_5_num_mapped', 'Industry_level_6_num_mapped', 'ID_BB_UNIQUE', 'ID_BB_GLOBAL']
cols_excluded = ['Year','REVISION_ID','ROW_NUMBER','FLAG','SECURITY_DESCRIPTION','RCODE','NFIELDS','EQY_FUND_CRNCY']
cols_excluded.extend(cat_cols)
selected_features = [item for item in feature_cols if item not in cols_excluded]

final_df_list = []

for idx, sub_df in df.groupby(['Year','Quarter']):
    year, quarter = idx[0], idx[1]
    for grp in industry_groupings:
        sub_df = create_industry_stats_for_year(sub_df, selected_features, grp, year)
    final_df_list.append(sub_df)

df_w_industry_stats = pd.concat(final_df_list)

df_w_industry_stats.to_csv(save_file_path, index=False)


## Create industry sum
final_df_list = []
y_cols = ['SALES_REV_TURN','CF_CASH_FROM_OPER','EBITDA','ARD_CAPITAL_EXPENDITURES']

for idx, sub_df in df.groupby(['Year','Quarter']):
    year, quarter = idx[0], idx[1]
    for feature in y_cols:
        sub_df[feature + "_industry_sum"] = sub_df[feature].sum()

    final_df_list.append(sub_df)

final_df = pd.concat(final_df_list)