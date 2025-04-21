import pandas as pd

file_path = '/home/siyi/PycharmProjects/AutoML-ForecastingPipeline/data/external_data/raw/s&p_estimates.xlsx'
save_file_path = '/home/siyi/PycharmProjects/AutoML-ForecastingPipeline/data/external_data/processed/processed_s&p_estimates.csv'

df_list = []
company_list = ['xom','cop','cvx']

for company_name in company_list:
    sub_df = pd.read_excel(file_path, sheet_name='xom')

    sub_df['FQ3 2000 - Sep 2000'] = sub_df['FQ3 2000 - Sep 2000'].str.replace(' E', '', regex=True)  # Remove 'E'
    sub_df['FQ3 2000 - Sep 2000'] = sub_df['FQ3 2000 - Sep 2000'].str.replace(',', '', regex=True)   # Remove comma
    sub_df['FQ3 2000 - Sep 2000'] = sub_df['FQ3 2000 - Sep 2000'].astype(float)

    for col in sub_df.columns:
        if col != 'Company Level ($M)' and sub_df[col].dtype == 'object':
            sub_df[col] = sub_df[col].str.replace(' E', '', regex=True)  # Remove 'E'
            sub_df[col] = sub_df[col].str.replace(',', '', regex=True)   # Remove comma
            sub_df[col] = sub_df[col].astype(float)

    sub_df = sub_df.set_index('Company Level ($M)')

    converted_df = sub_df.T
    if company_name == 'xom':
        converted_df['ID_BB_UNIQUE'] = 'EQ0010054600001000'
    elif company_name == 'cop':
        converted_df['ID_BB_UNIQUE'] = 'EQ0010117400001000'
    elif company_name == 'cvx':
        converted_df['ID_BB_UNIQUE'] = 'EQ0010031500001000'

    converted_df.reset_index(inplace=True)

    converted_df[['Quarter', 'Year']] = converted_df['index'].str.extract(r'FQ(\d+) (\d{4})')
    converted_df = converted_df.dropna(subset=['Quarter', 'Year'])
    converted_df['Quarter'] = converted_df['Quarter'].astype(int)
    converted_df['Year'] = converted_df['Year'].astype(int)

    converted_df = converted_df.drop(columns='index')

    df_list.append(converted_df)



concat_df = pd.concat(df_list, ignore_index=True)

for id, sub_df in concat_df.groupby('ID_BB_UNIQUE'):
    print(id)
    print(len(sub_df))

concat_df.columns = [f"s&p_estimate_{col}" for col in concat_df.columns]
concat_df.columns = concat_df.columns.str.replace(' ', '_')
concat_df.columns = concat_df.columns.str.replace(r'[^a-zA-Z0-9_.()]', '_', regex=True)

concat_df.to_csv(save_file_path, index=False)