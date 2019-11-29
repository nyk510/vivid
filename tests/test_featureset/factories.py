import pandas as pd


def generate_price_dataframe():
    out_df = pd.DataFrame([
        [1, 2, 1, 'foo'],
        [2, 2, 1, 'hoge'],
        [3, 1, 2, 'bar'],
    ], columns=['id', 'price', 'company_id', 'name'])

    return out_df
