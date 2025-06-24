import pandas as pd
from utils.file_generation import write_df_to_csv

def get_total_orders_on_date(orders_df, orderDate):
    return len(orders_df[orders_df['orderDate'] == orderDate])

def process_no_of_orders_by_date_and_store():
    orders_df = pd.read_csv('./data/processed_order_data.csv')

    total_orders_by_day_df = pd.DataFrame(columns=['date', 'number_of_orders'])

    for index, row in orders_df.iterrows():
        total_orders_by_day_df = pd.concat([
            total_orders_by_day_df, 
            pd.DataFrame(data={'date': row['orderDate'], 'number_of_orders': [get_total_orders_on_date(orders_df, row['orderDate'])]})
        ])

    write_df_to_csv(total_orders_by_day_df, './data/no_of_orders_by_day.csv')
