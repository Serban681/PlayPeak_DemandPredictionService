import pandas as pd
import torch

def build_model_inputs(total_orders_by_date, total_users_by_date, scaler, sequence_length=200, months_forward=6):
    orders_df = pd.DataFrame(total_orders_by_date)
    users_df = pd.DataFrame(total_users_by_date)

    # Determine last date in data
    latest_order_date = pd.to_datetime(orders_df['date']).max()
    latest_user_date = pd.to_datetime(users_df['date']).max()
    last_date_in_data = min(latest_order_date, latest_user_date)

    # Define target_date: X months forward
    target_date = last_date_in_data + pd.DateOffset(months=months_forward)

    # Generate date sequence (last N days up to LAST DATE IN DATA — NOT target date!)
    def generate_date_sequence(last_date_in_data, sequence_length):
        start_date = last_date_in_data - pd.Timedelta(days=sequence_length - 1)
        date_range = pd.date_range(start=start_date, end=last_date_in_data)
        return date_range.strftime('%Y-%m-%d').tolist()

    date_sequence = generate_date_sequence(last_date_in_data, sequence_length)

    # Filter and sort data
    orders_df = orders_df[orders_df['date'].isin(date_sequence)].sort_values('date')
    users_df = users_df[users_df['date'].isin(date_sequence)].sort_values('date')

    # Merge
    final_df = pd.merge(users_df, orders_df, on='date')
    final_df['day'] = pd.to_datetime(final_df['date']).dt.day
    final_df['month'] = pd.to_datetime(final_df['date']).dt.month

    # Scale sequence
    temp_df = final_df[['number_of_users', 'day', 'month', 'number_of_orders']].copy()
    temp_scaled = scaler.transform(temp_df)

    # Sequence part
    sequence_scaled = temp_scaled[:, [0, 3]]

    # Extra features based on target_date (future!)
    target_day = target_date.day
    target_month = target_date.month
    extra_scaled = scaler.transform([[0, target_day, target_month, 0]])[:, [1, 2]]

    # Convert to tensors
    x_seq_tensor = torch.tensor(sequence_scaled, dtype=torch.float32).unsqueeze(0)
    x_extra_tensor = torch.tensor(extra_scaled, dtype=torch.float32)

    return x_seq_tensor, x_extra_tensor, target_date.strftime('%Y-%m-%d')
