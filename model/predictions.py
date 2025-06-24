from model.transformer_model import TransformerModel
import torch
import joblib
import numpy as np
import pandas as pd

def init_model(model_path='model/model.pth', scaler_path='model/scaler.pkl'):
    input_size = 2
    hidden_size = 10
    extra_size = 2
    output_size = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransformerModel(input_size, hidden_size, extra_size, output_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    scaler = joblib.load(scaler_path)

    return model, scaler, device

def forecast_future_with_context(
    total_orders_by_date,
    total_users_by_date,
    scaler,
    model=None,
    device=None,
    sequence_length=200,
    forecast_days=180
):
    if model is None or device is None:
        model, scaler_loaded, device = init_model()
        scaler = scaler or scaler_loaded

    orders_df = pd.DataFrame(total_orders_by_date)
    users_df = pd.DataFrame(total_users_by_date)

    latest_order_date = pd.to_datetime(orders_df['date']).max()
    latest_user_date = pd.to_datetime(users_df['date']).max()
    last_date_in_data = min(latest_order_date, latest_user_date)

    context_days = sequence_length - forecast_days
    if context_days < 1:
        raise ValueError("sequence_length must be >= forecast_days + 1")

    total_days = context_days + forecast_days

    future_dates = pd.date_range(start=last_date_in_data + pd.Timedelta(days=1), periods=forecast_days)
    future_dates = future_dates.strftime('%Y-%m-%d').tolist()

    final_df = pd.merge(users_df, orders_df, on='date')
    final_df = final_df.sort_values('date')
    final_df['day'] = pd.to_datetime(final_df['date']).dt.day
    final_df['month'] = pd.to_datetime(final_df['date']).dt.month

    temp_df = final_df[['number_of_users', 'day', 'month', 'number_of_orders']].copy()
    temp_scaled = scaler.transform(temp_df)

    full_sequence_scaled = temp_scaled[-sequence_length:, [0, 3]]

    output_sequence = []

    date_range_context = pd.date_range(end=last_date_in_data, periods=context_days)
    date_range_context = date_range_context.strftime('%Y-%m-%d').tolist()

    for i, date_str in enumerate(date_range_context):
        real_value_scaled = full_sequence_scaled[i, 1]
        output_sequence.append({
            'target_date': date_str,
            'predicted_orders_scaled': real_value_scaled,
            'is_predicted': False
        })

    sequence_scaled = full_sequence_scaled[-sequence_length:, :]

    last_scaled_users = temp_scaled[-1, 0]

    for i, target_date_str in enumerate(future_dates):
        target_date = pd.to_datetime(target_date_str)

        target_day = target_date.day
        target_month = target_date.month
        extra_scaled = scaler.transform([[0, target_day, target_month, 0]])[:, [1, 2]]
        x_extra_tensor = torch.tensor(extra_scaled, dtype=torch.float32)

        x_seq_tensor = torch.tensor(sequence_scaled[-sequence_length:, :], dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            y_pred_scaled = model(x_seq_tensor.to(device), x_extra_tensor.to(device)).item()

        output_sequence.append({
            'target_date': target_date_str,
            'predicted_orders_scaled': y_pred_scaled,
            'is_predicted': True
        })

        new_entry = np.array([last_scaled_users, y_pred_scaled])
        sequence_scaled = np.vstack([sequence_scaled[1:], new_entry])

    return output_sequence
