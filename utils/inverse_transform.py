import numpy as np

def pad_for_inverse_transform(pred_column, scaler, target_index=2):
    padded = np.zeros((len(pred_column), scaler.scale_.shape[0]))
    padded[:, target_index] = pred_column.flatten()
    return padded

def inverse_transform_predictions(y_preds, scaler, target_index=2):
    scaled_values = np.array([p['predicted_orders_scaled'] for p in y_preds])
    padded = pad_for_inverse_transform(scaled_values, scaler, target_index)
    inversed = scaler.inverse_transform(padded)[:, target_index]
    return inversed
