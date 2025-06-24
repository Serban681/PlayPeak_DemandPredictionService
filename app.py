from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
import os
from data_processing.extract import process_user_data, process_order_data
from data_processing.insights import process_orders_by_date, process_total_users_by_date
from model.predictions import forecast_future_with_context
from utils.inverse_transform import inverse_transform_predictions
import joblib

load_dotenv

app = Flask(__name__)
CORS(app)
DB_API_URL = os.getenv('DB_API_URL')

@app.route('/generate-demand-prediction', methods=['POST'])
def add_order_data():
    data = request.get_json()

    total_users_by_date = process_total_users_by_date(process_user_data(data['users']))
    total_orders_by_date = process_orders_by_date(process_order_data(data['orders'], data['productVarianceId']))

    scaler = joblib.load('model/scaler.pkl')

    y_preds = forecast_future_with_context(
        total_orders_by_date, total_users_by_date, scaler, 
        forecast_days=data['daysToPredict'], 
        sequence_length=data['totalDays']
    )

    inversed_data = inverse_transform_predictions(y_preds, scaler, target_index=2)

    dates = [p['target_date'] for p in y_preds]

    predictions = [
        {
            'date': date,
            'noOfOrders': float(value),
            'isPredicted': p['is_predicted']
        }
        for date, value, p in zip(dates, inversed_data, y_preds)
    ]
    
    productVarianceDemand = {
        'productVarianceId': data['productVarianceId'],
        'demandPredictions': predictions
    }
    
    return jsonify(productVarianceDemand), 200

if __name__ == '__main__':
    app.run(debug=True)
