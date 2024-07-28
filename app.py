import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify, current_app
from flasgger import Swagger
import logging
from sklearn.metrics import accuracy_score

app = Flask(__name__)
swagger = Swagger(app)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

model_file = os.path.join('trained-model', 'chemato-weather-model.pkl')
preprocessor_file = os.path.join('trained-model', 'preprocessor.pkl')

logging.debug(f'Loading model from {model_file}')
model = joblib.load(model_file)
logging.debug(f'Loading preprocessor from {preprocessor_file}')
preprocessor = joblib.load(preprocessor_file)

# Define the column names
columns_needed = ['lux', 'temp', 'humid']

@app.route('/transform', methods=['POST'])
def transform():
    try:
        data = request.get_json()
        logging.debug(f'Received data for transformation: {data}')
        
        # Ensure data is in the correct format
        if 'features' not in data or not isinstance(data['features'], list):
            logging.error('Input data should be a dictionary with a key "features" containing a list of lists')
            return jsonify({'error': 'Input data should be a dictionary with a key "features" containing a list of lists'}), 400
        
        # Convert the data to a DataFrame
        new_data = pd.DataFrame(data['features'], columns=columns_needed)
        logging.debug(f'Converted data to DataFrame: {new_data}')
        
        # Reorder columns to match the expected order
        new_data = new_data[['lux', 'temp', 'humid']]
        logging.debug(f'Data with all required columns: {new_data}')
        
        # Preprocess the data
        new_data_preprocessed = preprocessor.transform(new_data)
        logging.debug(f'Preprocessed data: {new_data_preprocessed}')
        
        # Convert transformed data to list for JSON response
        transformed_data_list = new_data_preprocessed.tolist()
        
        return jsonify({'transformed_features': transformed_data_list})
    except Exception as e:
        logging.error(f'Error during transformation: {e}')
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        logging.debug(f'Received data: {data}')
        
        # Ensure data is in the correct format
        if 'features' not in data or not isinstance(data['features'], list):
            logging.error('Input data should be a dictionary with a key "features" containing a list of lists')
            return jsonify({'error': 'Input data should be a dictionary with a key "features" containing a list of lists'}), 400
        
        # Transform the data
        new_data = pd.DataFrame(data['features'], columns=columns_needed)
        logging.debug(f'Converted data to DataFrame: {new_data}')
        
        new_data_preprocessed = preprocessor.transform(new_data)
        logging.debug(f'Preprocessed data: {new_data_preprocessed}')
        
        # Predict using the model
        prediction = model.predict(new_data_preprocessed)
        logging.debug(f'Model prediction: {prediction}')
        
        pred_map = {
            0: 'Kurang',
            1: 'Cukup',
            2: 'Baik'
        }
        
        predicted_label = pred_map.get(prediction[0])
        logging.debug(f'Predicted label: {predicted_label}')
        
        return jsonify({'prediction': predicted_label})
    except Exception as e:
        logging.error(f'Error during prediction: {e}')
        return jsonify({'error': str(e)}), 500



@app.route('/accuracy', methods=['POST'])
def accuracy():
    try:
        data = request.get_json()
        logging.debug(f'Received data for accuracy calculation: {data}')
        
        # Ensure data is in the correct format
        if 'features' not in data or 'labels' not in data or not isinstance(data['features'], list) or not isinstance(data['labels'], list):
            logging.error('Input data should be a dictionary with keys "features" and "labels" containing lists of lists and lists respectively')
            return jsonify({'error': 'Input data should be a dictionary with keys "features" and "labels" containing lists of lists and lists respectively'}), 400
        
        # Convert the data to a DataFrame
        new_data = pd.DataFrame(data['features'], columns=columns_needed)
        logging.debug(f'Converted data to DataFrame: {new_data}')
        
        # Transform the data
        new_data_preprocessed = preprocessor.transform(new_data)
        logging.debug(f'Preprocessed data: {new_data_preprocessed}')
        
        # Predict using the model
        predictions = model.predict(new_data_preprocessed)
        logging.debug(f'Model predictions: {predictions}')
        
        # Calculate accuracy
        accuracy = accuracy_score(data['labels'], predictions)
        logging.debug(f'Calculated accuracy: {accuracy}')
        
        return jsonify({'accuracy': accuracy})
    except Exception as e:
        logging.error(f'Error during accuracy calculation: {e}')
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Run the app on Google Cloud App Engine
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))