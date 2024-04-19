import os
import pandas as pd
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from flask import Flask,request,jsonify
import joblib

def load_model(filename):
    # Load the model from disk
    return joblib.load(filename)

def predict_energy_consumption(model, X):
    # Predict energy consumption using the trained model
    return model.predict(X)

app = Flask(__name__)
@app.route('/', methods=['POST'])
def receive_data():
    try:
        data = request.get_json()
        columns = ['Temperature', 'Humidity', 'SquareFootage', 'Occupancy', 'HVACUsage', 
           'LightingUsage', 'RenewableEnergy', 'DayOfWeek', 'Holiday']
        values = list(data.values())
        df = pd.DataFrame([values], columns=columns)
        csv_file_path = 'data.csv'
        with open(csv_file_path, 'a+') as f:
            for index, row in df.iterrows():
                row_values = ','.join(map(str, row.values))
                f.write(row_values)
                f.write('\n')

        df1 = pd.read_csv('data.csv')
        print(df1.info()) 
        df_encoded = pd.get_dummies(data=df1)
        print(df_encoded)
        loaded_rf_model = load_model("random_forest_model.pkl")
        
        example_predictions = predict_energy_consumption(loaded_rf_model,df_encoded)
        print(example_predictions)
        last_value = example_predictions[-1]
        print(last_value)
        response_data = {
        "pakage": last_value}
        return jsonify(response_data)
    except Exception as e:
        # Handle any errors that occurred during processing
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)