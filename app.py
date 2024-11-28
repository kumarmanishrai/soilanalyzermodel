from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)



@app.route('/predict/crop', methods=['POST'])
def predict_crop():
    try:
        data = request.json
        print(data.get('nitrogen'))
        nitrogen = data.get("nitrogen", 0)
        phosphorus = data.get("phosphorus", 0)
        potassium = data.get("potassium", 0)
        ph = data.get("ph", 0)
        moisture = data.get("moisture", 0)
        temperature = data.get("temperature", 0)
        humidity = data.get("humidity", 0)
        

        input_data = [nitrogen, phosphorus, potassium, ph, moisture, temperature, humidity]

        if ph > 0 and humidity > 0 and temperature <100:
            joblib.load('crop_prediction_model_large_data', 'r')
            model = joblib.load(open('crop_prediction_model_large_data', 'rb'))
            arr = [input_data]

            crop_name = model.predict(arr)

            print(crop_name[0])
            return jsonify({"predicted_crop": crop_name[0]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500



fertilizer_model = joblib.load('fertilizer_prediction_model_large_data.pkl')

fertilizer_encoder = joblib.load('fertilizer_prediction_model_large_data_encoder.pkl')


@app.route('/predict/fertilizer', methods=['POST'])
def predict_fertilizer():   
    try:
        data = request.json
        Crop = data.get('Crop')
        temperature = data.get('temperature')
        humidity = data.get('humidity')
        moisture = data.get('moisture')

        encoded_crop = fertilizer_encoder.transform([Crop])[0]

        input_data = pd.DataFrame({
            'Crop': [encoded_crop],
            'temperature': [temperature],
            'humidity': [humidity],
            'moisture': [moisture]
        })

        predicted_values = fertilizer_model.predict(input_data)

        n, p, k, ph = predicted_values[0]

        return jsonify({
            'N': n,
            'P': p,
            'K': k,
            'pH': ph
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500





if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=os.environ.get('PORT') or 5500)















