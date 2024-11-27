from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)



@app.route('/predict/crop', methods=['POST'])
def predict_crop():
    try:
        data = request.json
        print(data.get('nitrogen'))
        nitrogen = data.get("nitrogen", 0)
        phosphorus = data.get("phosphorus", 0)
        potassium = data.get("potassium", 0)
        humidity = data.get("humidity", 0)
        temperature = data.get("temperature", 0)
        ph = data.get("ph", 0)
        rainfall = data.get("rainfall", 0)

        input_data = [nitrogen, phosphorus, potassium, humidity, temperature, ph, rainfall]

        if ph > 0 and humidity > 0 and temperature <100:
            joblib.load('crop_prediction_model', 'r')
            model = joblib.load(open('crop_prediction_model', 'rb'))
            arr = [input_data]

            crop_name = model.predict(arr)

            print(crop_name[0])
            return jsonify({"predicted_crop": crop_name[0]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5500)
