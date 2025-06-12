from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('house_price_pridiction_by_lincoln_chatikobo.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Define the expected input fields
        input_fields = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm',
                        'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']
        features = []

        # Validate and convert inputs to float
        for field in input_fields:
            value = request.form.get(field)
            if value is None or value.strip() == '':
                return render_template('index.html', prediction_text="⚠️ Please fill in all fields.")
            features.append(float(value))

        # Convert to numpy array for prediction
        features_array = np.array([features])
        prediction = model.predict(features_array)[0]
        output = round(prediction, 2)

        return render_template('index.html', prediction_text=f"Predicted House Price: ${output}")
    
    except ValueError:
        return render_template('index.html', prediction_text="⚠️ Invalid input. Please enter numeric values only.")

if __name__ == "__main__":
    app.run(debug=True)
# Ensure the model file exists before running the app