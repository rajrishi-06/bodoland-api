from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

# 1. Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# 2. Load trained pipeline
pipeline = joblib.load('pipeline.joblib')
transformer = pipeline['transformer']
model = pipeline['model']

# 3. Define prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Expect JSON: {"data": [{...}, {...}]}
        payload = request.get_json()
        records = payload.get('data')
        # Convert to DataFrame
        import pandas as pd
        df = pd.DataFrame(records)

        # Preprocess and predict
        X_t = transformer.transform(df)
        preds = model.predict(X_t)
        probs = model.predict_proba(X_t)[:, 1]

        # Build response
        results = []
        for pred, prob in zip(preds, probs):
            results.append({ 'prediction': int(pred), 'probability': float(prob) })

        return jsonify({ 'status': 'success', 'results': results })
    except Exception as e:
        return jsonify({ 'status': 'error', 'message': str(e) }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)