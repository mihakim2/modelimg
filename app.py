from flask import Flask, request, jsonify
from make_pred import pred

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'no file'}), 400

    file = request.files['file'].read()
    prediction = pred(file)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)