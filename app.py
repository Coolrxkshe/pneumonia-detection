from flask import Flask, render_template, request
from predict import predict_image
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    file = request.files['xray']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    result = predict_image(filepath)
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
