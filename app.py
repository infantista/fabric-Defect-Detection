from flask import Flask, render_template, request, jsonify
from predict import FabricDefectDetector
import os
import uuid

app = Flask(__name__)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fabric_defect.h5")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "temp_uploads")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

try:
    print(f"Loading model from: {MODEL_PATH}")
    detector = FabricDefectDetector(model_path=MODEL_PATH)
except Exception as e:
    print(f"Failed to initialize detector: {str(e)}")
    raise

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Allowed file types are png, jpg, jpeg'}), 400

    file_ext = os.path.splitext(file.filename)[1]
    temp_filename = f"{uuid.uuid4()}{file_ext}"
    temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)

    try:
        file.save(temp_path)
        result = detector.predict(temp_path)
        
        if result['error']:
            return jsonify({'error': result['error']}), 500
            
        return jsonify({
            'status': result['status'],
            'confidence': f"{result['confidence']:.2%}",
            'filename': file.filename
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)