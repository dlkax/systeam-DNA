from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import os
import json
import sqlite3
from datetime import datetime

app = Flask(__name__)
CORS(app)

MODEL_PATH = 'saved_model/fossil_classifier.h5'  

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def load_model():
    global model
    try:
        model = keras.models.load_model(MODEL_PATH)
        print("Modelo carregado com sucesso!")
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
   
    IMG_SIZE = (224, 224)  
  
    if image.mode != 'RGB':
        image = image.convert('RGB')
 
    image = image.resize(IMG_SIZE)
    
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    
    img_array = img_array.astype('float32') / 255.0
    
    return img_array

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head><title>PLA API</title></head>
    <body>
        <h1>Pleistocene Life Analyzer API</h1>
        <p>API funcionando! Acesse <a href="/web">interface web</a></p>
    </body>
    </html>
    """

@app.route('/web')
def web_interface():

    return open('interface.html').read()

@app.route('/analyze', methods=['POST'])
def analyze_fossil():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Nenhum arquivo enviado'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Nenhum arquivo selecionado'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Tipo de arquivo não permitido'}), 400
        
        if model is None:
            return jsonify({'error': 'Modelo não carregado'}), 500
        

        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(image)
        

        predictions = model.predict(processed_image)
        

        class_names = [
            'mammuthus_primigenius',
            'smilodon_fatalis', 
            'triceratops_horridus'
        ]
        

        results = []
        for i, confidence in enumerate(predictions[0]):
            results.append({
                'species': class_names[i],
                'confidence': float(confidence * 100)
            })
        

        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        save_result(file.filename, main_prediction, results)

        return jsonify({
    'success': True,
    'predictions': results,
    'main_prediction': main_prediction
})
    
    except Exception as e:
        return jsonify({'error': f'Erro na análise: {str(e)}'}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None
    })


@app.route('/history', methods=['GET'])
def get_history():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute("SELECT id, filename, species, confidence, predictions, created_at FROM analyses ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()

    history = []
    for row in rows:
        history.append({
            "id": row[0],
            "filename": row[1],
            "species": row[2],
            "confidence": row[3],
            "predictions": json.loads(row[4]),
            "created_at": row[5]
        })

    return jsonify(history)


if __name__ == '__main__':
    init_db()
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)





'''/ ----------------------DATABASE-------------------------------- /'''


def init_db():
    conn - sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            species TEXT,
            confidence REAL,
            predictions TEXT,
            created_at TEXT
        )   
    """)

def save_result(filename, main_prediction, predictions):
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO analyses (filename, species, confidence, predictions, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (
        filename, 
        main_prediction["species"],
        main_prediction["confidence"],
        json.dumps(predictions, ensure_ascii=False),
        datetime.now().isoformat()
    ))

    conn.comit()
    conn.close()