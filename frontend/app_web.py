from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import os
import json

app = Flask(__name__)
CORS(app)


import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
MODEL_PATH = os.path.join(parent_dir, 'saved_model', 'fossil_classifier.h5')


if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(current_dir, 'saved_model', 'fossil_classifier.h5')

if not os.path.exists(MODEL_PATH):
    MODEL_PATH = 'fossil_classifier.h5'

model = None


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def load_model():
    global model
    try:
        print(f"Tentando carregar modelo de: {MODEL_PATH}")
        
        search_paths = [
            MODEL_PATH,
            os.path.join(parent_dir, 'saved_model', 'fossil_classifier.h5'),
            os.path.join(current_dir, 'saved_model', 'fossil_classifier.h5'),
            'fossil_classifier.h5',
            'best_model.h5'
        ]

        found_path = None
        for path in search_paths:
            if os.path.exists(path):
                found_path = path
                break

        if not found_path:
            print("❌ Modelo não encontrado em nenhum local!")
            for path in search_paths:
                print(f"  - {os.path.abspath(path)} {'(EXISTE)' if os.path.exists(path) else '(NÃO EXISTE)'}")
            return False

        print(f"✅ Modelo encontrado em: {found_path}")
        model = keras.models.load_model(found_path)
        return True

    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        model = None
        return False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):

    IMG_SIZE = (128, 128)

    if image.mode != 'RGB':
        image = image.convert('RGB')
  
    image = image.resize(IMG_SIZE)

    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)

    img_array = img_array.astype('float32') / 255.0
    
    print(f"Shape da imagem processada: {img_array.shape}")
    
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
    return '''
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pleistocene Life Analyzer</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background-color: #f5f5f5;
            padding: 20px;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }
        
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 30px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .header h1 {
            color: #2c3e50;
            margin-bottom: 10px;
            font-weight: 600;
            font-size: 2.5rem;
        }
        
        .header p {
            color: #666;
            font-size: 1.1rem;
            font-weight: 400;
        }
        
        .upload-area {
            border: 2px dashed #3498db;
            border-radius: 8px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            margin-bottom: 20px;
            transition: all 0.3s ease;
            background-color: #fafafa;
        }
        
        .upload-area p {
            margin: 8px 0;
            font-weight: 500;
        }
        
        .upload-area p:first-child {
            font-size: 1.1rem;
            color: #2c3e50;
        }
        
        .upload-area p:last-child {
            font-size: 0.9rem;
            color: #7f8c8d;
        }
        
        .upload-area:hover {
            border-color: #2980b9;
            background-color: #f8f9fa;
        }
        
        .upload-area.dragover {
            border-color: #27ae60;
            background-color: #d5f4e6;
        }
        
        .file-input {
            display: none;
        }
        
        .btn {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            margin: 10px;
            transition: background-color 0.3s;
        }
        
        .btn:hover {
            background-color: #2980b9;
        }
        
        .btn:disabled {
            background-color: #bdc3c7;
            cursor: not-allowed;
        }
        
        .preview {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        
        .preview img {
            max-width: 300px;
            max-height: 200px;
            border-radius: 4px;
            border: 1px solid #ddd;
        }
        
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        
        .loader {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #3498db;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .results {
            display: none;
            margin-top: 30px;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 8px;
        }
        
        .result-item {
            background: white;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
            border-left: 4px solid #3498db;
        }
        
        .result-main {
            border-left-color: #27ae60;
        }
        
        .species-name {
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }
        
        .confidence {
            color: #27ae60;
            font-weight: bold;
        }
        
        .error {
            display: none;
            background-color: #e74c3c;
            color: white;
            padding: 15px;
            border-radius: 4px;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🦕 Pleistocene Life Analyzer</h1>
            <p>Identifique fósseis usando Inteligência Artificial</p>
        </div>
        
        <div class="upload-area" onclick="document.getElementById('fileInput').click()">
            <p><strong>Clique aqui ou arraste uma imagem</strong></p>
            <p>Formatos aceitos: JPG, PNG, GIF</p>
            <input type="file" id="fileInput" class="file-input" accept="image/*">
        </div>
        
        <div class="preview" id="preview">
            <img id="previewImg" alt="Preview">
            <br>
            <button class="btn" onclick="analyzeImage()">🔬 Analisar Fóssil</button>
        </div>
        
        <div class="loading" id="loading">
            <div class="loader"></div>
            <p>Analisando... Aguarde alguns segundos</p>
        </div>
        
        <div class="error" id="error"></div>
        
        <div class="results" id="results">
            <h3>Resultados da Análise</h3>
            <div id="resultsList"></div>
        </div>
    </div>

    <script>
        let selectedFile = null;
        
        const uploadArea = document.querySelector('.upload-area');
        const fileInput = document.getElementById('fileInput');
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        });
        
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });
        
        function handleFile(file) {
            if (!file.type.startsWith('image/')) {
                showError('Por favor, selecione apenas arquivos de imagem.');
                return;
            }
            
            if (file.size > 10 * 1024 * 1024) {
                showError('Arquivo muito grande. Máximo: 10MB');
                return;
            }
            
            selectedFile = file;
            
            const reader = new FileReader();
            reader.onload = (e) => {
                document.getElementById('previewImg').src = e.target.result;
                document.getElementById('preview').style.display = 'block';
                hideResults();
            };
            reader.readAsDataURL(file);
        }
        
        async function analyzeImage() {
            if (!selectedFile) {
                showError('Selecione uma imagem primeiro.');
                return;
            }
            
            showLoading();
            hideError();
            
            const formData = new FormData();
            formData.append('file', selectedFile);
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.success) {
                    showResults(data.predictions);
                } else {
                    showError(data.error || 'Erro na análise');
                }
            } catch (error) {
                showError('Erro de conexão. Verifique se o servidor está rodando.');
                console.error('Erro:', error);
            }
            
            hideLoading();
        }
        
        function showResults(predictions) {
            const resultsList = document.getElementById('resultsList');
            let html = '';
            
            predictions.forEach((pred, index) => {
                const isMain = index === 0;
                const className = isMain ? 'result-item result-main' : 'result-item';
                
                html += `
                    <div class="${className}">
                        <div class="species-name">${formatSpeciesName(pred.species)}</div>
                        <div class="confidence">Confiança: ${pred.confidence.toFixed(1)}%</div>
                        ${isMain ? '<small>Identificação mais provável</small>' : ''}
                    </div>
                `;
            });
            
            resultsList.innerHTML = html;
            document.getElementById('results').style.display = 'block';
        }
        
        function formatSpeciesName(species) {
            const names = {
                'mammuthus_primigenius': 'Mammuthus primigenius (Mamute Lanoso)',
                'smilodon_fatalis': 'Smilodon fatalis (Tigre Dente-de-sabre)',
                'triceratops_horridus': 'Triceratops horridus'
            };
            
            return names[species] || species.replace('_', ' ');
        }
        
        function showLoading() {
            document.getElementById('loading').style.display = 'block';
        }
        
        function hideLoading() {
            document.getElementById('loading').style.display = 'none';
        }
        
        function showError(message) {
            const errorDiv = document.getElementById('error');
            errorDiv.textContent = message;
            errorDiv.style.display = 'block';
        }
        
        function hideError() {
            document.getElementById('error').style.display = 'none';
        }
        
        function hideResults() {
            document.getElementById('results').style.display = 'none';
        }
    </script>
</body>
</html>
    '''

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

        try:

            labels_paths = [
                os.path.join(parent_dir, 'model', 'labels.json'),
                os.path.join(current_dir, 'model', 'labels.json'),
                'model/labels.json'
            ]
            
            labels_data = None
            for path in labels_paths:
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as f:
                        labels_data = json.load(f)
                    break
            
            if labels_data:
                class_names = [labels_data['id_to_name'][str(i)] for i in range(labels_data['num_classes'])]
            else:
                raise FileNotFoundError("labels.json não encontrado")
                
        except Exception as e:
            print(f"Aviso: Não foi possível carregar labels.json ({e}). Usando classes padrão.")

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
        
        return jsonify({
            'success': True,
            'predictions': results,
            'main_prediction': results[0]
        })
    
    except Exception as e:
        return jsonify({'error': f'Erro na análise: {str(e)}'}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    print("Iniciando Pleistocene Life Analyzer...")
    print("Carregando modelo...")
    
    if load_model():
        print("✅ Modelo carregado com sucesso!")
        print("🚀 Servidor iniciando em http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Erro ao carregar modelo. Verifique o caminho do arquivo.")
        print(f"Procurando por: {MODEL_PATH}")
        print("Certifique-se de que o arquivo existe.")