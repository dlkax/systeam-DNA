"""
Sistema de Predição de Fósseis com Alta Confiança
Versão 3.0 - Com melhorias para aumentar a precisão
"""

import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os
from pathlib import Path
import cv2
from typing import Dict, Tuple, List

# Caminhos
MODEL_PATH = "saved_model/fossil_classifier.h5"
LABELS_PATH = "model/labels.json"

# Configurações de predição
IMG_SIZE = (128, 128)
CONFIDENCE_THRESHOLD = 70.0  # Limiar mínimo de confiança
TOP_K_PREDICTIONS = 3  # Mostrar top 3 predições

class FossilPredictor:
    """Classe aprimorada para predição de espécies fósseis"""
    
    def __init__(self):
        """Inicializa o preditor carregando modelo e labels"""
        self.model = None
        self.labels = None
        self.species_info = None
        self._load_model()
        self._load_labels()
        self._load_species_info()
        
    def _load_model(self):
        """Carrega o modelo treinado"""
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"❌ Modelo não encontrado: {MODEL_PATH}")
        
        print("📂 Carregando modelo...")
        self.model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Modelo carregado com sucesso!")
        
    def _load_labels(self):
        """Carrega o mapeamento de labels"""
        if not os.path.exists(LABELS_PATH):
            raise FileNotFoundError(f"❌ Labels não encontrados: {LABELS_PATH}")
        
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Suporta diferentes formatos de labels.json
        if isinstance(data, dict):
            if "id_to_name" in data:
                self.labels = data["id_to_name"]
            elif "0" in data or 0 in data:
                self.labels = data
            else:
                self.labels = {str(i): name for i, name in enumerate(data.keys())}
        else:
            self.labels = {str(i): name for i, name in enumerate(data)}
            
        print(f"✅ {len(self.labels)} classes carregadas")
        
    def _load_species_info(self):
        """Carrega informações detalhadas das espécies"""
        self.species_info = {
            "mammuthus_primigenius": {
                "nome_comum": "Mamute-lanoso",
                "periodo": "Pleistoceno (2.5 milhões - 10.000 anos atrás)",
                "descricao": "Mamífero proboscídeo extinto, adaptado ao clima frio com pelagem espessa. Parente dos elefantes modernos.",
                "habitat": "Tundra e estepes do hemisfério norte",
                "tamanho": "3-4 metros de altura, 6 toneladas"
            },
            "smilodon": {
                "nome_comum": "Tigre-dentes-de-sabre",
                "periodo": "Pleistoceno (2.5 milhões - 10.000 anos atrás)",
                "descricao": "Felino pré-histórico com caninos de até 28cm, predador apex de sua época.",
                "habitat": "Américas do Norte e Sul",
                "tamanho": "1-1.2m de altura, 160-280kg"
            },
            "smilodon_fatalis": {
                "nome_comum": "Tigre-dentes-de-sabre americano",
                "periodo": "Pleistoceno",
                "descricao": "Subespécie de Smilodon encontrada principalmente na América do Norte. Extremamente musculoso.",
                "habitat": "Florestas e savanas da América do Norte",
                "tamanho": "1m de altura, até 280kg"
            },
            "megatherium": {
                "nome_comum": "Preguiça-gigante",
                "periodo": "Plioceno ao Pleistoceno",
                "descricao": "Mamífero gigante herbívoro, uma das maiores preguiças terrestres que já existiram.",
                "habitat": "América do Sul",
                "tamanho": "6m de comprimento, 4 toneladas"
            },
            "preguica_gigante": {
                "nome_comum": "Preguiça-gigante",
                "periodo": "Pleistoceno",
                "descricao": "Megafauna herbívora terrestre, ancestral das preguiças modernas mas de proporções gigantescas.",
                "habitat": "Florestas e campos da América do Sul",
                "tamanho": "Até 6m de comprimento"
            }
        }
    
    def preprocess_image_advanced(self, img_path: str) -> np.ndarray:
        """
        Preprocessamento avançado da imagem para melhor predição
        """
        # Carregar imagem com OpenCV para preprocessamento avançado
        img_cv = cv2.imread(img_path)
        if img_cv is None:
            raise ValueError(f"Não foi possível carregar a imagem: {img_path}")
        
        # Converter BGR para RGB
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        
        # Aplicar CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # para melhorar o contraste
        lab = cv2.cvtColor(img_cv, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        img_enhanced = cv2.merge([l, a, b])
        img_enhanced = cv2.cvtColor(img_enhanced, cv2.COLOR_LAB2RGB)
        
        # Redimensionar mantendo aspect ratio
        img_resized = self._resize_with_padding(img_enhanced, IMG_SIZE)
        
        # Normalizar
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Expandir dimensões para batch
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch
    
    def _resize_with_padding(self, img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Redimensiona mantendo aspect ratio e adiciona padding se necessário"""
        h, w = img.shape[:2]
        target_h, target_w = target_size
        
        # Calcular scale mantendo aspect ratio
        scale = min(target_w/w, target_h/h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Redimensionar
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Criar imagem com padding
        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        padded[:] = [128, 128, 128]  # Cor cinza para padding
        
        # Centralizar imagem redimensionada
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return padded
    
    def predict_with_augmentation(self, img_path: str, n_augmentations: int = 5) -> np.ndarray:
        """
        Faz predição usando Test Time Augmentation (TTA) para maior precisão
        """
        predictions = []
        
        # Predição original
        img_batch = self.preprocess_image_advanced(img_path)
        pred = self.model.predict(img_batch, verbose=0)
        predictions.append(pred[0])
        
        # Augmentações para TTA
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        for i in range(n_augmentations - 1):
            # Aplicar diferentes augmentações
            augmented = img.copy()
            
            if i == 0:  # Flip horizontal
                augmented = cv2.flip(augmented, 1)
            elif i == 1:  # Rotação pequena
                angle = np.random.uniform(-10, 10)
                center = (augmented.shape[1]//2, augmented.shape[0]//2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                augmented = cv2.warpAffine(augmented, M, (augmented.shape[1], augmented.shape[0]))
            elif i == 2:  # Ajuste de brilho
                value = np.random.uniform(0.8, 1.2)
                augmented = cv2.convertScaleAbs(augmented, alpha=value, beta=0)
            elif i == 3:  # Zoom leve
                scale = np.random.uniform(0.9, 1.1)
                center = (augmented.shape[1]//2, augmented.shape[0]//2)
                M = cv2.getRotationMatrix2D(center, 0, scale)
                augmented = cv2.warpAffine(augmented, M, (augmented.shape[1], augmented.shape[0]))
            
            # Preprocessar e prever
            augmented = cv2.resize(augmented, IMG_SIZE)
            augmented = augmented.astype(np.float32) / 255.0
            augmented = np.expand_dims(augmented, axis=0)
            
            pred = self.model.predict(augmented, verbose=0)
            predictions.append(pred[0])
        
        # Média das predições (ensemble)
        final_prediction = np.mean(predictions, axis=0)
        
        return final_prediction
    
    def apply_confidence_calibration(self, predictions: np.ndarray) -> np.ndarray:
        """
        Aplica calibração de confiança para tornar as probabilidades mais realistas
        """
        # Temperature scaling
        temperature = 1.5  # Ajuste este valor baseado na validação
        scaled_logits = np.log(predictions + 1e-10) / temperature
        calibrated = np.exp(scaled_logits) / np.sum(np.exp(scaled_logits))
        
        return calibrated
    
    def predict_species(self, img_path: str, use_tta: bool = True) -> Dict:
        """
        Predição principal com todas as melhorias
        """
        try:
            # Verificar se arquivo existe
            if not os.path.exists(img_path):
                return {"erro": f"Arquivo não encontrado: {img_path}"}
            
            print(f"\n🔬 Analisando: {os.path.basename(img_path)}")
            
            # Fazer predição (com ou sem TTA)
            if use_tta:
                print("  📊 Aplicando Test Time Augmentation...")
                predictions = self.predict_with_augmentation(img_path)
            else:
                img_batch = self.preprocess_image_advanced(img_path)
                predictions = self.model.predict(img_batch, verbose=0)[0]
            
            # Aplicar calibração
            predictions = self.apply_confidence_calibration(predictions)
            
            # Obter top K predições
            top_k_indices = np.argsort(predictions)[-TOP_K_PREDICTIONS:][::-1]
            
            # Resultado principal
            best_idx = top_k_indices[0]
            best_confidence = predictions[best_idx] * 100
            
            # Obter nome da espécie
            species_name = self._get_species_name(best_idx)
            
            # Verificar qualidade da predição
            quality = self._assess_prediction_quality(predictions)
            
            # Montar resultado
            result = {
                "especie": species_name,
                "confianca": f"{best_confidence:.1f}%",
                "qualidade_predicao": quality,
                "arquivo": os.path.basename(img_path),
                "top_3_predicoes": []
            }
            
            # Adicionar top 3 predições
            for idx in top_k_indices:
                species = self._get_species_name(idx)
                conf = predictions[idx] * 100
                result["top_3_predicoes"].append({
                    "especie": species,
                    "confianca": f"{conf:.1f}%"
                })
            
            # Adicionar informações da espécie se disponível
            if species_name in self.species_info:
                result["info_detalhada"] = self.species_info[species_name]
            
            # Adicionar recomendação baseada na confiança
            if best_confidence < 60:
                result["recomendacao"] = "⚠️ Confiança muito baixa. Considere tirar uma foto melhor ou de ângulo diferente."
            elif best_confidence < 80:
                result["recomendacao"] = "⚡ Confiança moderada. Resultado pode precisar de verificação adicional."
            else:
                result["recomendacao"] = "✅ Alta confiança na identificação!"
            
            return result
            
        except Exception as e:
            return {"erro": f"Erro ao processar imagem: {str(e)}"}
    
    def _get_species_name(self, idx: int) -> str:
        """Obtém o nome da espécie pelo índice"""
        if str(idx) in self.labels:
            return self.labels[str(idx)]
        elif idx in self.labels:
            return self.labels[idx]
        else:
            return f"especie_desconhecida_{idx}"
    
    def _assess_prediction_quality(self, predictions: np.ndarray) -> str:
        """Avalia a qualidade da predição baseada na distribuição de probabilidades"""
        max_prob = np.max(predictions)
        entropy = -np.sum(predictions * np.log(predictions + 1e-10))
        
        if max_prob > 0.8 and entropy < 0.5:
            return "EXCELENTE"
        elif max_prob > 0.6 and entropy < 1.0:
            return "BOA"
        elif max_prob > 0.4:
            return "MODERADA"
        else:
            return "BAIXA"
    
    def analyze_batch(self, image_folder: str) -> List[Dict]:
        """Analisa múltiplas imagens de uma pasta"""
        results = []
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        for file in os.listdir(image_folder):
            if any(file.lower().endswith(ext) for ext in extensions):
                img_path = os.path.join(image_folder, file)
                result = self.predict_species(img_path)
                results.append(result)
        
        return results

def display_result(result: Dict):
    """Exibe o resultado de forma formatada"""
    print("\n" + "="*60)
    
    if "erro" in result:
        print(f"❌ ERRO: {result['erro']}")
    else:
        print("🦴 ANÁLISE DE FÓSSIL CONCLUÍDA")
        print("-"*60)
        print(f"📷 Arquivo: {result['arquivo']}")
        print(f"🎯 Espécie Identificada: {result['especie']}")
        print(f"📊 Confiança: {result['confianca']}")
        print(f"⭐ Qualidade da Predição: {result['qualidade_predicao']}")
        print(f"\n{result['recomendacao']}")
        
        if "info_detalhada" in result:
            info = result["info_detalhada"]
            print("\n📚 INFORMAÇÕES DA ESPÉCIE:")
            print(f"  Nome Comum: {info.get('nome_comum', 'N/A')}")
            print(f"  Período: {info.get('periodo', 'N/A')}")
            print(f"  Habitat: {info.get('habitat', 'N/A')}")
            print(f"  Tamanho: {info.get('tamanho', 'N/A')}")
            print(f"  Descrição: {info.get('descricao', 'N/A')}")
        
        print("\n🏆 TOP 3 POSSIBILIDADES:")
        for i, pred in enumerate(result["top_3_predicoes"], 1):
            print(f"  {i}. {pred['especie']}: {pred['confianca']}")
    
    print("="*60)

def interactive_menu():
    """Menu interativo melhorado"""
    predictor = FossilPredictor()
    
    while True:
        print("\n🧬 === ANALISADOR DE DNA FÓSSIL v3.0 ===")
        print("1. 📁 Analisar imagem (com TTA - mais preciso)")
        print("2. ⚡ Análise rápida (sem TTA)")
        print("3. 📂 Analisar pasta inteira")
        print("4. 🔍 Ver imagens disponíveis")
        print("5. ❌ Sair")
        
        opcao = input("\nEscolha (1-5): ").strip()
        
        if opcao == "1":
            caminho = input("📁 Caminho da imagem: ").strip().strip('"\'')
            if caminho and os.path.exists(caminho):
                result = predictor.predict_species(caminho, use_tta=True)
                display_result(result)
            else:
                print("❌ Arquivo não encontrado!")
        
        elif opcao == "2":
            caminho = input("📁 Caminho da imagem: ").strip().strip('"\'')
            if caminho and os.path.exists(caminho):
                result = predictor.predict_species(caminho, use_tta=False)
                display_result(result)
            else:
                print("❌ Arquivo não encontrado!")
        
        elif opcao == "3":
            pasta = input("📂 Caminho da pasta: ").strip().strip('"\'')
            if pasta and os.path.exists(pasta):
                results = predictor.analyze_batch(pasta)
                for result in results:
                    display_result(result)
                print(f"\n✅ Total de {len(results)} imagens analisadas")
            else:
                print("❌ Pasta não encontrada!")
        
        elif opcao == "4":
            # Listar imagens em pastas comuns
            pastas = ["data/test", "test", "images", "."]
            encontradas = []
            
            for pasta in pastas:
                if os.path.exists(pasta):
                    for file in os.listdir(pasta):
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            encontradas.append(os.path.join(pasta, file))
            
            if encontradas:
                print(f"\n📋 {len(encontradas)} imagens encontradas:")
                for i, img in enumerate(encontradas, 1):
                    print(f"  {i}. {img}")
                
                escolha = input(f"\nAnalisar qual? (1-{len(encontradas)}) ou 0 para voltar: ")
                try:
                    idx = int(escolha) - 1
                    if 0 <= idx < len(encontradas):
                        result = predictor.predict_species(encontradas[idx])
                        display_result(result)
                except:
                    pass
            else:
                print("❌ Nenhuma imagem encontrada!")
        
        elif opcao == "5":
            print("👋 Até logo!")
            break
        
        else:
            print("❌ Opção inválida!")

# Execução principal
if __name__ == "__main__":
    # Verificar se modelo existe
    if not os.path.exists(MODEL_PATH):
        print("⚠️ ATENÇÃO: Modelo não encontrado!")
        print("Execute primeiro o script de treinamento (train.py)")
    else:
        interactive_menu()