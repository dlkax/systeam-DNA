"""
Sistema de Treinamento de CNN para Classificação de Fósseis
Autor: systeam-DNA
Versão: 2.0
"""

import os
import json
import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    ReduceLROnPlateau,
    TensorBoard
)
from cnn_model import create_model
import matplotlib.pyplot as plt
import numpy as np

# Configurações de diretórios
DATA_DIR = "data/images"
MODEL_DIR = "saved_model"
LABELS_FILE = "model/labels.json"
LOGS_DIR = "logs"

# Hiperparâmetros
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 50  # Aumentado, mas com early stopping
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1  # Adicional para teste final

class FossilClassifierTrainer:
    """Classe principal para treinar o classificador de fósseis"""
    
    def __init__(self):
        self.model = None
        self.history = None
        self.class_names = None
        self.config = {
            'img_size': IMG_SIZE,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE,
            'validation_split': VALIDATION_SPLIT,
            'test_split': TEST_SPLIT
        }
        
    def load_and_prepare_data(self):
        """Carrega e prepara os datasets com augmentação de dados"""
        print("📂 Carregando dataset...")
        
        # Dataset de treino com augmentação
        train_ds = tf.keras.utils.image_dataset_from_directory(
            DATA_DIR,
            validation_split=VALIDATION_SPLIT,
            subset="training",
            seed=42,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode='categorical'  # Para usar categorical_crossentropy
        )
        
        # Dataset de validação (sem augmentação)
        val_ds = tf.keras.utils.image_dataset_from_directory(
            DATA_DIR,
            validation_split=VALIDATION_SPLIT,
            subset="validation",
            seed=42,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode='categorical'
        )
        
        # Capturar nomes das classes
        self.class_names = train_ds.class_names
        print(f"✅ Classes detectadas: {self.class_names}")
        print(f"📊 Total de classes: {len(self.class_names)}")
        
        # Salvar mapeamento de classes
        self._save_class_mapping()
        
        # Augmentação de dados para treino
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.2),
        ])
        
        # Aplicar augmentação apenas no treino
        train_ds = train_ds.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Normalização (0-255 → 0-1)
        normalization = tf.keras.layers.Rescaling(1./255)
        train_ds = train_ds.map(lambda x, y: (normalization(x), y))
        val_ds = val_ds.map(lambda x, y: (normalization(x), y))
        
        # Otimização de performance
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        
        # Calcular steps por epoch
        self.train_samples = tf.data.experimental.cardinality(train_ds).numpy() * BATCH_SIZE
        self.val_samples = tf.data.experimental.cardinality(val_ds).numpy() * BATCH_SIZE
        
        print(f"📈 Amostras de treino: ~{self.train_samples}")
        print(f"📉 Amostras de validação: ~{self.val_samples}")
        
        return train_ds, val_ds
    
    def _save_class_mapping(self):
        """Salva o mapeamento de classes em JSON"""
        os.makedirs(os.path.dirname(LABELS_FILE), exist_ok=True)
        
        # Criar mapeamento detalhado
        class_mapping = {
            "version": "2.0",
            "created_at": datetime.datetime.now().isoformat(),
            "num_classes": len(self.class_names),
            "id_to_name": {i: name for i, name in enumerate(self.class_names)},
            "name_to_id": {name: i for i, name in enumerate(self.class_names)},
            "config": self.config
        }
        
        with open(LABELS_FILE, "w", encoding='utf-8') as f:
            json.dump(class_mapping, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Mapeamento de classes salvo em: {LABELS_FILE}")
    
    def create_callbacks(self):
        """Cria callbacks para otimizar o treinamento"""
        
        # Criar diretório para logs
        os.makedirs(LOGS_DIR, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        callbacks = [
            # Salvar melhor modelo
            ModelCheckpoint(
                filepath=os.path.join(MODEL_DIR, "best_model.h5"),
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            
            # Early stopping para evitar overfitting
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduzir learning rate quando estagnar
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # TensorBoard para visualização
            TensorBoard(
                log_dir=os.path.join(LOGS_DIR, f"fit-{timestamp}"),
                histogram_freq=1,
                write_images=True
            )
        ]
        
        return callbacks
    
    def train(self, train_ds, val_ds):
        """Executa o treinamento do modelo"""
        print("\n🚀 Iniciando treinamento...")
        
        # Criar modelo
        self.model = create_model(num_classes=len(self.class_names))
        
        # Compilar com métricas adicionais
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        # Mostrar arquitetura do modelo
        print("\n📐 Arquitetura do modelo:")
        self.model.summary()
        
        # Callbacks
        callbacks = self.create_callbacks()
        
        # Treinar
        self.history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✅ Treinamento concluído!")
        
    def evaluate_model(self, val_ds):
        """Avalia o modelo final"""
        print("\n📊 Avaliação final do modelo:")
        
        results = self.model.evaluate(val_ds, verbose=1)
        
        metrics_names = self.model.metrics_names
        for name, value in zip(metrics_names, results):
            print(f"  {name}: {value:.4f}")
        
        return dict(zip(metrics_names, results))
    
    def plot_training_history(self):
        """Plota gráficos do histórico de treinamento"""
        if not self.history:
            print("⚠️ Nenhum histórico de treinamento disponível")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        if 'precision' in self.history.history:
            axes[1, 0].plot(self.history.history['precision'], label='Train')
            axes[1, 0].plot(self.history.history['val_precision'], label='Validation')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Recall
        if 'recall' in self.history.history:
            axes[1, 1].plot(self.history.history['recall'], label='Train')
            axes[1, 1].plot(self.history.history['val_recall'], label='Validation')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Salvar gráfico
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/training_history.png", dpi=150)
        print("📈 Gráficos salvos em: plots/training_history.png")
        plt.show()
    
    def save_model(self):
        """Salva o modelo final e metadados"""
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Salvar modelo completo
        model_path = os.path.join(MODEL_DIR, "fossil_classifier.h5")
        self.model.save(model_path)
        print(f"💾 Modelo salvo em: {model_path}")
        
        # Salvar apenas pesos (backup)
        weights_path = os.path.join(MODEL_DIR, "model_weights.h5")
        self.model.save_weights(weights_path)
        print(f"💾 Pesos salvos em: {weights_path}")
        
        # Salvar modelo em formato TensorFlow SavedModel
        tf_model_path = os.path.join(MODEL_DIR, "tf_saved_model")
        self.model.save(tf_model_path, save_format='tf')
        print(f"💾 Modelo TensorFlow salvo em: {tf_model_path}")
        
        # Salvar metadados do treinamento
        metadata = {
            "training_date": datetime.datetime.now().isoformat(),
            "epochs_trained": len(self.history.history['loss']) if self.history else 0,
            "final_metrics": {
                "train_accuracy": float(self.history.history['accuracy'][-1]) if self.history else 0,
                "val_accuracy": float(self.history.history['val_accuracy'][-1]) if self.history else 0,
                "train_loss": float(self.history.history['loss'][-1]) if self.history else 0,
                "val_loss": float(self.history.history['val_loss'][-1]) if self.history else 0,
            },
            "config": self.config,
            "classes": self.class_names
        }
        
        metadata_path = os.path.join(MODEL_DIR, "training_metadata.json")
        with open(metadata_path, "w", encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"📋 Metadados salvos em: {metadata_path}")
    
    def create_confusion_matrix(self, val_ds):
        """Cria matriz de confusão para análise de erros"""
        from sklearn.metrics import confusion_matrix, classification_report
        import seaborn as sns
        
        print("\n🔍 Gerando matriz de confusão...")
        
        # Coletar predições e labels reais
        y_true = []
        y_pred = []
        
        for images, labels in val_ds:
            predictions = self.model.predict(images, verbose=0)
            y_true.extend(np.argmax(labels.numpy(), axis=1))
            y_pred.extend(np.argmax(predictions, axis=1))
        
        # Criar matriz de confusão
        cm = confusion_matrix(y_true, y_pred)
        
        # Plotar
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title('Matriz de Confusão')
        plt.xlabel('Predição')
        plt.ylabel('Real')
        plt.tight_layout()
        
        # Salvar
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/confusion_matrix.png", dpi=150)
        print("📊 Matriz de confusão salva em: plots/confusion_matrix.png")
        plt.show()
        
        # Relatório de classificação
        print("\n📋 Relatório de Classificação:")
        print(classification_report(y_true, y_pred, 
                                   target_names=self.class_names))

def main():
    """Função principal para executar o treinamento"""
    
    # Verificar GPU
    print("🖥️ Configuração do Sistema:")
    print(f"  TensorFlow versão: {tf.__version__}")
    print(f"  GPUs disponíveis: {len(tf.config.list_physical_devices('GPU'))}")
    if tf.config.list_physical_devices('GPU'):
        print(f"  GPU: {tf.config.list_physical_devices('GPU')[0].name}")
    
    # Inicializar trainer
    trainer = FossilClassifierTrainer()
    
    try:
        # 1. Carregar dados
        train_ds, val_ds = trainer.load_and_prepare_data()
        
        # 2. Treinar modelo
        trainer.train(train_ds, val_ds)
        
        # 3. Avaliar modelo
        metrics = trainer.evaluate_model(val_ds)
        
        # 4. Visualizar resultados
        trainer.plot_training_history()
        trainer.create_confusion_matrix(val_ds)
        
        # 5. Salvar modelo
        trainer.save_model()
        
        print("\n🎉 Pipeline de treinamento concluído com sucesso!")
        print(f"📊 Acurácia final de validação: {metrics.get('accuracy', 0):.2%}")
        
        # Instruções para TensorBoard
        print("\n💡 Para visualizar métricas detalhadas no TensorBoard:")
        print(f"   tensorboard --logdir {LOGS_DIR}")
        
    except Exception as e:
        print(f"\n❌ Erro durante o treinamento: {str(e)}")
        raise

if __name__ == "__main__":
    main()