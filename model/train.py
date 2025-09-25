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
import matplotlib.pyplot as plt
import numpy as np

# ===========================
# Configurações
# ===========================
DATA_DIR = "data/images"
MODEL_DIR = "saved_model"
LABELS_FILE = "model/labels.json"
LOGS_DIR = "logs"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

# ===========================
# Modelo - CORRIGIDO
# ===========================
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

def create_model(num_classes: int, input_shape=(224, 224, 3), pretrained=True):
    """Cria o modelo CNN com EfficientNetB0 - VERSÃO CORRIGIDA"""
    
    if pretrained:
        weights = 'imagenet'
        # ⚠️ CORREÇÃO: Forçar input_shape para RGB quando usar pesos ImageNet
        input_shape = (224, 224, 3)
        print("🔧 Usando pesos ImageNet - forçando input_shape=(224, 224, 3)")
    else:
        weights = None
        print(f"🔧 Sem pesos pré-treinados - usando input_shape={input_shape}")

    try:
        # Backbone EfficientNetB0
        base_model = EfficientNetB0(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
            pooling='avg'
        )
        
        print(f"✅ EfficientNetB0 carregado com sucesso")
        print(f"📐 Input shape do modelo: {base_model.input_shape}")
        
    except Exception as e:
        print(f"❌ Erro ao carregar EfficientNetB0: {str(e)}")
        print("🔄 Tentando sem pesos pré-treinados...")
        
        # Fallback: criar modelo sem pesos pré-treinados
        base_model = EfficientNetB0(
            include_top=False,
            weights=None,
            input_shape=input_shape,
            pooling='avg'
        )
        pretrained = False

    # Congelar o backbone se usando pesos pré-treinados
    if pretrained:
        base_model.trainable = False
        print("🔒 Backbone congelado para fine-tuning gradual")
    else:
        base_model.trainable = True
        print("🔓 Backbone liberado para treinamento completo")

    # Construir o modelo completo
    model = models.Sequential([
        base_model,
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu', name='dense_1'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax', name='predictions')
    ], name='fossil_classifier')

    return model

# ===========================
# Classe de Treinamento
# ===========================
class FossilClassifierTrainer:
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

    # -------- Carregamento de Dados --------
    def load_and_prepare_data(self):
        print("📂 Carregando dataset...")
        
        # Verificar se o diretório existe
        if not os.path.exists(DATA_DIR):
            raise FileNotFoundError(f"Diretório de dados não encontrado: {DATA_DIR}")

        try:
            train_ds = tf.keras.utils.image_dataset_from_directory(
                DATA_DIR,
                validation_split=VALIDATION_SPLIT,
                subset="training",
                seed=42,
                image_size=IMG_SIZE,
                batch_size=BATCH_SIZE,
                label_mode='categorical',
                color_mode='rgb'  # ⚠️ Garantir RGB
            )

            val_ds = tf.keras.utils.image_dataset_from_directory(
                DATA_DIR,
                validation_split=VALIDATION_SPLIT,
                subset="validation",
                seed=42,
                image_size=IMG_SIZE,
                batch_size=BATCH_SIZE,
                label_mode='categorical',
                color_mode='rgb'  # ⚠️ Garantir RGB
            )
        except Exception as e:
            print(f"❌ Erro ao carregar dataset: {str(e)}")
            raise

        self.class_names = train_ds.class_names
        print(f"✅ Classes detectadas: {self.class_names}")
        print(f"📊 Total de classes: {len(self.class_names)}")
        
        # Verificar shape dos dados
        for images, labels in train_ds.take(1):
            print(f"📐 Shape das imagens: {images.shape}")
            print(f"📊 Shape dos labels: {labels.shape}")
            break
        
        self._save_class_mapping()

        # Augmentação de dados
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.2),
        ], name='data_augmentation')

        train_ds = train_ds.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Normalização
        normalization = tf.keras.layers.Rescaling(1./255, name='rescaling')
        train_ds = train_ds.map(lambda x, y: (normalization(x), y))
        val_ds = val_ds.map(lambda x, y: (normalization(x), y))

        # Otimizações de performance
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        # Contar amostras
        self.train_samples = tf.data.experimental.cardinality(train_ds).numpy() * BATCH_SIZE
        self.val_samples = tf.data.experimental.cardinality(val_ds).numpy() * BATCH_SIZE

        print(f"📈 Amostras de treino: ~{self.train_samples}")
        print(f"📉 Amostras de validação: ~{self.val_samples}")

        return train_ds, val_ds

    def _save_class_mapping(self):
        os.makedirs(os.path.dirname(LABELS_FILE), exist_ok=True)
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

    # -------- Callbacks --------
    def create_callbacks(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(LOGS_DIR, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        callbacks = [
            ModelCheckpoint(
                filepath=os.path.join(MODEL_DIR, "best_model.h5"),
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            TensorBoard(
                log_dir=os.path.join(LOGS_DIR, f"fit-{timestamp}"),
                histogram_freq=1,
                write_images=True
            )
        ]
        return callbacks

    # -------- Treinamento --------
    def train(self, train_ds, val_ds):
        print("\n🚀 Iniciando treinamento...")
        
        try:
            # Criar modelo
            self.model = create_model(num_classes=len(self.class_names))
            
            # Compilar modelo
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
            
            print("\n📐 Arquitetura do modelo:")
            self.model.summary()
            
            # Criar callbacks
            callbacks = self.create_callbacks()
            
            # Treinar modelo
            self.history = self.model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=EPOCHS,
                callbacks=callbacks,
                verbose=1
            )
            
            print("\n✅ Treinamento concluído!")
            
        except Exception as e:
            print(f"❌ Erro durante o treinamento: {str(e)}")
            raise

    # -------- Avaliação --------
    def evaluate_model(self, val_ds):
        print("\n📊 Avaliação final do modelo:")
        try:
            results = self.model.evaluate(val_ds, verbose=1)
            metrics_names = self.model.metrics_names
            for name, value in zip(metrics_names, results):
                print(f"  {name}: {value:.4f}")
            return dict(zip(metrics_names, results))
        except Exception as e:
            print(f"❌ Erro na avaliação: {str(e)}")
            return {}

    # -------- Plot History --------
    def plot_training_history(self):
        if not self.history:
            print("⚠️ Nenhum histórico de treinamento disponível")
            return

        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # Accuracy
            axes[0, 0].plot(self.history.history['accuracy'], label='Train')
            axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation')
            axes[0, 0].set_title('Model Accuracy')
            axes[0, 0].legend()
            axes[0, 0].grid(True)

            # Loss
            axes[0, 1].plot(self.history.history['loss'], label='Train')
            axes[0, 1].plot(self.history.history['val_loss'], label='Validation')
            axes[0, 1].set_title('Model Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True)

            # Precision
            if 'precision' in self.history.history:
                axes[1, 0].plot(self.history.history['precision'], label='Train')
                axes[1, 0].plot(self.history.history['val_precision'], label='Validation')
                axes[1, 0].set_title('Model Precision')
                axes[1, 0].legend()
                axes[1, 0].grid(True)

            # Recall
            if 'recall' in self.history.history:
                axes[1, 1].plot(self.history.history['recall'], label='Train')
                axes[1, 1].plot(self.history.history['val_recall'], label='Validation')
                axes[1, 1].set_title('Model Recall')
                axes[1, 1].legend()
                axes[1, 1].grid(True)

            plt.tight_layout()
            os.makedirs("plots", exist_ok=True)
            plt.savefig("plots/training_history.png", dpi=150, bbox_inches='tight')
            plt.show()
            print("📈 Gráficos salvos em: plots/training_history.png")
            
        except Exception as e:
            print(f"❌ Erro ao plotar gráficos: {str(e)}")

    # -------- Salvar Modelo --------
    def save_model(self):
        if not self.model:
            print("⚠️ Nenhum modelo para salvar")
            return
            
        try:
            os.makedirs(MODEL_DIR, exist_ok=True)
            
            # Salvar modelo completo
            model_path = os.path.join(MODEL_DIR, "fossil_classifier.h5")
            self.model.save(model_path)
            
            # Salvar apenas os pesos
            weights_path = os.path.join(MODEL_DIR, "model_weights.h5")
            self.model.save_weights(weights_path)
            
            # Salvar no formato TensorFlow
            tf_model_path = os.path.join(MODEL_DIR, "tf_saved_model")
            self.model.save(tf_model_path, save_format='tf')
            
            print(f"💾 Modelos salvos em: {MODEL_DIR}")

            # Salvar metadados
            metadata = {
                "training_date": datetime.datetime.now().isoformat(),
                "tensorflow_version": tf.__version__,
                "epochs_trained": len(self.history.history['loss']) if self.history else 0,
                "final_metrics": {
                    "train_accuracy": float(self.history.history['accuracy'][-1]) if self.history else 0,
                    "val_accuracy": float(self.history.history['val_accuracy'][-1]) if self.history else 0,
                    "train_loss": float(self.history.history['loss'][-1]) if self.history else 0,
                    "val_loss": float(self.history.history['val_loss'][-1]) if self.history else 0,
                },
                "config": self.config,
                "classes": self.class_names,
                "model_input_shape": self.model.input_shape if self.model else None
            }
            
            metadata_path = os.path.join(MODEL_DIR, "training_metadata.json")
            with open(metadata_path, "w", encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"📋 Metadados salvos em: {metadata_path}")
            
        except Exception as e:
            print(f"❌ Erro ao salvar modelo: {str(e)}")

# ===========================
# Main
# ===========================
def main():
    print("🖥️ Configuração do Sistema:")
    print(f"  TensorFlow versão: {tf.__version__}")
    print(f"  Python versão: {tf.version.VERSION}")
    
    # Configurar GPU se disponível
    physical_devices = tf.config.list_physical_devices('GPU')
    print(f"  GPUs físicas disponíveis: {len(physical_devices)}")
    
    if physical_devices:
        try:
            # Configurar crescimento de memória
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"  ✅ GPU configurada: {physical_devices[0].name}")
        except RuntimeError as e:
            print(f"  ⚠️ Erro na configuração da GPU: {e}")
    else:
        print("  ℹ️ Executando na CPU")
    
    # Verificar diretórios
    print(f"\n📁 Verificando estrutura de diretórios:")
    print(f"  Data dir: {DATA_DIR} - {'✅' if os.path.exists(DATA_DIR) else '❌'}")
    
    trainer = FossilClassifierTrainer()

    try:
        # Pipeline de treinamento
        train_ds, val_ds = trainer.load_and_prepare_data()
        trainer.train(train_ds, val_ds)
        metrics = trainer.evaluate_model(val_ds)
        trainer.plot_training_history()
        trainer.save_model()
        
        print("\n🎉 Pipeline de treinamento concluído com sucesso!")
        if metrics.get('accuracy'):
            print(f"📊 Acurácia final de validação: {metrics.get('accuracy', 0):.2%}")
        print(f"\n💡 Para visualizar o TensorBoard:")
        print(f"    tensorboard --logdir {LOGS_DIR}")

    except Exception as e:
        print(f"\n❌ Erro durante o treinamento: {str(e)}")
        print("\n🔧 Possíveis soluções:")
        print("1. Verifique se o diretório 'data/images' existe e contém as imagens")
        print("2. Certifique-se de ter pelo menos 2 classes de imagens")
        print("3. Verifique se as imagens estão em formato válido (jpg, png)")
        print("4. Tente executar sem pesos pré-treinados modificando pretrained=False")
        raise

if __name__ == "__main__":
    main()