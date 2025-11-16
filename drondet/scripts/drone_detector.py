import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from typing import Tuple, List, Optional
import os
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scripts.audio_utils import extract_features

class DroneDetector:
    def __init__(self, input_shape: Tuple[int, int, int] = (64, 44, 1)):
        """
        Инициализирует детектор дронов
        input_shape: (height, width, channels) для Mel-спектрограмм
        """
        self.input_shape = input_shape
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def build_model(self) -> tf.keras.Model:
        """
        Создает CNN модель для детектирования дронов
        """
        model = Sequential([
            # Слой 1
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape, padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.2),
            
            # Слой 2
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.3),
            
            # Слой 3
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.4),
            
            # Полносвязные слои
            Flatten(),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')  # Бинарная классификация: дрон/не дрон
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        self.model = model
        return model
    
    def preprocess_data(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Предобрабатывает данные для модели
        
        Args:
            X: Входные данные
            fit: Если True, обучает scaler (для обучающих данных), иначе только transform
        """
        # Нормализация
        X_flat = X.reshape(X.shape[0], -1)
        if fit or not hasattr(self.scaler, 'mean_'):
            # Обучаем scaler только на обучающих данных
            X_scaled = self.scaler.fit_transform(X_flat)
        else:
            # Используем уже обученный scaler
            X_scaled = self.scaler.transform(X_flat)
        X_processed = X_scaled.reshape(X.shape)
        
        return X_processed
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 50, batch_size: int = 32,
              model_path: str = None) -> tf.keras.callbacks.History:
        """
        Обучает модель на данных
        """
        if model_path is None:
            # Используем абсолютный путь
            project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(project_dir, "models", "drone_detector.h5")
        
        if self.model is None:
            self.build_model()
        
        # Предобработка данных
        X_train_proc = self.preprocess_data(X_train, fit=True)  # Обучаем scaler на train данных
        X_val_proc = self.preprocess_data(X_val, fit=False)  # Используем обученный scaler для val
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        model_checkpoint = ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        # Обучение
        history = self.model.fit(
            X_train_proc, y_train,
            validation_data=(X_val_proc, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )
        
        self.is_trained = True

        # Сохраняем scaler отдельно
        scaler_path = model_path.replace('.h5', '_scaler.pkl')
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(self.scaler, scaler_path)
        print(f"✅ Scaler сохранен: {scaler_path}")

        print(f"✅ Модель сохранена: {model_path}")
        return history
    
    def predict(self, audio_segment: np.ndarray, threshold: float = 0.7) -> Tuple[float, bool]:
        """
        Предсказывает вероятность наличия дрона в аудио сегменте
        
        Args:
            audio_segment: Аудио сегмент (сырой сигнал или Mel-спектрограмма)
            threshold: Порог вероятности для детекции дрона (по умолчанию 0.7)
        
        Returns:
            Tuple[float, bool]: (вероятность, является_ли_дроном)
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Модель не обучена. Сначала вызовите метод train() или load_model()")
        
        # Извлекаем признаки
        if len(audio_segment.shape) == 1:  # Если это сырой аудио сигнал
            mel_spec = extract_features(audio_segment, sr=22050)
        else:
            mel_spec = audio_segment
        
        # Обрезаем/дополняем до нужного размера
        if mel_spec.shape[0] < self.input_shape[0]:
            mel_spec = np.pad(mel_spec, ((0, self.input_shape[0] - mel_spec.shape[0]), (0, 0)), mode='constant')
        elif mel_spec.shape[0] > self.input_shape[0]:
            mel_spec = mel_spec[:self.input_shape[0], :]
        
        if mel_spec.shape[1] < self.input_shape[1]:
            mel_spec = np.pad(mel_spec, ((0, 0), (0, self.input_shape[1] - mel_spec.shape[1])), mode='constant')
        elif mel_spec.shape[1] > self.input_shape[1]:
            mel_spec = mel_spec[:, :self.input_shape[1]]
        
        # Подготавливаем для модели
        mel_spec = mel_spec.reshape(1, *self.input_shape)
        
        # Проверяем, обучен ли scaler правильно
        if not hasattr(self.scaler, 'mean_') or np.all(self.scaler.mean_ == 0):
            # Если scaler не обучен или обучен на нулях, используем стандартную нормализацию
            mel_spec = mel_spec / (np.max(np.abs(mel_spec)) + 1e-8)
        else:
            # Используем обученный scaler
            mel_spec_flat = mel_spec.reshape(1, -1)
            mel_spec_flat = self.scaler.transform(mel_spec_flat)
            mel_spec = mel_spec_flat.reshape(1, *self.input_shape)
        
        # Предсказание
        prediction = self.model.predict(mel_spec, verbose=0)[0][0]
        is_drone = prediction > threshold
        
        return prediction, is_drone
    
    def load_model(self, model_path: str = None):
        """
        Загружает предобученную модель и scaler
        """
        if model_path is None:
            # Используем абсолютный путь
            project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(project_dir, "models", "drone_detector.h5")
        
        if os.path.exists(model_path):
            self.model = load_model(model_path)
            self.is_trained = True

            # Загружаем scaler
            scaler_path = model_path.replace('.h5', '_scaler.pkl')
            if os.path.exists(scaler_path):
                try:
                    self.scaler = joblib.load(scaler_path)
                    print(f"✅ Scaler загружен: {scaler_path}")
                except Exception as e:
                    print(f"⚠️ Ошибка загрузки scaler: {e}")
                    print("⚠️ Создаем новый scaler")
                    self.scaler = StandardScaler()
            else:
                print(f"⚠️ Scaler не найден: {scaler_path}")
                print("⚠️ ВНИМАНИЕ: Модель будет использовать упрощенную нормализацию")
                print("⚠️ Для лучшей точности переобучите модель или используйте сохраненный scaler")
                # Не обучаем scaler на dummy данных - это даст неправильные результаты
                # Вместо этого в predict будет использоваться альтернативная нормализация

            print(f"✅ Модель загружена: {model_path}")
        else:
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
    
    def plot_training_history(self, history):
        """
        Отображает историю обучения
        """
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Точность (обучение)')
        plt.plot(history.history['val_accuracy'], label='Точность (валидация)')
        plt.title('Точность обучения')
        plt.xlabel('Эпохи')
        plt.ylabel('Точность')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Потери (обучение)')
        plt.plot(history.history['val_loss'], label='Потери (валидация)')
        plt.title('Функция потерь')
        plt.xlabel('Эпохи')
        plt.ylabel('Потери')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    print("Детектор дронов готов к использованию!")