import numpy as np
import os
import sys
import librosa
import tensorflow as tf
from scripts.drone_detector import DroneDetector
from scripts.audio_utils import load_audio, extract_features
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def collect_user_samples(data_dir: str = None, generate_background: bool = True) -> tuple:
    """
    Собирает образцы от пользователя для дообучения
    
    Args:
        data_dir: Директория с пользовательскими данными
        generate_background: Если True, генерирует синтетические фоновые данные для баланса
    """
    print("🎧 Сбор образцов для дообучения...")
    
    # Используем абсолютный путь
    if data_dir is None:
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(project_dir, "data", "user_samples")
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"📁 Создана директория: {data_dir}")
    
    X = []
    y = []
    
    # Ищем все аудио файлы в директории (любые форматы)
    audio_extensions = ('.wav', '.m4a', '.mp3', '.flac', '.ogg', '.aac', '.wma', '.m4p', '.m4b', '.3gp', '.amr', '.au', '.ra')
    audio_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(audio_extensions):
                audio_files.append(os.path.join(root, file))
    
    if not audio_files:
        print("⚠️ Не найдено аудио файлов для дообучения")
        print("Поместите аудио файлы (любого формата) в директорию data/user_samples/")
        print("Все файлы будут считаться звуками дрона")
        return None, None
    
    print(f"📁 Найдено {len(audio_files)} аудио файлов")
    print("ℹ️ Все файлы будут обработаны как звуки дрона")
    
    for audio_path in audio_files:
        try:
            print(f"  Обработка: {os.path.basename(audio_path)}")
            
            # Все файлы считаются звуками дрона
            label = 1
            print(f"    ➡️ Метка: ДРОН")
            
            # Загружаем аудио
            audio, sr = load_audio(audio_path, sr=22050)
            
            # Разбиваем на сегменты по 1 секунде
            segment_length = 1.0
            segment_samples = int(segment_length * sr)
            
            segments_count_before = len(X)
            
            for i in range(0, len(audio), segment_samples):
                segment = audio[i:i + segment_samples]
                if len(segment) < segment_samples / 2:  # Пропускаем слишком короткие сегменты
                    continue
                
                if len(segment) < segment_samples:
                    segment = np.pad(segment, (0, segment_samples - len(segment)))
                
                # Извлекаем признаки
                mel_spec = extract_features(segment, sr)
                mel_spec = mel_spec[:64, :44]  # Обрезаем до размера модели
                mel_spec = mel_spec.reshape(64, 44, 1)
                
                X.append(mel_spec)
                y.append(label)
            
            segments_added = len(X) - segments_count_before
            print(f"    ✅ Добавлено сегментов: {segments_added}")
            
        except Exception as e:
            print(f"    ❌ Ошибка обработки {audio_path}: {e}")
    
    if not X:
        print("❌ Не удалось собрать данные для дообучения")
        return None, None
    
    # Генерируем фоновые данные для баланса, если нужно
    if generate_background and len(X) > 0:
        drone_count = np.sum(y) if y else 0
        background_count = len(y) - drone_count if y else 0
        
        if drone_count > 0 and background_count == 0:
            print(f"⚠️ Обнаружен дисбаланс: {drone_count} дронов, {background_count} фоновых записей")
            print("🔊 Генерирую синтетические фоновые данные для баланса...")
            
            sr = 22050
            segment_length = 1.0
            segment_samples = int(segment_length * sr)
            
            # Генерируем примерно столько же фоновых данных
            num_background = min(drone_count, 100)  # Ограничиваем до 100 для экономии времени
            
            for i in range(num_background):
                # Генерируем фоновый шум
                duration = 1.0
                t = np.linspace(0, duration, segment_samples, False)
                background_noise = 0.01 * np.random.normal(0, 0.1, len(t))
                # Добавляем случайные помехи
                if np.random.random() > 0.7:
                    freq = np.random.uniform(50, 1000)
                    interference = 0.005 * np.sin(2 * np.pi * freq * t)
                    background_noise += interference
                
                # Извлекаем признаки
                mel_spec = extract_features(background_noise, sr)
                mel_spec = mel_spec[:64, :44]
                mel_spec = mel_spec.reshape(64, 44, 1)
                
                X.append(mel_spec)
                y.append(0)  # Фон
            
            print(f"✅ Добавлено {num_background} синтетических фоновых сегментов")
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"✅ Собрано {len(X)} сегментов: {np.sum(y)} дронов, {len(y)-np.sum(y)} фоновых записей")
    return X, y

def main():
    print("🔄 Запуск дообучения модели")
    print("=" * 50)
    
    # Пути
    # Используем абсолютные пути
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_dir, "models", "drone_detector.h5")
    new_model_path = os.path.join(project_dir, "models", "drone_detector_retrained.h5")
    
    if not os.path.exists(model_path):
        print(f"❌ Основная модель не найдена: {model_path}")
        print("Пожалуйста, сначала обучите модель с помощью scripts/train.py")
        return
    
    # Сбор данных от пользователя
    X_new, y_new = collect_user_samples()
    
    if X_new is None or y_new is None:
        return
    
    # Загрузка предобученной модели
    print("🧠 Загрузка предобученной модели...")
    detector = DroneDetector(input_shape=(64, 44, 1))
    detector.load_model(model_path)
    
    # Разделение данных
    X_train, X_val, y_train, y_val = train_test_split(
        X_new, y_new, test_size=0.2, random_state=42
    )
    
    print(f"📊 Размеры данных для дообучения:")
    print(f"  Обучение: {X_train.shape}")
    print(f"  Валидация: {X_val.shape}")
    
    # Предобработка данных (scaler уже обучен при загрузке модели)
    X_train_proc = detector.preprocess_data(X_train, fit=False)
    X_val_proc = detector.preprocess_data(X_val, fit=False)
    
    # Дообучение (fine-tuning)
    print("🔧 Дообучение модели...")
    
    # Размораживаем последние слои для fine-tuning
    for layer in detector.model.layers[:-4]:  # Замораживаем все кроме последних 4 слоев
        layer.trainable = False
    
    # Компилируем модель заново
    detector.model.compile(
        optimizer=Adam(learning_rate=0.0001),  # Меньший learning rate для fine-tuning
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(new_model_path, monitor='val_accuracy', save_best_only=True)
    
    # Обучение
    history = detector.model.fit(
        X_train_proc, y_train,
        validation_data=(X_val_proc, y_val),
        epochs=20,
        batch_size=8,  # Маленький батч для экономии памяти
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    # Оценка
    print("🎯 Оценка дообученной модели...")
    loss, accuracy, precision, recall = detector.model.evaluate(X_val_proc, y_val, verbose=0)
    
    print(f"📈 Результаты дообучения:")
    print(f"  Точность: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    
    # Визуализация
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['accuracy'], label='Точность (обучение)')
    plt.plot(history.history['val_accuracy'], label='Точность (валидация)')
    plt.title('Результаты дообучения')
    plt.xlabel('Эпохи')
    plt.ylabel('Точность')
    plt.legend()
    plt.show()
    
    # Замена основной модели
    if accuracy > 0.7:  # Если качество хорошее
        print("✅ Замена основной модели на дообученную версию")
        os.replace(new_model_path, model_path)
        print(f"🔄 Модель успешно дообучена и сохранена: {model_path}")
    else:
        print("⚠️ Качество дообучения недостаточно высокое. Основная модель не заменена.")
        print(f"Сохранена дообученная версия: {new_model_path}")

if __name__ == "__main__":
    main()