import numpy as np
import os
import sys
from scripts.audio_utils import (
    load_audio,
    extract_features,
    save_alert_sound
)
from scripts.drone_detector import DroneDetector
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def collect_real_training_data(data_dir: str = None, sr: int = 22050) -> tuple:
    """
    Собирает реальные обучающие данные из директории
    
    Структура директории:
    data/raw/
    ├── drone/          # Файлы с дронами (имена должны содержать 'drone')
    └── background/     # Фоновые файлы (имена должны содержать 'background', 'noise', 'ambient')
    
    Args:
        data_dir: Директория с данными (по умолчанию data/raw)
        sr: Частота дискретизации
    
    Returns:
        tuple: (X, y) - признаки и метки
    """
    if data_dir is None:
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(project_dir, "data", "raw")
    
    print(f"📁 Поиск реальных данных в: {data_dir}")
    
    if not os.path.exists(data_dir):
        print(f"⚠️ Директория не найдена: {data_dir}")
        return None, None
    
    X = []
    y = []
    
    # Ищем все аудио файлы
    audio_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(('.wav', '.m4a', '.mp3', '.flac')):
                audio_files.append(os.path.join(root, file))
    
    if not audio_files:
        print(f"❌ Не найдено аудио файлов в {data_dir}")
        print("\n📋 Инструкция по подготовке данных:")
        print("1. Создайте директорию data/raw/")
        print("2. Поместите файлы с дронами (имена должны содержать 'drone')")
        print("3. Поместите фоновые файлы (имена должны содержать 'background', 'noise' или 'ambient')")
        print("4. Поддерживаемые форматы: .wav, .m4a, .mp3, .flac")
        return None, None
    
    print(f"📁 Найдено {len(audio_files)} аудио файлов")
    
    segment_length = 1.0
    segment_samples = int(segment_length * sr)
    
    for audio_path in audio_files:
        try:
            filename = os.path.basename(audio_path).lower()
            
            # Определяем метку по имени файла или директории
            if 'drone' in filename or 'drone' in os.path.dirname(audio_path).lower():
                label = 1
                label_name = "ДРОН"
            elif any(keyword in filename for keyword in ['background', 'noise', 'ambient', 'silence', 'quiet']):
                label = 0
                label_name = "ФОН"
            elif 'background' in os.path.dirname(audio_path).lower() or 'noise' in os.path.dirname(audio_path).lower():
                label = 0
                label_name = "ФОН"
            else:
                print(f"  ⚠️ Пропущен {os.path.basename(audio_path)} - не удалось определить метку")
                continue
            
            print(f"  📄 {os.path.basename(audio_path)} → {label_name}")
            
            # Загружаем аудио
            audio, file_sr = load_audio(audio_path, sr=sr)
            
            # Разбиваем на сегменты по 1 секунде
            for i in range(0, len(audio), segment_samples):
                segment = audio[i:i + segment_samples]
                if len(segment) < segment_samples / 2:  # Пропускаем слишком короткие
                    continue
                
                # Дополняем до нужной длины
                if len(segment) < segment_samples:
                    segment = np.pad(segment, (0, segment_samples - len(segment)))
                
                # Извлекаем признаки
                mel_spec = extract_features(segment, sr)
                mel_spec = mel_spec[:64, :44]  # Обрезаем до размера модели
                mel_spec = mel_spec.reshape(64, 44, 1)
                
                X.append(mel_spec)
                y.append(label)
            
        except Exception as e:
            print(f"  ❌ Ошибка обработки {os.path.basename(audio_path)}: {e}")
            continue
    
    if not X:
        print("❌ Не удалось собрать данные для обучения")
        return None, None
    
    X = np.array(X)
    y = np.array(y)
    
    drone_count = np.sum(y)
    background_count = len(y) - drone_count
    
    print(f"\n✅ Собрано {len(X)} сегментов:")
    print(f"   🚁 Дроны: {drone_count} сегментов")
    print(f"   🔇 Фон: {background_count} сегментов")
    
    if drone_count == 0:
        print("\n❌ ОШИБКА: Не найдено ни одного сегмента с дронами!")
        print("   Добавьте файлы с дронами в data/raw/")
        return None, None
    
    if background_count == 0:
        print("\n" + "=" * 60)
        print("⚠️ КРИТИЧЕСКОЕ ПРЕДУПРЕЖДЕНИЕ: Не найдено фоновых данных!")
        print("=" * 60)
        print("   Модель НЕ СМОЖЕТ правильно работать без отрицательных примеров!")
        print("   Она будет предсказывать 'дрон' для всех входных данных.")
        print("\n   📋 Что нужно сделать:")
        print("   1. Добавьте фоновые аудио файлы в data/raw/")
        print("   2. Имена файлов должны содержать: 'background', 'noise', 'ambient', 'silence' или 'quiet'")
        print("   3. Рекомендуется иметь примерно столько же фоновых данных, сколько дронов")
        print("\n   ⚠️ Обучение продолжится, но модель будет неработоспособна!")
        print("=" * 60)
        
        try:
            response = input("\n   Продолжить обучение без фоновых данных? (yes/no): ").strip().lower()
            if response not in ['yes', 'y', 'да', 'д']:
                print("   Обучение отменено.")
                return None, None
        except (EOFError, KeyboardInterrupt):
            print("\n   Обучение отменено.")
            return None, None
    
    return X, y

def main():
    # Пути
    # Используем абсолютные пути
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_dir, "models", "drone_detector.h5")
    alert_sound_path = os.path.join(project_dir, "alerts", "alarm.wav")
    
    # Создаем директории если не существуют
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(alert_sound_path), exist_ok=True)
    
    # Генерация звука тревоги
    if not save_alert_sound(alert_sound_path):
        print("⚠️ Продолжаем без звука тревоги")
    
    # Сбор реальных данных
    print("\n" + "=" * 60)
    print("📊 СБОР РЕАЛЬНЫХ ДАННЫХ ДЛЯ ОБУЧЕНИЯ")
    print("=" * 60)
    X, y = collect_real_training_data()
    
    if X is None or y is None:
        print("\n" + "=" * 60)
        print("❌ НЕ УДАЛОСЬ СОБРАТЬ ДАННЫЕ ДЛЯ ОБУЧЕНИЯ")
        print("=" * 60)
        print("\n📋 Что нужно сделать:")
        print("1. Создайте директорию: data/raw/")
        print("2. Поместите туда аудио файлы:")
        print("   - Файлы с дронами (имена должны содержать 'drone')")
        print("   - Фоновые файлы (имена должны содержать 'background', 'noise' или 'ambient')")
        print("3. Поддерживаемые форматы: .wav, .m4a, .mp3, .flac")
        print("\n💡 Пример структуры:")
        print("   data/raw/")
        print("   ├── drone_sample1.mp3")
        print("   ├── drone_sample2.wav")
        print("   ├── background_noise1.wav")
        print("   └── ambient_sound.mp3")
        return
    
    # Разделение данных
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"📊 Размеры данных:")
    print(f"  Обучение: {X_train.shape}")
    print(f"  Валидация: {X_val.shape}")
    
    # Создание и обучение модели
    detector = DroneDetector(input_shape=(64, 44, 1))
    detector.build_model()
    
    print("🧠 Обучение модели...")
    history = detector.train(
        X_train, y_train,
        X_val, y_val,
        epochs=30,
        batch_size=16,  # Меньший размер батча для экономии памяти
        model_path=model_path
    )
    
    # Оценка модели
    print("🎯 Оценка модели...")
    X_val_proc = detector.preprocess_data(X_val)
    loss, accuracy, precision, recall = detector.model.evaluate(X_val_proc, y_val, verbose=0)
    
    print(f"📈 Результаты на валидации:")
    print(f"  Точность: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    
    # Визуализация
    detector.plot_training_history(history)
    
    # Тестирование на примерах
    print("🧪 Тестирование на примерах...")
    test_indices = np.random.choice(len(X_val), 5, replace=False)
    
    for idx in test_indices:
        sample = X_val[idx]
        true_label = y_val[idx]
        pred_prob, is_drone = detector.predict(sample)
        
        print(f"Пример {idx}:")
        print(f"  Истинная метка: {'Дрон' if true_label == 1 else 'Фон'}")
        print(f"  Предсказание: {'Дрон' if is_drone else 'Фон'} (вероятность: {pred_prob:.4f})")
        print("-" * 50)
    
    print("✅ Обучение завершено! Модель готова к использованию.")

if __name__ == "__main__":
    main()