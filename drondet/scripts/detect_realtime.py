import numpy as np
import os
import time
import soundfile as sf
from scripts.drone_detector import DroneDetector
from scripts.audio_utils import extract_features
import warnings
warnings.filterwarnings('ignore')

# Импорт sounddevice с обработкой ошибок
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("⚠️ Предупреждение: sounddevice не установлен. Режим реального времени недоступен.")
    print("   Установите: pip install sounddevice")
    print("   На macOS может потребоваться: pip install sounddevice numpy")

def play_alert_sound(alert_path: str = None):
    """
    Воспроизводит звук тревоги
    """
    if not SOUNDDEVICE_AVAILABLE:
        print("🔔 Звуковое оповещение недоступно (sounddevice не установлен)")
        return
    
    if alert_path is None:
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        alert_path = os.path.join(project_dir, "alerts", "alarm.wav")
    
    if os.path.exists(alert_path):
        try:
            data, sr = sf.read(alert_path)
            sd.play(data, sr)
            sd.wait()
        except Exception as e:
            print(f"⚠️ Ошибка воспроизведения звука тревоги: {e}")
    else:
        # Генерируем простой звуковой сигнал
        duration = 0.5
        sr = 22050
        t = np.linspace(0, duration, int(sr * duration), False)
        alert_sound = 0.5 * np.sin(2 * np.pi * 880 * t)  # A5 нота
        sd.play(alert_sound, sr)
        sd.wait()

def detect_realtime(duration: float = 1.0, 
                    threshold: float = 0.7,
                    model_path: str = None,
                    alert_enabled: bool = True):
    """
    Детектирует дроны в реальном времени с микрофона
    
    Args:
        duration: Длительность аудио сегмента в секундах
        threshold: Порог вероятности для детекции дрона
        model_path: Путь к модели (если None, используется путь по умолчанию)
        alert_enabled: Включить ли звуковое оповещение
    """
    if not SOUNDDEVICE_AVAILABLE:
        print("❌ Режим реального времени недоступен: sounddevice не установлен")
        print("   Установите: pip install sounddevice")
        return
    
    print("🎤 Запуск детекции в реальном времени...")
    print("=" * 50)
    
    # Определяем пути
    if model_path is None:
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(project_dir, "models", "drone_detector.h5")
    
    # Загружаем модель
    print(f"📂 Загрузка модели: {model_path}")
    if not os.path.exists(model_path):
        print(f"❌ Модель не найдена: {model_path}")
        print("Пожалуйста, сначала обучите модель с помощью scripts/train.py")
        return
    
    detector = DroneDetector(input_shape=(64, 44, 1))
    try:
        detector.load_model(model_path)
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return
    
    print("✅ Модель загружена успешно")
    print(f"🎯 Порог детекции: {threshold}")
    print(f"⏱️  Длительность сегмента: {duration} сек")
    print("=" * 50)
    print("🎧 Начинаю запись с микрофона...")
    print("Нажмите Ctrl+C для остановки")
    print("=" * 50)
    
    sr = 22050  # Частота дискретизации
    segment_samples = int(duration * sr)
    
    detection_count = 0
    total_segments = 0
    
    try:
        while True:
            # Записываем аудио с микрофона
            print(f"\n🎤 Запись {duration} сек...", end=" ", flush=True)
            audio_data = sd.rec(int(segment_samples), samplerate=sr, channels=1, dtype='float32')
            sd.wait()  # Ждем окончания записи
            
            # Преобразуем в одномерный массив
            audio_data = audio_data.flatten()
            
            # Предсказание
            try:
                pred_prob, is_drone = detector.predict(audio_data, threshold=threshold)
                total_segments += 1
                
                if is_drone:
                    detection_count += 1
                    print(f"🚨 ДРОН ОБНАРУЖЕН! (вероятность: {pred_prob:.2%})")
                    
                    if alert_enabled:
                        play_alert_sound()
                else:
                    print(f"✅ Фон (вероятность дрона: {pred_prob:.2%})")
                
                # Статистика
                if total_segments % 10 == 0:
                    print(f"\n📊 Статистика: {detection_count}/{total_segments} детекций ({detection_count/total_segments*100:.1f}%)")
                
            except Exception as e:
                print(f"⚠️ Ошибка предсказания: {e}")
                continue
                
    except KeyboardInterrupt:
        print("\n\n" + "=" * 50)
        print("🛑 Детекция остановлена пользователем")
        print(f"📊 Итоговая статистика: {detection_count}/{total_segments} детекций")
        if total_segments > 0:
            print(f"📈 Процент детекций: {detection_count/total_segments*100:.1f}%")
        print("=" * 50)
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        print("Детекция остановлена")

def detect_from_file(audio_file: str,
                     model_path: str = None,
                     threshold: float = 0.7,
                     segment_length: float = 1.0):
    """
    Детектирует дроны в аудио файле
    
    Args:
        audio_file: Путь к аудио файлу
        model_path: Путь к модели
        threshold: Порог вероятности для детекции
        segment_length: Длина сегмента для анализа в секундах
    """
    print(f"📁 Анализ файла: {audio_file}")
    print("=" * 50)
    
    if not os.path.exists(audio_file):
        print(f"❌ Файл не найден: {audio_file}")
        return
    
    # Определяем пути
    if model_path is None:
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(project_dir, "models", "drone_detector.h5")
    
    # Загружаем модель
    detector = DroneDetector(input_shape=(64, 44, 1))
    try:
        detector.load_model(model_path)
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return
    
    # Загружаем аудио
    from scripts.audio_utils import load_audio
    try:
        audio, sr = load_audio(audio_file, sr=22050)
        print(f"✅ Аудио загружено: {len(audio)/sr:.2f} сек, {sr} Hz")
    except Exception as e:
        print(f"❌ Ошибка загрузки аудио: {e}")
        return
    
    # Разбиваем на сегменты
    segment_samples = int(segment_length * sr)
    detections = []
    
    print(f"\n🔍 Анализ сегментов по {segment_length} сек...")
    print("=" * 50)
    
    for i in range(0, len(audio), segment_samples):
        segment = audio[i:i + segment_samples]
        if len(segment) < segment_samples / 2:
            continue
        
        # Дополняем до нужной длины
        if len(segment) < segment_samples:
            segment = np.pad(segment, (0, segment_samples - len(segment)))
        
        # Предсказание
        try:
            pred_prob, is_drone = detector.predict(segment, threshold=threshold)
            time_start = i / sr
            time_end = (i + len(segment)) / sr
            
            detections.append({
                'time_start': time_start,
                'time_end': time_end,
                'probability': pred_prob,
                'is_drone': is_drone
            })
            
            status = "🚨 ДРОН" if is_drone else "✅ Фон"
            print(f"[{time_start:6.1f}-{time_end:6.1f} сек] {status} (вероятность: {pred_prob:.2%})")
            
        except Exception as e:
            print(f"⚠️ Ошибка обработки сегмента [{i/sr:.1f} сек]: {e}")
            continue
    
    # Итоговая статистика
    print("\n" + "=" * 50)
    drone_segments = sum(1 for d in detections if d['is_drone'])
    total_segments = len(detections)
    
    print(f"📊 Результаты анализа:")
    print(f"  Всего сегментов: {total_segments}")
    print(f"  Детекций дронов: {drone_segments}")
    if total_segments > 0:
        print(f"  Процент детекций: {drone_segments/total_segments*100:.1f}%")
    
    # Показываем временные метки детекций
    if drone_segments > 0:
        print(f"\n🚨 Временные метки детекций:")
        for d in detections:
            if d['is_drone']:
                print(f"  [{d['time_start']:6.1f}-{d['time_end']:6.1f} сек] вероятность: {d['probability']:.2%}")
    
    print("=" * 50)

def main():
    """
    Главная функция для запуска детекции
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Детекция дронов в реальном времени или в файле')
    parser.add_argument('--mode', type=str, choices=['realtime', 'file'], default='realtime',
                        help='Режим работы: realtime (реальное время) или file (анализ файла)')
    parser.add_argument('--file', type=str, default=None,
                        help='Путь к аудио файлу (для режима file)')
    parser.add_argument('--model', type=str, default=None,
                        help='Путь к модели (по умолчанию models/drone_detector.h5)')
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='Порог вероятности для детекции (по умолчанию 0.7)')
    parser.add_argument('--duration', type=float, default=1.0,
                        help='Длительность сегмента в секундах (для realtime, по умолчанию 1.0)')
    parser.add_argument('--no-alert', action='store_true',
                        help='Отключить звуковое оповещение')
    
    args = parser.parse_args()
    
    if args.mode == 'realtime':
        detect_realtime(
            duration=args.duration,
            threshold=args.threshold,
            model_path=args.model,
            alert_enabled=not args.no_alert
        )
    elif args.mode == 'file':
        if args.file is None:
            print("❌ Для режима 'file' необходимо указать путь к файлу через --file")
            return
        detect_from_file(
            audio_file=args.file,
            model_path=args.model,
            threshold=args.threshold
        )

if __name__ == "__main__":
    main()
