import os
import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt

# Импорт pydub с обработкой ошибок (требует ffmpeg на macOS)
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("⚠️ Предупреждение: pydub не установлен. Конвертация m4a будет недоступна.")
    print("   Установите: pip install pydub")
    print("   На macOS также требуется ffmpeg: brew install ffmpeg")

def convert_m4a_to_wav(input_path: str, output_path: str) -> bool:
    """
    Конвертирует .m4a файл в .wav формат
    Требует pydub и ffmpeg (на macOS: brew install ffmpeg)
    """
    if not PYDUB_AVAILABLE:
        print(f"❌ pydub не установлен. Не удалось конвертировать {input_path}")
        print("   Установите: pip install pydub")
        print("   На macOS также требуется: brew install ffmpeg")
        return False
    
    try:
        audio = AudioSegment.from_file(input_path, format="m4a")
        audio.export(output_path, format="wav")
        print(f"✅ Конвертировано: {input_path} -> {output_path}")
        return True
    except FileNotFoundError as e:
        if "ffmpeg" in str(e).lower() or "ffprobe" in str(e).lower():
            print(f"❌ Ошибка конвертации {input_path}: ffmpeg не найден")
            print("   На macOS установите: brew install ffmpeg")
        else:
            print(f"❌ Ошибка конвертации {input_path}: {e}")
        return False
    except Exception as e:
        print(f"❌ Ошибка конвертации {input_path}: {e}")
        return False

def load_audio(file_path: str, sr: int = 22050) -> Tuple[np.ndarray, int]:
    """
    Загружает аудио файл, автоматически конвертируя в wav при необходимости
    Поддерживает любые форматы, которые может обработать librosa или pydub
    """
    # Список форматов, которые librosa может загрузить напрямую
    librosa_formats = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma']
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    # Пробуем загрузить напрямую через librosa
    try:
        audio, sr = librosa.load(file_path, sr=sr)
        return audio, sr
    except Exception as librosa_error:
        # Если librosa не смог загрузить, пробуем конвертировать через pydub
        if PYDUB_AVAILABLE:
            try:
                # Определяем формат по расширению
                if file_ext == '.m4a':
                    format_name = 'm4a'
                elif file_ext == '.aac':
                    format_name = 'aac'
                elif file_ext == '.wma':
                    format_name = 'wma'
                elif file_ext == '.mp3':
                    format_name = 'mp3'
                elif file_ext == '.flac':
                    format_name = 'flac'
                elif file_ext == '.ogg':
                    format_name = 'ogg'
                else:
                    format_name = None  # pydub попытается определить автоматически
                
                # Создаем временный wav файл
                temp_wav = file_path.rsplit('.', 1)[0] + '_temp.wav'
                
                try:
                    if format_name:
                        audio_segment = AudioSegment.from_file(file_path, format=format_name)
                    else:
                        audio_segment = AudioSegment.from_file(file_path)
                    
                    audio_segment.export(temp_wav, format="wav")
                    print(f"🔄 Конвертировано через pydub: {os.path.basename(file_path)} -> временный WAV")
                    
                    # Загружаем конвертированный файл
                    audio, sr = librosa.load(temp_wav, sr=sr)
                    
                    # Удаляем временный файл
                    try:
                        os.remove(temp_wav)
                    except:
                        pass
                    
                    return audio, sr
                except Exception as pydub_error:
                    # Удаляем временный файл при ошибке
                    try:
                        if os.path.exists(temp_wav):
                            os.remove(temp_wav)
                    except:
                        pass
                    raise ValueError(f"Не удалось конвертировать {file_path} через pydub: {pydub_error}")
            except Exception as convert_error:
                raise ValueError(f"Ошибка загрузки аудио {file_path}: {librosa_error}. Попытка конвертации также не удалась: {convert_error}")
        else:
            raise ValueError(f"Ошибка загрузки аудио {file_path}: {librosa_error}. Для конвертации установите pydub: pip install pydub")

def extract_features(audio: np.ndarray, sr: int, n_mels: int = 64, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    """
    Извлекает Mel-спектрограмму из аудио сигнала
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio, 
        sr=sr, 
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

def create_spectrogram_segment(audio: np.ndarray, sr: int, segment_length: float = 1.0) -> List[np.ndarray]:
    """
    Разбивает аудио на сегменты и создает спектрограммы
    """
    segment_samples = int(segment_length * sr)
    segments = []
    
    for i in range(0, len(audio), segment_samples):
        segment = audio[i:i + segment_samples]
        if len(segment) < segment_samples:
            # Дополняем нулями до нужной длины
            segment = np.pad(segment, (0, segment_samples - len(segment)))
        
        mel_spec = extract_features(segment, sr)
        segments.append(mel_spec)
    
    return segments

def plot_spectrogram(mel_spec: np.ndarray, sr: int, hop_length: int = 512, title: str = "Спектрограмма"):
    """
    Отображает спектрограмму
    """
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        mel_spec, 
        sr=sr, 
        hop_length=hop_length,
        x_axis='time', 
        y_axis='mel'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def save_alert_sound(output_path: str, duration: float = 1.0, sr: int = 22050):
    """
    Генерирует и сохраняет звук тревоги
    """
    # Создаем директорию если не существует
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    t = np.linspace(0, duration, int(sr * duration), False)
    frequency = 880  # A5 нота
    
    # Создаем прерывистый звук тревоги
    alert_sound = np.zeros_like(t)
    for i in range(0, len(t), int(sr * 0.1)):
        segment_end = min(i + int(sr * 0.05), len(t))
        alert_sound[i:segment_end] = 0.8 * np.sin(2 * np.pi * frequency * t[i:segment_end])
    
    # Добавляем эффект затухания
    envelope = np.exp(-5 * t)
    alert_sound = alert_sound * envelope
    
    # Нормализуем и сохраняем
    alert_sound = alert_sound / np.max(np.abs(alert_sound))
    
    try:
        sf.write(output_path, alert_sound, sr)
        print(f"✅ Звук тревоги сохранен: {output_path}")
        return True
    except Exception as e:
        print(f"❌ Ошибка сохранения звука тревоги {output_path}: {e}")
        return False

def record_audio_chunk(duration: float = 1.0, sr: int = 22050) -> np.ndarray:
    """
    Записывает аудио с микрофона (заглушка для реальной реализации)
    """
    print(f"🎤 Запись аудио {duration} секунд...")
    # В реальной реализации здесь будет код для записи с микрофона
    # Заглушка: возвращает тишину (нулевой массив)
    t = np.linspace(0, duration, int(sr * duration), False)
    background_noise = np.zeros(len(t))
    return background_noise

if __name__ == "__main__":
    print("Аудио утилиты готовы к использованию!")