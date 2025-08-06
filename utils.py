"""
FakeBuster Yardımcı Fonksiyonlar
Bu modül, deepfake tespit sistemi için yardımcı fonksiyonları içerir.
"""

import cv2
import numpy as np
import os
import json
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Logging konfigürasyonu
logger = logging.getLogger(__name__)

class ImageProcessor:
    """Görüntü işleme yardımcı sınıfı"""
    
    @staticmethod
    def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Görüntüyü yeniden boyutlandır"""
        return cv2.resize(image, target_size)
    
    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        """Görüntüyü normalize et (0-1 aralığına)"""
        return image.astype(np.float32) / 255.0
    
    @staticmethod
    def enhance_image(image: np.ndarray) -> np.ndarray:
        """Görüntü kalitesini artır"""
        # Histogram eşitleme
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        return enhanced
    
    @staticmethod
    def extract_face_region(image: np.ndarray, face_bbox: Tuple) -> np.ndarray:
        """Yüz bölgesini çıkar"""
        x, y, w, h = face_bbox
        return image[y:y+h, x:x+w]
    
    @staticmethod
    def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Gaussian blur uygula"""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

class VideoProcessor:
    """Video işleme yardımcı sınıfı"""
    
    @staticmethod
    def get_video_info(video_path: str) -> Dict:
        """Video bilgilerini al"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Video dosyası açılamadı: {video_path}")
        
        info = {
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': 0
        }
        
        if info['fps'] > 0:
            info['duration'] = info['frame_count'] / info['fps']
        
        cap.release()
        return info
    
    @staticmethod
    def extract_frames(video_path: str, max_frames: int = 30) -> List[np.ndarray]:
        """Videodan frame'leri çıkar"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Video dosyası açılamadı: {video_path}")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, frame_count // max_frames)
        
        for i in range(0, min(frame_count, max_frames * frame_interval), frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            
            if ret:
                frames.append(frame)
        
        cap.release()
        return frames
    
    @staticmethod
    def create_video_from_frames(frames: List[np.ndarray], output_path: str, fps: int = 30):
        """Frame'lerden video oluştur"""
        if not frames:
            raise ValueError("Frame listesi boş")
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()

class ResultAnalyzer:
    """Sonuç analiz yardımcı sınıfı"""
    
    @staticmethod
    def calculate_confidence_score(predictions: List[float]) -> float:
        """Güven skorunu hesapla"""
        if not predictions:
            return 0.5
        
        return np.mean(predictions)
    
    @staticmethod
    def determine_fake_threshold(confidence: float, threshold: float = 0.6) -> bool:
        """Sahte olup olmadığını belirle"""
        return confidence > threshold
    
    @staticmethod
    def analyze_face_results(face_results: List[Dict]) -> Dict:
        """Yüz analiz sonuçlarını analiz et"""
        if not face_results:
            return {
                'total_faces': 0,
                'fake_faces': 0,
                'real_faces': 0,
                'fake_percentage': 0,
                'avg_confidence': 0.5
            }
        
        total_faces = len(face_results)
        fake_faces = sum(1 for f in face_results if f['is_fake'])
        real_faces = total_faces - fake_faces
        fake_percentage = (fake_faces / total_faces) * 100
        avg_confidence = np.mean([f['confidence'] for f in face_results])
        
        return {
            'total_faces': total_faces,
            'fake_faces': fake_faces,
            'real_faces': real_faces,
            'fake_percentage': fake_percentage,
            'avg_confidence': avg_confidence
        }
    
    @staticmethod
    def analyze_video_results(frame_results: List[Dict]) -> Dict:
        """Video analiz sonuçlarını analiz et"""
        if not frame_results:
            return {
                'total_frames': 0,
                'fake_frames': 0,
                'real_frames': 0,
                'fake_percentage': 0,
                'avg_confidence': 0.5,
                'is_fake': False
            }
        
        total_frames = len(frame_results)
        fake_frames = sum(1 for f in frame_results if f['is_fake'])
        real_frames = total_frames - fake_frames
        fake_percentage = (fake_frames / total_frames) * 100
        avg_confidence = np.mean([f['confidence'] for f in frame_results])
        is_fake = fake_percentage > 50
        
        return {
            'total_frames': total_frames,
            'fake_frames': fake_frames,
            'real_frames': real_frames,
            'fake_percentage': fake_percentage,
            'avg_confidence': avg_confidence,
            'is_fake': is_fake
        }

class VisualizationHelper:
    """Görselleştirme yardımcı sınıfı"""
    
    @staticmethod
    def plot_confidence_distribution(confidences: List[float], title: str = "Güven Skoru Dağılımı"):
        """Güven skoru dağılımını çiz"""
        plt.figure(figsize=(10, 6))
        
        plt.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(np.mean(confidences), color='red', linestyle='--', 
                   label=f'Ortalama: {np.mean(confidences):.3f}')
        
        plt.xlabel('Güven Skoru')
        plt.ylabel('Frekans')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return plt.gcf()
    
    @staticmethod
    def plot_frame_analysis(frame_results: List[Dict], threshold: float = 0.6):
        """Frame analiz sonuçlarını çiz"""
        if not frame_results:
            return None
        
        frame_numbers = [r['frame_number'] for r in frame_results]
        confidences = [r['confidence'] for r in frame_results]
        is_fake = [r['is_fake'] for r in frame_results]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Güven skoru grafiği
        colors = ['green' if not fake else 'red' for fake in is_fake]
        ax1.scatter(frame_numbers, confidences, c=colors, alpha=0.7)
        ax1.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, 
                   label=f'Eşik ({threshold:.2%})')
        ax1.set_ylabel('Güven Skoru')
        ax1.set_title('Frame Güven Skorları')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Sahte frame dağılımı
        fake_frames = [i for i, fake in enumerate(is_fake) if fake]
        real_frames = [i for i, fake in enumerate(is_fake) if not fake]
        
        ax2.hist([real_frames, fake_frames], label=['Gerçek', 'Sahte'], 
                bins=10, alpha=0.7, color=['green', 'red'])
        ax2.set_xlabel('Frame Numarası')
        ax2.set_ylabel('Frame Sayısı')
        ax2.set_title('Frame Dağılımı')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_confidence_gauge(confidence: float, threshold: float = 0.6):
        """Güven skoru gauge grafiği oluştur"""
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Renk belirleme
        if confidence < 0.5:
            color = '#ff4444'
            label = "Düşük Güven"
        elif confidence < threshold:
            color = '#ffaa00'
            label = "Orta Güven"
        else:
            color = '#44ff44'
            label = "Yüksek Güven"
        
        # Bar grafiği
        ax.bar(['Güven Skoru'], [confidence], color=color, alpha=0.7)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Güven Oranı')
        ax.set_title(f'Analiz Sonucu: {label}')
        
        # Eşik çizgisi
        ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, 
                  label=f'Eşik ({threshold:.2%})')
        ax.legend()
        
        return fig

class FileManager:
    """Dosya yönetimi yardımcı sınıfı"""
    
    @staticmethod
    def ensure_directory(directory: str):
        """Dizinin var olduğundan emin ol"""
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Dizin oluşturuldu: {directory}")
    
    @staticmethod
    def save_results(results: Dict, output_path: str):
        """Analiz sonuçlarını kaydet"""
        try:
            # Tarih bilgisi ekle
            results['timestamp'] = datetime.now().isoformat()
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Sonuçlar kaydedildi: {output_path}")
            
        except Exception as e:
            logger.error(f"Sonuç kaydetme hatası: {e}")
    
    @staticmethod
    def load_results(input_path: str) -> Dict:
        """Kaydedilmiş sonuçları yükle"""
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            logger.info(f"Sonuçlar yüklendi: {input_path}")
            return results
            
        except Exception as e:
            logger.error(f"Sonuç yükleme hatası: {e}")
            return {}
    
    @staticmethod
    def get_supported_formats() -> Dict[str, List[str]]:
        """Desteklenen dosya formatlarını döndür"""
        return {
            'image': ['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'],
            'video': ['mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv']
        }
    
    @staticmethod
    def is_supported_format(file_path: str, file_type: str) -> bool:
        """Dosya formatının desteklenip desteklenmediğini kontrol et"""
        supported_formats = FileManager.get_supported_formats()
        
        if file_type not in supported_formats:
            return False
        
        file_extension = file_path.lower().split('.')[-1]
        return file_extension in supported_formats[file_type]

class PerformanceMonitor:
    """Performans izleme yardımcı sınıfı"""
    
    def __init__(self):
        self.metrics = {}
    
    def start_timer(self, name: str):
        """Zamanlayıcı başlat"""
        self.metrics[name] = {
            'start_time': datetime.now(),
            'end_time': None,
            'duration': None
        }
    
    def end_timer(self, name: str):
        """Zamanlayıcı bitir"""
        if name in self.metrics:
            self.metrics[name]['end_time'] = datetime.now()
            self.metrics[name]['duration'] = (
                self.metrics[name]['end_time'] - self.metrics[name]['start_time']
            ).total_seconds()
    
    def get_duration(self, name: str) -> Optional[float]:
        """Süreyi al"""
        if name in self.metrics and self.metrics[name]['duration'] is not None:
            return self.metrics[name]['duration']
        return None
    
    def get_all_metrics(self) -> Dict:
        """Tüm metrikleri al"""
        return self.metrics
    
    def reset_metrics(self):
        """Metrikleri sıfırla"""
        self.metrics = {}

class ConfigManager:
    """Konfigürasyon yönetimi yardımcı sınıfı"""
    
    DEFAULT_CONFIG = {
        'models': {
            'xception': {
                'enabled': True,
                'confidence_threshold': 0.6,
                'input_size': (299, 299)
            },
            'faceforensics': {
                'enabled': True,
                'confidence_threshold': 0.65,
                'input_size': (224, 224)
            },
            'gan_detector': {
                'enabled': True,
                'confidence_threshold': 0.6,
                'input_size': (224, 224)
            }
        },
        'face_detection': {
            'min_confidence': 0.5,
            'max_faces': 10
        },
        'video_analysis': {
            'max_frames': 30,
            'frame_interval': 1
        },
        'output': {
            'save_results': True,
            'output_dir': 'results',
            'save_visualizations': True
        }
    }
    
    @staticmethod
    def load_config(config_path: str = None) -> Dict:
        """Konfigürasyon dosyasını yükle"""
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                logger.info(f"Konfigürasyon yüklendi: {config_path}")
                return config
            except Exception as e:
                logger.error(f"Konfigürasyon yükleme hatası: {e}")
        
        logger.info("Varsayılan konfigürasyon kullanılıyor")
        return ConfigManager.DEFAULT_CONFIG.copy()
    
    @staticmethod
    def save_config(config: Dict, config_path: str):
        """Konfigürasyonu kaydet"""
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            logger.info(f"Konfigürasyon kaydedildi: {config_path}")
        except Exception as e:
            logger.error(f"Konfigürasyon kaydetme hatası: {e}")
    
    @staticmethod
    def get_model_config(config: Dict, model_name: str) -> Dict:
        """Belirli bir model konfigürasyonunu al"""
        return config.get('models', {}).get(model_name, {})
    
    @staticmethod
    def update_model_config(config: Dict, model_name: str, updates: Dict):
        """Model konfigürasyonunu güncelle"""
        if 'models' not in config:
            config['models'] = {}
        
        if model_name not in config['models']:
            config['models'][model_name] = {}
        
        config['models'][model_name].update(updates)
