import cv2
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import mediapipe as mp
import time
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.ensemble import IsolationForest, RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Logging konfigürasyonu
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepfakeDetector:
    """
    Deepfake ve AI üretimi medya tespiti için ana sınıf
    """
    
    def __init__(self, model_type: str = "XceptionNet"):
        """
        DeepfakeDetector sınıfını başlat
        
        Args:
            model_type: Kullanılacak model türü
        """
        self.model_type = model_type
        self.models = {}
        self.face_detector = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Gelişmiş analiz araçları
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.ensemble_classifiers = {}
        self.uncertainty_estimator = None
        self.probability_calibrator = None
        
        # Model yükleme
        self._load_models()
        self._setup_face_detection()
        self._setup_ensemble_models()
        self._setup_uncertainty_estimation()
        
        logger.info(f"DeepfakeDetector başlatıldı - Model: {model_type}, Cihaz: {self.device}")
    
    def _load_models(self):
        """Gerekli modelleri yükle"""
        try:
            if self.model_type == "XceptionNet":
                self._load_xception_model()
            elif self.model_type == "FaceForensics++":
                self._load_faceforensics_model()
            elif self.model_type == "GANDetector":
                self._load_gan_detector_model()
            elif self.model_type == "Hibrit Model":
                self._load_hybrid_models()
            else:
                logger.warning(f"Bilinmeyen model türü: {self.model_type}")
                
        except Exception as e:
            logger.error(f"Model yükleme hatası: {e}")
            # Fallback olarak basit model kullan
            self._load_simple_model()
    
    def _load_xception_model(self):
        """XceptionNet modelini yükle"""
        try:
            # XceptionNet model mimarisi (basitleştirilmiş)
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(299, 299, 3)),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            self.models['xception'] = model
            logger.info("XceptionNet modeli yüklendi")
            
        except Exception as e:
            logger.error(f"XceptionNet yükleme hatası: {e}")
    
    def _load_faceforensics_model(self):
        """FaceForensics++ modelini yükle"""
        try:
            # FaceForensics++ model mimarisi
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            self.models['faceforensics'] = model
            logger.info("FaceForensics++ modeli yüklendi")
            
        except Exception as e:
            logger.error(f"FaceForensics++ yükleme hatası: {e}")
    
    def _load_gan_detector_model(self):
        """GANDetector modelini yükle"""
        try:
            # GANDetector PyTorch modeli
            class GANDetector(nn.Module):
                def __init__(self):
                    super(GANDetector, self).__init__()
                    self.features = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        nn.Conv2d(64, 128, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        nn.Conv2d(128, 256, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.AdaptiveAvgPool2d((1, 1))
                    )
                    self.classifier = nn.Sequential(
                        nn.Linear(256, 128),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.5),
                        nn.Linear(128, 1),
                        nn.Sigmoid()
                    )
                
                def forward(self, x):
                    x = self.features(x)
                    x = x.view(x.size(0), -1)
                    x = self.classifier(x)
                    return x
            
            model = GANDetector().to(self.device)
            self.models['gan_detector'] = model
            logger.info("GANDetector modeli yüklendi")
            
        except Exception as e:
            logger.error(f"GANDetector yükleme hatası: {e}")
    
    def _load_hybrid_models(self):
        """Hibrit model kombinasyonu"""
        self._load_xception_model()
        self._load_faceforensics_model()
        self._load_gan_detector_model()
        logger.info("Hibrit modeller yüklendi")
    
    def _load_simple_model(self):
        """Basit fallback model"""
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            self.models['simple'] = model
            logger.info("Basit fallback model yüklendi")
            
        except Exception as e:
            logger.error(f"Basit model yükleme hatası: {e}")
    
    def _setup_face_detection(self):
        """Yüz tespit sistemini kur"""
        try:
            # MediaPipe yüz tespiti
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5
            )
            logger.info("MediaPipe yüz tespit sistemi kuruldu")
            
        except Exception as e:
            logger.error(f"Yüz tespit sistemi kurulum hatası: {e}")
            self.face_detection = None
    
    def _setup_ensemble_models(self):
        """
        Ensemble (topluluk) modellerini kur
        """
        try:
            # Random Forest sınıflandırıcı
            rf_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            # SVM sınıflandırıcı
            svm_classifier = SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            )
            
            # Gaussian Process sınıflandırıcı
            gp_classifier = GaussianProcessClassifier(
                random_state=42,
                n_restarts_optimizer=3
            )
            
            # Voting sınıflandırıcı (ensemble)
            self.ensemble_classifiers['voting'] = VotingClassifier(
                estimators=[
                    ('rf', rf_classifier),
                    ('svm', svm_classifier),
                    ('gp', gp_classifier)
                ],
                voting='soft'  # Olasılık tabanlı oylama
            )
            
            # Kalibre edilmiş sınıflandırıcı
            self.probability_calibrator = CalibratedClassifierCV(
                self.ensemble_classifiers['voting'],
                method='isotonic',
                cv=3
            )
            
            logger.info("Ensemble modelleri başarıyla kuruldu")
            
        except Exception as e:
            logger.error(f"Ensemble model kurulum hatası: {e}")
    
    def _setup_uncertainty_estimation(self):
        """
        Belirsizlik tahmini kurulumu
        """
        try:
            # Gaussian Process belirsizlik tahmini için
            self.uncertainty_estimator = GaussianProcessClassifier(
                random_state=42,
                n_restarts_optimizer=5
            )
            
            logger.info("Belirsizlik tahmini kuruldu")
            
        except Exception as e:
            logger.error(f"Belirsizlik tahmini kurulum hatası: {e}")
    
    def preprocess_image(self, image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Görüntüyü model için ön işleme
        
        Args:
            image: İşlenecek görüntü
            target_size: Hedef boyut
            
        Returns:
            Ön işlenmiş görüntü
        """
        try:
            # Boyut değiştirme
            resized = cv2.resize(image, target_size)
            
            # Normalizasyon
            normalized = resized.astype(np.float32) / 255.0
            
            # Batch boyutu ekleme
            if len(normalized.shape) == 3:
                normalized = np.expand_dims(normalized, axis=0)
            
            return normalized
            
        except Exception as e:
            logger.error(f"Görüntü ön işleme hatası: {e}")
            return None
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Görüntüdeki yüzleri tespit et
        
        Args:
            image: İşlenecek görüntü
            
        Returns:
            Tespit edilen yüzlerin listesi
        """
        faces = []
        
        try:
            if self.face_detection is not None:
                # MediaPipe ile yüz tespiti
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.face_detection.process(rgb_image)
                
                if results.detections:
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        h, w, _ = image.shape
                        
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        width = int(bbox.width * w)
                        height = int(bbox.height * h)
                        
                        faces.append({
                            'bbox': (x, y, width, height),
                            'confidence': detection.score[0]
                        })
            
        except Exception as e:
            logger.error(f"Yüz tespit hatası: {e}")
        
        return faces
    
    def extract_face_features(self, image: np.ndarray, face_bbox: Tuple) -> np.ndarray:
        """
        Ultra gelişmiş yüz özellik çıkarma
        
        Args:
            image: Görüntü
            face_bbox: Yüz bounding box'ı
            
        Returns:
            Yüz özellikleri
        """
        try:
            x, y, w, h = face_bbox
            face_region = image[y:y+h, x:x+w]
            
            # Yüz bölgesini yeniden boyutlandır
            face_resized = cv2.resize(face_region, (256, 256))
            
            features = []
            
            # 1. Renk histogramı (512)
            color_hist = cv2.calcHist([face_resized], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            color_hist = color_hist.flatten() / (color_hist.sum() + 1e-8)
            features.extend(color_hist)
            
            # 2. Gri tonlama özellikleri
            gray_face = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            
            # Histogram (256)
            gray_hist = cv2.calcHist([gray_face], [0], None, [256], [0, 256])
            gray_hist = gray_hist.flatten() / (gray_hist.sum() + 1e-8)
            features.extend(gray_hist)
            
            # 3. Gradient özellikleri (256)
            grad_x = cv2.Sobel(gray_face, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_face, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            gradient_direction = np.arctan2(grad_y, grad_x)
            
            # Gradient magnitude histogramı (128)
            grad_mag_hist = cv2.calcHist([gradient_magnitude.astype(np.uint8)], [0], None, [128], [0, 256])
            grad_mag_hist = grad_mag_hist.flatten() / (grad_mag_hist.sum() + 1e-8)
            features.extend(grad_mag_hist)
            
            # Gradient direction histogramı (128)
            grad_dir_hist = cv2.calcHist([gradient_direction.astype(np.uint8)], [0], None, [128], [0, 256])
            grad_dir_hist = grad_dir_hist.flatten() / (grad_dir_hist.sum() + 1e-8)
            features.extend(grad_dir_hist)
            
            # 4. Laplacian özellikleri (128)
            laplacian = cv2.Laplacian(gray_face, cv2.CV_64F)
            laplacian_hist = cv2.calcHist([laplacian.astype(np.uint8)], [0], None, [128], [0, 256])
            laplacian_hist = laplacian_hist.flatten() / (laplacian_hist.sum() + 1e-8)
            features.extend(laplacian_hist)
            
            # 5. LBP özellikleri (256)
            lbp = self._compute_lbp(gray_face)
            lbp_hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
            lbp_hist = lbp_hist.flatten() / (lbp_hist.sum() + 1e-8)
            features.extend(lbp_hist)
            
            # 6. Yüz simetri analizi (128)
            symmetry_features = self._analyze_face_symmetry(gray_face)
            features.extend(symmetry_features)
            
            # 7. Yüz bölgesi analizi (256)
            region_features = self._analyze_face_regions(gray_face)
            features.extend(region_features)
            
            # 8. Yüz kenar analizi (128)
            edge_features = self._analyze_face_edges(gray_face)
            features.extend(edge_features)
            
            # 9. Yüz doku analizi (128)
            texture_features = self._analyze_face_texture(gray_face)
            features.extend(texture_features)
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Yüz özellik çıkarma hatası: {e}")
            return None
    
    def _analyze_face_symmetry(self, gray_face: np.ndarray) -> np.ndarray:
        """
        Gelişmiş yüz simetri analizi
        
        Args:
            gray_face: Gri tonlamalı yüz görüntüsü
            
        Returns:
            Simetri özellikleri
        """
        try:
            h, w = gray_face.shape
            center_x = w // 2
            
            # Sol ve sağ yarıyı karşılaştır
            left_half = gray_face[:, :center_x]
            right_half = cv2.flip(gray_face[:, center_x:], 1)
            
            # Boyutları eşitle
            min_width = min(left_half.shape[1], right_half.shape[1])
            left_half = left_half[:, :min_width]
            right_half = right_half[:, :min_width]
            
            # Fark hesapla
            diff = np.abs(left_half.astype(np.float32) - right_half.astype(np.float32))
            
            # Simetri özellikleri
            symmetry_features = []
            
            # 1. Ortalama fark
            symmetry_features.append(np.mean(diff) / 255.0)
            
            # 2. Fark standart sapması
            symmetry_features.append(np.std(diff) / 255.0)
            
            # 3. Fark entropisi
            diff_entropy = -np.sum(diff.flatten() * np.log(diff.flatten() + 1e-8))
            symmetry_features.append(diff_entropy / 1000.0)
            
            # 4. Fark histogramı (64)
            diff_hist = cv2.calcHist([diff.astype(np.uint8)], [0], None, [64], [0, 255])
            diff_hist = diff_hist.flatten() / (diff_hist.sum() + 1e-8)
            symmetry_features.extend(diff_hist)
            
            # 5. Bölgesel simetri analizi
            regions = ['upper', 'middle', 'lower']
            for region in regions:
                if region == 'upper':
                    region_diff = diff[:h//3, :]
                elif region == 'middle':
                    region_diff = diff[h//3:2*h//3, :]
                else:  # lower
                    region_diff = diff[2*h//3:, :]
                
                if region_diff.size > 0:
                    region_mean = np.mean(region_diff) / 255.0
                    region_std = np.std(region_diff) / 255.0
                    symmetry_features.extend([region_mean, region_std])
                else:
                    symmetry_features.extend([0.0, 0.0])
            
            # 6. Simetri skoru (düşük değer = yüksek simetri)
            symmetry_score = 1.0 - (np.mean(diff) / 255.0)
            symmetry_features.append(symmetry_score)
            
            return np.array(symmetry_features)
            
        except Exception as e:
            logger.error(f"Simetri analizi hatası: {e}")
            return np.zeros(128)
    
    def _analyze_face_regions(self, gray_face: np.ndarray) -> np.ndarray:
        """
        Yüz bölgesi analizi
        
        Args:
            gray_face: Gri tonlamalı yüz görüntüsü
            
        Returns:
            Bölge özellikleri
        """
        try:
            h, w = gray_face.shape
            
            # Yüz bölgelerini tanımla
            regions = {
                'forehead': gray_face[:h//4, :],
                'eyes': gray_face[h//4:h//2, :],
                'nose': gray_face[h//2:3*h//4, w//4:3*w//4],
                'mouth': gray_face[3*h//4:, :],
                'left_cheek': gray_face[h//4:3*h//4, :w//4],
                'right_cheek': gray_face[h//4:3*h//4, 3*w//4:]
            }
            
            features = []
            
            for region_name, region in regions.items():
                if region.size > 0:
                    # Her bölge için özellikler
                    region_mean = np.mean(region)
                    region_std = np.std(region)
                    region_entropy = -np.sum(region.flatten() * np.log(region.flatten() + 1e-8))
                    
                    # Histogram (32 özellik)
                    hist = cv2.calcHist([region], [0], None, [32], [0, 256])
                    hist = hist.flatten() / (hist.sum() + 1e-8)
                    
                    features.extend([region_mean/255.0, region_std/255.0, region_entropy/1000.0])
                    features.extend(hist)
                else:
                    features.extend([0.0] * 35)  # 3 + 32 = 35 özellik
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Yüz bölgesi analizi hatası: {e}")
            return np.zeros(256)
    
    def _analyze_face_edges(self, gray_face: np.ndarray) -> np.ndarray:
        """
        Yüz kenar analizi
        
        Args:
            gray_face: Gri tonlamalı yüz görüntüsü
            
        Returns:
            Kenar özellikleri
        """
        try:
            features = []
            
            # Canny kenar tespiti
            edges = cv2.Canny(gray_face, 50, 150)
            
            # Kenar yoğunluğu
            edge_density = np.sum(edges > 0) / edges.size
            features.append(edge_density)
            
            # Kenar histogramı (64)
            edge_hist = cv2.calcHist([edges], [0], None, [64], [0, 256])
            edge_hist = edge_hist.flatten() / (edge_hist.sum() + 1e-8)
            features.extend(edge_hist)
            
            # Kenar yönü analizi
            grad_x = cv2.Sobel(gray_face, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_face, cv2.CV_64F, 0, 1, ksize=3)
            edge_angles = np.arctan2(grad_y, grad_x)
            
            # Açı histogramı (63)
            angle_hist, _ = np.histogram(edge_angles.flatten(), bins=63, range=(-np.pi, np.pi))
            angle_hist = angle_hist / (angle_hist.sum() + 1e-8)
            features.extend(angle_hist)
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Yüz kenar analizi hatası: {e}")
            return np.zeros(128)
    
    def _analyze_face_texture(self, gray_face: np.ndarray) -> np.ndarray:
        """
        Yüz doku analizi
        
        Args:
            gray_face: Gri tonlamalı yüz görüntüsü
            
        Returns:
            Doku özellikleri
        """
        try:
            features = []
            
            # Gabor filtreleri ile doku analizi
            angles = [0, 45, 90, 135]
            frequencies = [0.1, 0.3, 0.5]
            
            for angle in angles:
                for freq in frequencies:
                    kernel = cv2.getGaborKernel((21, 21), 3, np.radians(angle), 2*np.pi*freq, 0.5, 0, ktype=cv2.CV_32F)
                    filtered = cv2.filter2D(gray_face, cv2.CV_8UC3, kernel)
                    
                    # Filtrelenmiş görüntü özellikleri
                    filtered_mean = np.mean(filtered)
                    filtered_std = np.std(filtered)
                    filtered_entropy = -np.sum(filtered.flatten() * np.log(filtered.flatten() + 1e-8))
                    
                    features.extend([filtered_mean/255.0, filtered_std/255.0, filtered_entropy/1000.0])
            
            # LBP doku analizi
            lbp = self._compute_lbp(gray_face)
            lbp_hist = cv2.calcHist([lbp], [0], None, [32], [0, 256])
            lbp_hist = lbp_hist.flatten() / (lbp_hist.sum() + 1e-8)
            features.extend(lbp_hist)
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Yüz doku analizi hatası: {e}")
            return np.zeros(128)
    
    def _multi_scale_analysis(self, image: np.ndarray) -> Dict:
        """
        Çoklu ölçek analizi - farklı çözünürlüklerde analiz
        
        Args:
            image: Analiz edilecek görüntü
            
        Returns:
            Çoklu ölçek analiz sonuçları
        """
        try:
            scales = [(64, 64), (128, 128), (256, 256), (512, 512)]
            scale_results = {}
            
            for i, scale in enumerate(scales):
                # Görüntüyü yeniden boyutlandır
                resized_image = cv2.resize(image, scale)
                
                # Her ölçek için özellik çıkar
                features = self._extract_image_features(resized_image)
                
                if features is not None and len(features) > 0:
                    # Bu ölçek için tahmin yap
                    confidence = self._simple_feature_based_prediction(features)
                    
                    scale_results[f'scale_{scale[0]}x{scale[1]}'] = {
                        'confidence': confidence,
                        'features_count': len(features),
                        'scale': scale
                    }
                    
                    logger.info(f"Ölçek {scale}: Güven={confidence:.3f}")
            
            return scale_results
            
        except Exception as e:
            logger.error(f"Çoklu ölçek analizi hatası: {e}")
            return {}
    
    def _ensemble_prediction(self, features: np.ndarray) -> Dict:
        """
        Ensemble tahmin yöntemi
        
        Args:
            features: Özellik vektörü
            
        Returns:
            Ensemble tahmin sonuçları
        """
        try:
            if features is None or len(features) == 0:
                return {'confidence': 0.5, 'uncertainty': 1.0, 'methods': []}
            
            # Özellik vektörünü reshape et
            X = features.reshape(1, -1)
            
            # Farklı tahmin yöntemleri
            predictions = {}
            
            # 1. Özellik tabanlı tahmin (mevcut yöntem)
            feature_pred = self._simple_feature_based_prediction(features)
            predictions['feature_based'] = feature_pred
            
            # 2. İstatistiksel analiz
            statistical_pred = self._statistical_analysis(features)
            predictions['statistical'] = statistical_pred
            
            # 3. Anomali tespiti
            anomaly_pred = self._anomaly_detection(X)
            predictions['anomaly'] = anomaly_pred
            
            # 4. Entropi tabanlı analiz
            entropy_pred = self._entropy_analysis(features)
            predictions['entropy'] = entropy_pred
            
            # Ensemble sonucu
            confidences = list(predictions.values())
            
            # Ağırlıklı ortalama
            weights = [0.4, 0.25, 0.2, 0.15]  # Özellik tabanlı en yüksek ağırlık
            ensemble_confidence = np.average(confidences, weights=weights)
            
            # Belirsizlik hesapla (standart sapma)
            uncertainty = np.std(confidences)
            
            # Tutarlılık skoru
            consistency = 1.0 - uncertainty
            
            return {
                'confidence': ensemble_confidence,
                'uncertainty': uncertainty,
                'consistency': consistency,
                'methods': predictions,
                'weights_used': weights
            }
            
        except Exception as e:
            logger.error(f"Ensemble tahmin hatası: {e}")
            return {'confidence': 0.5, 'uncertainty': 1.0, 'methods': []}
    
    def _statistical_analysis(self, features: np.ndarray) -> float:
        """
        İstatistiksel analiz
        
        Args:
            features: Özellik vektörü
            
        Returns:
            İstatistiksel tahmin skoru
        """
        try:
            if features is None or len(features) == 0:
                return 0.5
            
            # İstatistiksel özellikler
            mean_val = np.mean(features)
            std_val = np.std(features)
            skewness = stats.skew(features)
            kurtosis = stats.kurtosis(features)
            
            # Z-score analizi
            z_scores = np.abs(stats.zscore(features))
            outlier_ratio = np.sum(z_scores > 2) / len(features)
            
            # Normallik testi
            _, p_value = stats.normaltest(features)
            normality = 1.0 - p_value  # P değeri düşükse normallik düşük
            
            # Skorları birleştir
            score = 0.0
            
            # Yüksek standart sapma şüpheli
            score += min(0.3, std_val * 0.5)
            
            # Yüksek çarpıklık şüpheli
            score += min(0.2, abs(skewness) * 0.1)
            
            # Yüksek basıklık şüpheli
            score += min(0.2, abs(kurtosis) * 0.05)
            
            # Yüksek outlier oranı şüpheli
            score += min(0.2, outlier_ratio * 2)
            
            # Düşük normallik şüpheli
            score += min(0.1, normality * 0.2)
            
            return min(0.95, max(0.05, score))
            
        except Exception as e:
            logger.error(f"İstatistiksel analiz hatası: {e}")
            return 0.5
    
    def _anomaly_detection(self, X: np.ndarray) -> float:
        """
        Anomali tespiti
        
        Args:
            X: Özellik matrisi
            
        Returns:
            Anomali skoru
        """
        try:
            # Isolation Forest ile anomali tespiti
            anomaly_score = self.anomaly_detector.decision_function(X)[0]
            
            # Skoru 0-1 aralığına normalize et
            # Negatif değer = anomali, pozitif değer = normal
            normalized_score = 1.0 / (1.0 + np.exp(anomaly_score))
            
            return normalized_score
            
        except Exception as e:
            logger.error(f"Anomali tespit hatası: {e}")
            return 0.5
    
    def _entropy_analysis(self, features: np.ndarray) -> float:
        """
        Entropi tabanlı analiz
        
        Args:
            features: Özellik vektörü
            
        Returns:
            Entropi skoru
        """
        try:
            if features is None or len(features) == 0:
                return 0.5
            
            # Histogram oluştur
            hist, _ = np.histogram(features, bins=50, density=True)
            hist = hist + 1e-8  # Sıfır değerlerden kaçın
            
            # Shannon entropisi
            entropy = -np.sum(hist * np.log2(hist))
            
            # Maksimum entropi (uniform dağılım)
            max_entropy = np.log2(len(hist))
            
            # Normalize et
            normalized_entropy = entropy / max_entropy
            
            # Düşük entropi şüpheli (manipülasyon sonucu düzen)
            score = 1.0 - normalized_entropy
            
            return min(0.95, max(0.05, score))
            
        except Exception as e:
            logger.error(f"Entropi analizi hatası: {e}")
            return 0.5

    def analyze_image(self, image: np.ndarray) -> Dict:
        """
        Ultra gelişmiş görüntü analizi - Ensemble + Belirsizlik + Çoklu Ölçek
        
        Args:
            image: Analiz edilecek görüntü
            
        Returns:
            Kapsamlı analiz sonuçları
        """
        start_time = time.time()
        
        try:
            logger.info("🔍 Gelişmiş analiz başlatılıyor...")
            
            # 1. Çoklu ölçek analizi
            logger.info("📏 Çoklu ölçek analizi yapılıyor...")
            multi_scale_results = self._multi_scale_analysis(image)
            
            # 2. Yüz tespiti
            logger.info("👤 Yüz tespiti yapılıyor...")
            faces = self.detect_faces(image)
            
            # 3. Görüntü özelliklerini çıkar
            logger.info("🧩 Görüntü özellikleri çıkarılıyor...")
            image_features = self._extract_image_features(image)
            
            # 4. Ensemble tahmin (görüntü için)
            logger.info("🎯 Ensemble analizi yapılıyor...")
            ensemble_result = self._ensemble_prediction(image_features)
            
            # 5. Yüz analizi (varsa)
            face_analysis = None
            if faces:
                logger.info(f"👥 {len(faces)} yüz tespit edildi, analiz ediliyor...")
                # En büyük yüzü seç
                largest_face = max(faces, key=lambda f: f['bbox'][2] * f['bbox'][3])
                face_features = self.extract_face_features(image, largest_face['bbox'])
                
                if face_features is not None:
                    face_analysis = self._ensemble_prediction(face_features)
                    logger.info("✅ Yüz analizi tamamlandı")
                else:
                    logger.warning("⚠️ Yüz özellikleri çıkarılamadı")
            else:
                logger.info("👤 Yüz tespit edilemedi, sadece görüntü analizi yapılıyor")
            
            # 6. Sonuçları birleştir
            logger.info("🔗 Sonuçlar birleştiriliyor...")
            final_confidence = ensemble_result['confidence']
            final_uncertainty = ensemble_result['uncertainty']
            
            # Yüz analizi varsa ağırlıklı ortalama al
            if face_analysis:
                # %60 yüz analizi, %40 görüntü analizi
                final_confidence = 0.6 * face_analysis['confidence'] + 0.4 * ensemble_result['confidence']
                final_uncertainty = 0.6 * face_analysis['uncertainty'] + 0.4 * ensemble_result['uncertainty']
                logger.info(f"🔄 Hibrit analiz: Yüz(%60) + Görüntü(%40)")
            
            # 7. Çoklu ölçek tutarlılığı kontrol et
            scale_consistency = self._calculate_scale_consistency(multi_scale_results)
            logger.info(f"📊 Ölçek tutarlılığı: {scale_consistency:.2f}")
            
            # 8. Güvenilirlik skoru hesapla
            reliability_score = self._calculate_reliability(
                final_uncertainty, 
                scale_consistency, 
                ensemble_result['consistency']
            )
            logger.info(f"🎖️ Güvenilirlik skoru: {reliability_score:.2f}")
            
            # 9. Son karar ver
            # Yüksek belirsizlik varsa daha muhafazakar ol
            adjusted_confidence = final_confidence
            if final_uncertainty > 0.3:
                adjustment_factor = 1 - final_uncertainty * 0.5
                adjusted_confidence *= adjustment_factor
                logger.info(f"⚖️ Belirsizlik ayarlaması yapıldı: {adjustment_factor:.2f}")
            
            # Düşük güvenilirlik varsa eşiği yükselt
            decision_threshold = 0.5
            if reliability_score < 0.7:
                decision_threshold = 0.6
                logger.info("🔒 Düşük güvenilirlik - Eşik %60'a yükseltildi")
            elif reliability_score < 0.5:
                decision_threshold = 0.7
                logger.info("🔒 Çok düşük güvenilirlik - Eşik %70'e yükseltildi")
            
            is_fake = adjusted_confidence > decision_threshold
            
            # 10. Sonuç kategorisi belirle
            if reliability_score >= 0.8:
                result_category = "Yüksek Güvenilirlik"
                category_icon = "🟢"
            elif reliability_score >= 0.6:
                result_category = "Orta Güvenilirlik"
                category_icon = "🟡"
            else:
                result_category = "Düşük Güvenilirlik"
                category_icon = "🔴"
            
            logger.info(f"✅ Analiz tamamlandı - {category_icon} {result_category}")
            logger.info(f"📊 Sonuç: {'SAHTE' if is_fake else 'GERÇEK'} (Güven: {final_confidence:.1%})")
            
            return {
                'is_fake': is_fake,
                'confidence': final_confidence,
                'adjusted_confidence': adjusted_confidence,
                'uncertainty': final_uncertainty,
                'reliability_score': reliability_score,
                'scale_consistency': scale_consistency,
                'decision_threshold': decision_threshold,
                'result_category': result_category,
                'category_icon': category_icon,
                'face_detected': len(faces) > 0,
                'face_count': len(faces),
                'analysis_methods': {
                    'ensemble': ensemble_result,
                    'face_analysis': face_analysis,
                    'multi_scale': multi_scale_results
                },
                'analysis_time': time.time() - start_time,
                'quality_metrics': {
                    'certainty': 1.0 - final_uncertainty,
                    'consistency': ensemble_result['consistency'],
                    'reliability': reliability_score
                },
                'technical_details': {
                    'methods_used': ['ensemble', 'multi_scale', 'uncertainty_estimation'],
                    'features_extracted': len(image_features) if image_features is not None else 0,
                    'ensemble_methods': list(ensemble_result['methods'].keys()) if 'methods' in ensemble_result else [],
                    'adjustment_applied': final_uncertainty > 0.3,
                    'threshold_elevated': decision_threshold > 0.5
                }
            }
                
        except Exception as e:
            logger.error(f"❌ Görüntü analiz hatası: {e}")
            # Hata durumunda basit fallback
            return {
                'is_fake': False,
                'confidence': 0.3,
                'adjusted_confidence': 0.3,
                'uncertainty': 1.0,
                'reliability_score': 0.0,
                'scale_consistency': 0.0,
                'decision_threshold': 0.5,
                'result_category': "Hata",
                'category_icon': "❌",
                'face_detected': False,
                'face_count': 0,
                'analysis_time': time.time() - start_time,
                'quality_metrics': {
                    'certainty': 0.0,
                    'consistency': 0.0,
                    'reliability': 0.0
                },
                'error': str(e)
            }
    
    def _predict_with_models(self, image: np.ndarray, face_features: np.ndarray) -> float:
        """
        Modellerle tahmin yap
        
        Args:
            image: Görüntü
            face_features: Yüz özellikleri
            
        Returns:
            Tahmin güven skoru
        """
        predictions = []
        
        try:
            # XceptionNet tahmini
            if 'xception' in self.models:
                preprocessed = self.preprocess_image(image, (299, 299))
                if preprocessed is not None:
                    pred = self.models['xception'].predict(preprocessed, verbose=0)
                    predictions.append(pred[0][0])
            
            # FaceForensics++ tahmini
            if 'faceforensics' in self.models:
                preprocessed = self.preprocess_image(image, (224, 224))
                if preprocessed is not None:
                    pred = self.models['faceforensics'].predict(preprocessed, verbose=0)
                    predictions.append(pred[0][0])
            
            # GANDetector tahmini
            if 'gan_detector' in self.models:
                preprocessed = self.preprocess_image(image, (224, 224))
                if preprocessed is not None:
                    # PyTorch formatına çevir
                    tensor = torch.from_numpy(preprocessed.transpose(0, 3, 1, 2)).float().to(self.device)
                    with torch.no_grad():
                        pred = self.models['gan_detector'](tensor)
                        predictions.append(pred.cpu().numpy()[0][0])
            
            # Basit model tahmini
            if 'simple' in self.models:
                preprocessed = self.preprocess_image(image, (224, 224))
                if preprocessed is not None:
                    pred = self.models['simple'].predict(preprocessed, verbose=0)
                    predictions.append(pred[0][0])
            
            # Ortalama tahmin
            if predictions:
                return np.mean(predictions)
            else:
                # Fallback: gelişmiş yüz özellik tabanlı tahmin
                return self._advanced_face_prediction(face_features)
                
        except Exception as e:
            logger.error(f"Model tahmin hatası: {e}")
            # Hata durumunda özellik tabanlı tahmin yap
            if face_features is not None:
                return self._advanced_face_prediction(face_features)
            else:
                return 0.5
    
    def _advanced_face_prediction(self, face_features: np.ndarray) -> float:
        """
        Ultra gelişmiş yüz özellik tabanlı tahmin
        
        Args:
            face_features: Yüz özellik vektörü
            
        Returns:
            Tahmin skoru
        """
        try:
            if face_features is not None and len(face_features) >= 2048:
                # Özellik gruplarını ayır
                color_features = face_features[:512]  # Renk histogramı
                gray_features = face_features[512:768]  # Gri histogram
                grad_mag_features = face_features[768:896]  # Gradient magnitude
                grad_dir_features = face_features[896:1024]  # Gradient direction
                laplacian_features = face_features[1024:1152]  # Laplacian
                lbp_features = face_features[1152:1408]  # LBP
                symmetry_features = face_features[1408:1536]  # Simetri
                region_features = face_features[1536:1792]  # Bölge analizi
                edge_features = face_features[1792:1920]  # Kenar analizi
                texture_features = face_features[1920:2048]  # Doku analizi
                
                score = 0.0
                
                # 1. Renk analizi (0-0.12)
                color_entropy = -np.sum(color_features * np.log(color_features + 1e-8))
                color_variance = np.var(color_features)
                color_score = max(0, 0.12 - color_entropy * 0.03) + min(0.06, color_variance * 0.5)
                score += color_score
                
                # 2. Gri tonlama analizi (0-0.10)
                gray_entropy = -np.sum(gray_features * np.log(gray_features + 1e-8))
                gray_variance = np.var(gray_features)
                gray_score = max(0, 0.10 - gray_entropy * 0.03) + min(0.05, gray_variance * 0.3)
                score += gray_score
                
                # 3. Gradient magnitude analizi (0-0.10)
                grad_mag_mean = np.mean(grad_mag_features)
                grad_mag_variance = np.var(grad_mag_features)
                grad_mag_score = max(0, 0.10 - grad_mag_mean * 0.2) + min(0.05, grad_mag_variance * 0.4)
                score += grad_mag_score
                
                # 4. Gradient direction analizi (0-0.08)
                grad_dir_entropy = -np.sum(grad_dir_features * np.log(grad_dir_features + 1e-8))
                grad_dir_score = max(0, 0.08 - grad_dir_entropy * 0.02)
                score += grad_dir_score
                
                # 5. Laplacian analizi (0-0.10)
                laplacian_variance = np.var(laplacian_features)
                laplacian_mean = np.mean(laplacian_features)
                laplacian_score = min(0.10, laplacian_variance * 8) + max(0, 0.05 - laplacian_mean * 0.1)
                score += laplacian_score
                
                # 6. LBP analizi (0-0.10)
                lbp_entropy = -np.sum(lbp_features * np.log(lbp_features + 1e-8))
                lbp_variance = np.var(lbp_features)
                lbp_score = max(0, 0.10 - lbp_entropy * 0.03) + min(0.05, lbp_variance * 0.3)
                score += lbp_score
                
                # 7. Simetri analizi (0-0.12)
                symmetry_mean = symmetry_features[0] if len(symmetry_features) > 0 else 0
                symmetry_std = symmetry_features[1] if len(symmetry_features) > 1 else 0
                symmetry_score = min(0.12, symmetry_mean * 1.5 + symmetry_std * 0.8)
                score += symmetry_score
                
                # 8. Bölge analizi (0-0.10)
                region_variance = np.var(region_features)
                region_entropy = -np.sum(region_features * np.log(region_features + 1e-8))
                region_score = min(0.10, region_variance * 0.5 + region_entropy * 0.01)
                score += region_score
                
                # 9. Kenar analizi (0-0.08)
                edge_density = edge_features[0] if len(edge_features) > 0 else 0
                edge_variance = np.var(edge_features)
                edge_score = min(0.08, edge_density * 0.4 + edge_variance * 0.3)
                score += edge_score
                
                # 10. Doku analizi (0-0.10)
                texture_variance = np.var(texture_features)
                texture_entropy = -np.sum(texture_features * np.log(texture_features + 1e-8))
                texture_score = min(0.10, texture_variance * 0.4 + texture_entropy * 0.01)
                score += texture_score
                
                # 11. Yüz özel tutarlılık kontrolü (0-0.06)
                feature_correlation = np.corrcoef(face_features.reshape(-1, 1), np.arange(len(face_features)))[0, 1]
                consistency_score = min(0.06, abs(feature_correlation) * 0.15)
                score += consistency_score
                
                # 12. Yüz anomali tespiti (0-0.06)
                z_scores = np.abs((face_features - np.mean(face_features)) / (np.std(face_features) + 1e-8))
                anomaly_score = min(0.06, np.sum(z_scores > 3) / len(face_features) * 0.3)
                score += anomaly_score
                
                # Sonuç normalizasyonu ve kalibrasyon
                final_score = min(0.95, max(0.05, score))
                
                # Yüz için özel kalibrasyon
                if final_score < 0.20:
                    final_score *= 0.5  # Gerçek yüzler için daha düşük
                elif final_score > 0.80:
                    final_score = 0.80 + (final_score - 0.80) * 2.0  # Sahte yüzler için daha yüksek
                elif final_score > 0.60:
                    final_score = 0.60 + (final_score - 0.60) * 1.3  # Orta seviye manipülasyonlar için
                
                return final_score
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Ultra gelişmiş yüz tahmin hatası: {e}")
            return 0.5
    
    def _extract_image_features(self, image: np.ndarray) -> np.ndarray:
        """
        Ultra gelişmiş görüntü özellik çıkarma
        
        Args:
            image: Görüntü
            
        Returns:
            Özellik vektörü
        """
        try:
            # Görüntüyü küçült
            resized = cv2.resize(image, (256, 256))
            
            # Gri tonlamaya çevir
            if len(resized.shape) == 3:
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            else:
                gray = resized
            
            features = []
            
            # 1. Histogram özellikleri (256)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist.flatten() / (hist.sum() + 1e-8)
            features.extend(hist)
            
            # 2. Gradient özellikleri
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            gradient_direction = np.arctan2(grad_y, grad_x)
            
            # Gradient magnitude histogramı (128)
            grad_mag_hist = cv2.calcHist([gradient_magnitude.astype(np.uint8)], [0], None, [128], [0, 256])
            grad_mag_hist = grad_mag_hist.flatten() / (grad_mag_hist.sum() + 1e-8)
            features.extend(grad_mag_hist)
            
            # Gradient direction histogramı (64)
            grad_dir_hist = cv2.calcHist([gradient_direction.astype(np.uint8)], [0], None, [64], [0, 256])
            grad_dir_hist = grad_dir_hist.flatten() / (grad_dir_hist.sum() + 1e-8)
            features.extend(grad_dir_hist)
            
            # 3. Laplacian özellikleri (128)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian_hist = cv2.calcHist([laplacian.astype(np.uint8)], [0], None, [128], [0, 256])
            laplacian_hist = laplacian_hist.flatten() / (laplacian_hist.sum() + 1e-8)
            features.extend(laplacian_hist)
            
            # 4. Gabor filtreleri (256)
            gabor_features = []
            angles = [0, 30, 60, 90, 120, 150]
            frequencies = [0.1, 0.2, 0.3, 0.4, 0.5]
            
            for angle in angles:
                for freq in frequencies:
                    kernel = cv2.getGaborKernel((31, 31), 5, np.radians(angle), 2*np.pi*freq, 0.5, 0, ktype=cv2.CV_32F)
                    filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
                    gabor_hist = cv2.calcHist([filtered], [0], None, [64], [0, 256])
                    gabor_hist = gabor_hist.flatten() / (gabor_hist.sum() + 1e-8)
                    gabor_features.extend(gabor_hist)
            
            features.extend(gabor_features)
            
            # 5. DCT (Discrete Cosine Transform) özellikleri (128)
            dct = cv2.dct(np.float32(gray))
            dct_features = dct[:16, :16].flatten()  # İlk 16x16 katsayı
            dct_features = dct_features / (np.linalg.norm(dct_features) + 1e-8)
            features.extend(dct_features)
            
            # 6. Yerel Binary Pattern (LBP) özellikleri (256)
            lbp = self._compute_lbp(gray)
            lbp_hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
            lbp_hist = lbp_hist.flatten() / (lbp_hist.sum() + 1e-8)
            features.extend(lbp_hist)
            
            # 7. Haralick özellikleri (14)
            haralick_features = self._compute_haralick_features(gray)
            features.extend(haralick_features)
            
            # 8. Tamura özellikleri (6)
            tamura_features = self._compute_tamura_features(gray)
            features.extend(tamura_features)
            
            # 9. Zernike momentleri (25)
            zernike_features = self._compute_zernike_moments(gray)
            features.extend(zernike_features)
            
            # 10. Renk özellikleri (RGB histogramı) (768)
            if len(resized.shape) == 3:
                color_features = self._extract_color_features(resized)
                features.extend(color_features)
            else:
                features.extend([0.0] * 768)  # Gri görüntü için boş renk özellikleri
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Özellik çıkarma hatası: {e}")
            # Hata durumunda basit özellik vektörü
            return np.ones(2048) * 0.01  # Güncellenmiş boyut
    
    def _compute_lbp(self, image: np.ndarray) -> np.ndarray:
        """
        Yerel Binary Pattern hesapla
        
        Args:
            image: Gri tonlamalı görüntü
            
        Returns:
            LBP görüntüsü
        """
        try:
            lbp = np.zeros_like(image)
            for i in range(1, image.shape[0]-1):
                for j in range(1, image.shape[1]-1):
                    center = image[i, j]
                    code = 0
                    # 8 komşu piksel
                    neighbors = [
                        image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                        image[i, j+1], image[i+1, j+1], image[i+1, j],
                        image[i+1, j-1], image[i, j-1]
                    ]
                    for k, neighbor in enumerate(neighbors):
                        if neighbor >= center:
                            code |= (1 << k)
                    lbp[i, j] = code
            return lbp.astype(np.uint8)
        except Exception as e:
            logger.error(f"LBP hesaplama hatası: {e}")
            return np.zeros_like(image, dtype=np.uint8)
    
    def _compute_haralick_features(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Haralick doku özelliklerini hesapla
        
        Args:
            gray_image: Gri tonlamalı görüntü
            
        Returns:
            Haralick özellikleri
        """
        try:
            # Görüntüyü küçült (GLCM hesaplaması için)
            small_image = cv2.resize(gray_image, (64, 64))
            
            # GLCM (Gray Level Co-occurrence Matrix) hesapla
            glcm = self._compute_glcm(small_image)
            
            features = []
            
            # 1. Contrast (Kontrast)
            contrast = np.sum(glcm * np.square(np.arange(glcm.shape[0])[:, None] - np.arange(glcm.shape[1])))
            features.append(contrast)
            
            # 2. Dissimilarity (Farklılık)
            dissimilarity = np.sum(glcm * np.abs(np.arange(glcm.shape[0])[:, None] - np.arange(glcm.shape[1])))
            features.append(dissimilarity)
            
            # 3. Homogeneity (Homojenlik)
            homogeneity = np.sum(glcm / (1 + np.square(np.arange(glcm.shape[0])[:, None] - np.arange(glcm.shape[1]))))
            features.append(homogeneity)
            
            # 4. Energy (Enerji)
            energy = np.sum(glcm ** 2)
            features.append(energy)
            
            # 5. Correlation (Korelasyon)
            mean_i = np.sum(glcm * np.arange(glcm.shape[0])[:, None])
            mean_j = np.sum(glcm * np.arange(glcm.shape[1]))
            std_i = np.sqrt(np.sum(glcm * np.square(np.arange(glcm.shape[0])[:, None] - mean_i)))
            std_j = np.sqrt(np.sum(glcm * np.square(np.arange(glcm.shape[1]) - mean_j)))
            correlation = np.sum(glcm * (np.arange(glcm.shape[0])[:, None] - mean_i) * (np.arange(glcm.shape[1]) - mean_j)) / (std_i * std_j + 1e-8)
            features.append(correlation)
            
            # 6. ASM (Angular Second Moment)
            asm = np.sum(glcm ** 2)
            features.append(asm)
            
            # 7. Entropy (Entropi)
            entropy = -np.sum(glcm * np.log(glcm + 1e-8))
            features.append(entropy)
            
            # 8. Variance (Varyans)
            variance = np.sum(glcm * np.square(np.arange(glcm.shape[0])[:, None] - mean_i))
            features.append(variance)
            
            # 9. Sum Average
            sum_avg = np.sum(glcm * (np.arange(glcm.shape[0])[:, None] + np.arange(glcm.shape[1])))
            features.append(sum_avg)
            
            # 10. Sum Variance
            sum_var = np.sum(glcm * np.square(np.arange(glcm.shape[0])[:, None] + np.arange(glcm.shape[1]) - sum_avg))
            features.append(sum_var)
            
            # 11. Sum Entropy
            sum_entropy = -np.sum(glcm * np.log(glcm + 1e-8))
            features.append(sum_entropy)
            
            # 12. Difference Variance
            diff_var = np.sum(glcm * np.square(np.arange(glcm.shape[0])[:, None] - np.arange(glcm.shape[1])))
            features.append(diff_var)
            
            # 13. Difference Entropy
            diff_entropy = -np.sum(glcm * np.log(glcm + 1e-8))
            features.append(diff_entropy)
            
            # 14. Information Measure of Correlation
            info_corr = (entropy - sum_entropy) / (entropy + 1e-8)
            features.append(info_corr)
            
            # Normalize features
            features = np.array(features)
            features = (features - np.mean(features)) / (np.std(features) + 1e-8)
            
            return features
            
        except Exception as e:
            logger.error(f"Haralick özellik hesaplama hatası: {e}")
            return np.zeros(14)
    
    def _compute_glcm(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Gray Level Co-occurrence Matrix hesapla
        
        Args:
            gray_image: Gri tonlamalı görüntü
            
        Returns:
            GLCM matrisi
        """
        try:
            # Görüntüyü 8 seviyeye indir
            levels = 8
            quantized = (gray_image * levels / 256).astype(np.uint8)
            
            # GLCM hesapla (sağa doğru)
            glcm = np.zeros((levels, levels))
            h, w = quantized.shape
            
            for i in range(h):
                for j in range(w-1):
                    glcm[quantized[i, j], quantized[i, j+1]] += 1
            
            # Normalize
            glcm = glcm / (glcm.sum() + 1e-8)
            
            return glcm
            
        except Exception as e:
            logger.error(f"GLCM hesaplama hatası: {e}")
            return np.eye(8) / 8
    
    def _compute_tamura_features(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Tamura doku özelliklerini hesapla
        
        Args:
            gray_image: Gri tonlamalı görüntü
            
        Returns:
            Tamura özellikleri
        """
        try:
            features = []
            
            # 1. Coarseness (Kaba doku)
            coarseness = self._compute_coarseness(gray_image)
            features.append(coarseness)
            
            # 2. Contrast (Kontrast)
            contrast = self._compute_contrast(gray_image)
            features.append(contrast)
            
            # 3. Directionality (Yönlülük)
            directionality = self._compute_directionality(gray_image)
            features.append(directionality)
            
            # 4. Line-likeness (Çizgi benzerliği)
            line_likeness = self._compute_line_likeness(gray_image)
            features.append(line_likeness)
            
            # 5. Regularity (Düzenlilik)
            regularity = self._compute_regularity(gray_image)
            features.append(regularity)
            
            # 6. Roughness (Pürüzlülük)
            roughness = coarseness + contrast
            features.append(roughness)
            
            # Normalize
            features = np.array(features)
            features = (features - np.mean(features)) / (np.std(features) + 1e-8)
            
            return features
            
        except Exception as e:
            logger.error(f"Tamura özellik hesaplama hatası: {e}")
            return np.zeros(6)
    
    def _compute_coarseness(self, image: np.ndarray) -> float:
        """Coarseness hesapla"""
        try:
            h, w = image.shape
            best_sizes = np.zeros((h, w))
            
            for size in [2, 4, 8, 16]:
                if size >= min(h, w):
                    break
                    
                # Ortalama hesapla
                kernel = np.ones((size, size)) / (size * size)
                avg = cv2.filter2D(image.astype(np.float32), -1, kernel)
                
                # Farkları hesapla
                diff = np.abs(image.astype(np.float32) - avg)
                best_sizes = np.maximum(best_sizes, diff)
            
            return np.mean(best_sizes)
        except:
            return 0.0
    
    def _compute_contrast(self, image: np.ndarray) -> float:
        """Contrast hesapla"""
        try:
            return np.std(image)
        except:
            return 0.0
    
    def _compute_directionality(self, image: np.ndarray) -> float:
        """Directionality hesapla"""
        try:
            # Gradient hesapla
            grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            
            # Gradient yönü
            angle = np.arctan2(grad_y, grad_x)
            
            # Histogram
            hist, _ = np.histogram(angle.flatten(), bins=16, range=(-np.pi, np.pi))
            hist = hist / hist.sum()
            
            # Peak sayısı
            peaks = 0
            for i in range(1, len(hist)-1):
                if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > 0.1:
                    peaks += 1
            
            return 1.0 - peaks / 16.0
        except:
            return 0.0
    
    def _compute_line_likeness(self, image: np.ndarray) -> float:
        """Line-likeness hesapla"""
        try:
            # Gradient magnitude
            grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            return np.mean(magnitude)
        except:
            return 0.0
    
    def _compute_regularity(self, image: np.ndarray) -> float:
        """Regularity hesapla"""
        try:
            # Basit düzenlilik ölçüsü
            return 1.0 - np.std(image) / 255.0
        except:
            return 0.0
    
    def _compute_zernike_moments(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Zernike momentlerini hesapla
        
        Args:
            gray_image: Gri tonlamalı görüntü
            
        Returns:
            Zernike momentleri
        """
        try:
            # Görüntüyü küçült
            small_image = cv2.resize(gray_image, (32, 32))
            
            # Dairesel bölge
            center = 16
            radius = 15
            
            moments = []
            
            # Basit Zernike momentleri (5x5 = 25 moment)
            for n in range(5):
                for m in range(-n, n+1, 2):
                    if abs(m) <= n:
                        moment = self._compute_single_zernike_moment(small_image, n, m, center, radius)
                        moments.append(abs(moment))
            
            # Normalize
            moments = np.array(moments)
            if len(moments) > 0:
                moments = moments / (np.linalg.norm(moments) + 1e-8)
            
            # 25 momente tamamla
            while len(moments) < 25:
                moments = np.append(moments, 0.0)
            
            return moments[:25]
            
        except Exception as e:
            logger.error(f"Zernike moment hesaplama hatası: {e}")
            return np.zeros(25)
    
    def _compute_single_zernike_moment(self, image: np.ndarray, n: int, m: int, center: int, radius: int) -> complex:
        """Tek Zernike momenti hesapla"""
        try:
            moment = 0.0
            count = 0
            
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    # Dairesel koordinatlar
                    x = (i - center) / radius
                    y = (j - center) / radius
                    r = np.sqrt(x*x + y*y)
                    
                    if r <= 1.0:
                        theta = np.arctan2(y, x)
                        
                        # Zernike polinomu (basitleştirilmiş)
                        if n == 0:
                            R = 1.0
                        elif n == 1:
                            R = r
                        elif n == 2:
                            R = 2*r*r - 1
                        elif n == 3:
                            R = 3*r*r*r - 2*r
                        elif n == 4:
                            R = 6*r*r*r*r - 6*r*r + 1
                        else:
                            R = 0.0
                        
                        # Açısal kısım
                        if m == 0:
                            angular = 1.0
                        elif m > 0:
                            angular = np.cos(m * theta)
                        else:
                            angular = np.sin(-m * theta)
                        
                        moment += image[i, j] * R * angular
                        count += 1
            
            if count > 0:
                moment = moment / count
            
            return moment
            
        except:
            return 0.0
    
    def _extract_color_features(self, color_image: np.ndarray) -> np.ndarray:
        """
        Renk özelliklerini çıkar
        
        Args:
            color_image: Renkli görüntü
            
        Returns:
            Renk özellikleri
        """
        try:
            features = []
            
            # RGB histogramları (256 x 3 = 768)
            for channel in range(3):
                hist = cv2.calcHist([color_image], [channel], None, [256], [0, 256])
                hist = hist.flatten() / (hist.sum() + 1e-8)
                features.extend(hist)
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Renk özellik çıkarma hatası: {e}")
            return np.zeros(768)
    
    def _simple_feature_based_prediction(self, features: np.ndarray) -> float:
        """
        Ultra gelişmiş özellik tabanlı tahmin
        
        Args:
            features: Özellik vektörü
            
        Returns:
            Tahmin skoru
        """
        try:
            if features is not None and len(features) >= 2048:
                # Özellik gruplarını ayır
                hist_features = features[:256]  # Histogram
                grad_mag_features = features[256:384]  # Gradient magnitude
                grad_dir_features = features[384:448]  # Gradient direction
                laplacian_features = features[448:576]  # Laplacian
                gabor_features = features[576:832]  # Gabor
                dct_features = features[832:960]  # DCT
                lbp_features = features[960:1216]  # LBP
                haralick_features = features[1216:1230]  # Haralick
                tamura_features = features[1230:1236]  # Tamura
                zernike_features = features[1236:1261]  # Zernike
                color_features = features[1261:2029]  # Color
                
                score = 0.0
                
                # 1. Histogram analizi (0-0.08)
                hist_entropy = -np.sum(hist_features * np.log(hist_features + 1e-8))
                hist_variance = np.var(hist_features)
                hist_score = max(0, 0.08 - hist_entropy * 0.03)
                score += hist_score
                
                # 2. Gradient magnitude analizi (0-0.08)
                grad_mag_mean = np.mean(grad_mag_features)
                grad_mag_variance = np.var(grad_mag_features)
                grad_mag_score = max(0, 0.08 - grad_mag_mean * 0.2)
                score += grad_mag_score
                
                # 3. Gradient direction analizi (0-0.06)
                grad_dir_entropy = -np.sum(grad_dir_features * np.log(grad_dir_features + 1e-8))
                grad_dir_score = max(0, 0.06 - grad_dir_entropy * 0.02)
                score += grad_dir_score
                
                # 4. Laplacian analizi (0-0.08)
                laplacian_variance = np.var(laplacian_features)
                laplacian_score = min(0.08, laplacian_variance * 8)
                score += laplacian_score
                
                # 5. Gabor filtre analizi (0-0.08)
                gabor_std = np.std(gabor_features)
                gabor_entropy = -np.sum(gabor_features * np.log(gabor_features + 1e-8))
                gabor_score = min(0.08, gabor_std * 3 + gabor_entropy * 0.01)
                score += gabor_score
                
                # 6. DCT analizi (0-0.06)
                dct_energy = np.sum(dct_features**2)
                dct_variance = np.var(dct_features)
                dct_score = min(0.06, dct_energy * 0.05 + dct_variance * 2)
                score += dct_score
                
                # 7. LBP analizi (0-0.08)
                lbp_entropy = -np.sum(lbp_features * np.log(lbp_features + 1e-8))
                lbp_variance = np.var(lbp_features)
                lbp_score = max(0, 0.08 - lbp_entropy * 0.03) + min(0.04, lbp_variance * 2)
                score += lbp_score
                
                # 8. Haralick analizi (0-0.08)
                haralick_contrast = haralick_features[0] if len(haralick_features) > 0 else 0
                haralick_energy = haralick_features[3] if len(haralick_features) > 3 else 0
                haralick_score = min(0.08, haralick_contrast * 0.5 + (1 - haralick_energy) * 0.3)
                score += haralick_score
                
                # 9. Tamura analizi (0-0.06)
                tamura_coarseness = tamura_features[0] if len(tamura_features) > 0 else 0
                tamura_contrast = tamura_features[1] if len(tamura_features) > 1 else 0
                tamura_score = min(0.06, tamura_coarseness * 0.3 + tamura_contrast * 0.2)
                score += tamura_score
                
                # 10. Zernike analizi (0-0.06)
                zernike_mean = np.mean(zernike_features)
                zernike_variance = np.var(zernike_features)
                zernike_score = min(0.06, zernike_mean * 0.4 + zernike_variance * 0.3)
                score += zernike_score
                
                # 11. Renk analizi (0-0.08)
                color_entropy = -np.sum(color_features * np.log(color_features + 1e-8))
                color_variance = np.var(color_features)
                color_score = max(0, 0.08 - color_entropy * 0.02) + min(0.04, color_variance * 0.5)
                score += color_score
                
                # 12. Genel tutarlılık kontrolü (0-0.06)
                feature_correlation = np.corrcoef(features.reshape(-1, 1), np.arange(len(features)))[0, 1]
                consistency_score = min(0.06, abs(feature_correlation) * 0.15)
                score += consistency_score
                
                # 13. Anomali tespiti (0-0.06)
                z_scores = np.abs((features - np.mean(features)) / (np.std(features) + 1e-8))
                anomaly_score = min(0.06, np.sum(z_scores > 3) / len(features) * 0.3)
                score += anomaly_score
                
                # 14. Doku tutarlılığı (0-0.06)
                texture_consistency = 1.0 - np.std([hist_variance, grad_mag_variance, laplacian_variance, lbp_variance])
                texture_score = min(0.06, texture_consistency * 0.3)
                score += texture_score
                
                # Sonuç normalizasyonu ve kalibrasyon
                final_score = min(0.95, max(0.05, score))
                
                # Gelişmiş kalibrasyon
                if final_score < 0.25:
                    final_score *= 0.6  # Gerçek fotoğraflar için daha düşük
                elif final_score > 0.75:
                    final_score = 0.75 + (final_score - 0.75) * 1.8  # Sahte fotoğraflar için daha yüksek
                elif final_score > 0.6:
                    final_score = 0.6 + (final_score - 0.6) * 1.2  # Orta seviye manipülasyonlar için
                
                return final_score
            else:
                # Özellik yoksa orta değer
                return 0.5
                
        except Exception as e:
            logger.error(f"Ultra gelişmiş tahmin hatası: {e}")
            return 0.5
    
    def analyze_video(self, video_path: str, max_frames: int = 30) -> Dict:
        """
        Video analizi
        
        Args:
            video_path: Video dosya yolu
            max_frames: Analiz edilecek maksimum frame sayısı
            
        Returns:
            Video analiz sonuçları
        """
        start_time = time.time()
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError("Video dosyası açılamadı")
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = frame_count / fps
            
            # Frame analizleri
            frame_results = []
            total_fake_frames = 0
            
            # Belirli aralıklarla frame'leri analiz et
            frame_interval = max(1, frame_count // max_frames)
            
            for i in range(0, min(frame_count, max_frames * frame_interval), frame_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                
                if ret:
                    # Frame analizi
                    result = self.analyze_image(frame)
                    
                    if result['face_detected'] and 'error' not in result:
                        frame_results.append({
                            'frame_number': i,
                            'timestamp': i / fps,
                            'confidence': result['confidence'],
                            'is_fake': result['is_fake'],
                            'faces_analyzed': result.get('faces_analyzed', 0)
                        })
                        
                        if result['is_fake']:
                            total_fake_frames += 1
            
            cap.release()
            
            # Genel video sonucu
            if frame_results:
                fake_percentage = (total_fake_frames / len(frame_results)) * 100
                overall_confidence = np.mean([f['confidence'] for f in frame_results])
                
                return {
                    'frame_results': frame_results,
                    'total_frames_analyzed': len(frame_results),
                    'total_video_frames': frame_count,
                    'fake_percentage': fake_percentage,
                    'overall_confidence': overall_confidence,
                    'is_fake': fake_percentage > 50,
                    'duration': duration,
                    'fps': fps,
                    'analysis_time': time.time() - start_time
                }
            else:
                return {
                    'frame_results': [],
                    'total_frames_analyzed': 0,
                    'total_video_frames': frame_count,
                    'fake_percentage': 0,
                    'overall_confidence': 0.3,  # Düşük güven skoru
                    'is_fake': False,
                    'duration': duration,
                    'fps': fps,
                    'analysis_time': time.time() - start_time,
                    'error': 'Hiçbir frame analiz edilemedi'
                }
                
        except Exception as e:
            logger.error(f"Video analiz hatası: {e}")
            return {
                'frame_results': [],
                'total_frames_analyzed': 0,
                'fake_percentage': 0,
                'overall_confidence': 0.3,  # Düşük güven skoru
                'is_fake': False,
                'analysis_time': time.time() - start_time,
                'error': str(e)
            }
    
    def _calculate_scale_consistency(self, scale_results: Dict) -> float:
        """
        Çoklu ölçek tutarlılığını hesapla
        
        Args:
            scale_results: Çoklu ölçek analiz sonuçları
            
        Returns:
            Tutarlılık skoru (0-1)
        """
        try:
            if not scale_results:
                return 0.0
            
            confidences = [result['confidence'] for result in scale_results.values()]
            
            if len(confidences) < 2:
                return 1.0
            
            # Standart sapma ile tutarlılık ölç
            consistency = 1.0 - np.std(confidences)
            return max(0.0, min(1.0, consistency))
            
        except Exception as e:
            logger.error(f"Ölçek tutarlılığı hesaplama hatası: {e}")
            return 0.0
    
    def _calculate_reliability(self, uncertainty: float, scale_consistency: float, ensemble_consistency: float) -> float:
        """
        Genel güvenilirlik skoru hesapla
        
        Args:
            uncertainty: Belirsizlik skoru
            scale_consistency: Ölçek tutarlılığı
            ensemble_consistency: Ensemble tutarlılığı
            
        Returns:
            Güvenilirlik skoru (0-1)
        """
        try:
            # Belirsizlik ne kadar düşükse güvenilirlik o kadar yüksek
            certainty = 1.0 - uncertainty
            
            # Ağırlıklı ortalama
            reliability = (
                0.4 * certainty +
                0.3 * scale_consistency +
                0.3 * ensemble_consistency
            )
            
            return max(0.0, min(1.0, reliability))
            
        except Exception as e:
            logger.error(f"Güvenilirlik hesaplama hatası: {e}")
            return 0.0

    def get_model_info(self) -> Dict:
        """Model bilgilerini döndür"""
        return {
            'model_type': self.model_type,
            'loaded_models': list(self.models.keys()),
            'device': str(self.device),
            'face_detection_available': self.face_detection is not None,
            'ensemble_methods': ['feature_based', 'statistical', 'anomaly', 'entropy'],
            'analysis_features': ['multi_scale', 'uncertainty_estimation', 'reliability_scoring']
        }
