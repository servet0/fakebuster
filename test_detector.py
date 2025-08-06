#!/usr/bin/env python3
"""
Deepfake Detector Test Script
Bu script, deepfake tespit sistemini test etmek için kullanılır.
"""

import cv2
import numpy as np
from PIL import Image
import os
import sys
from deepfake_detector import DeepfakeDetector
import time

def create_test_image(size=(512, 512)):
    """Test için basit bir görüntü oluştur"""
    # Basit bir test görüntüsü
    image = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    
    # Yüz benzeri bir şekil ekle
    center_x, center_y = size[0] // 2, size[1] // 2
    cv2.circle(image, (center_x, center_y), 80, (255, 200, 150), -1)  # Yüz
    cv2.circle(image, (center_x - 30, center_y - 20), 10, (0, 0, 0), -1)  # Sol göz
    cv2.circle(image, (center_x + 30, center_y - 20), 10, (0, 0, 0), -1)  # Sağ göz
    cv2.ellipse(image, (center_x, center_y + 20), (20, 10), 0, 0, 180, (0, 0, 0), 3)  # Ağız
    
    return image

def create_test_video(output_path="test_video.mp4", duration=5, fps=30):
    """Test için basit bir video oluştur"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (512, 512))
    
    for i in range(duration * fps):
        # Her frame için biraz farklı bir görüntü
        frame = create_test_image()
        
        # Frame'e timestamp ekle
        cv2.putText(frame, f"Frame: {i}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Test video oluşturuldu: {output_path}")

def test_photo_detection():
    """Fotoğraf tespit testi"""
    print("\n=== Fotoğraf Tespit Testi ===")
    
    # Test görüntüsü oluştur
    test_image = create_test_image()
    
    # Farklı modelleri test et
    models = ["XceptionNet", "FaceForensics++", "GANDetector", "Hibrit Model"]
    
    for model_type in models:
        print(f"\n--- {model_type} Modeli Test Ediliyor ---")
        
        try:
            # Detector'ı başlat
            detector = DeepfakeDetector(model_type)
            
            # Model bilgilerini göster
            model_info = detector.get_model_info()
            print(f"Model Bilgileri: {model_info}")
            
            # Görüntüyü analiz et
            start_time = time.time()
            result = detector.analyze_image(test_image)
            analysis_time = time.time() - start_time
            
            # Sonuçları göster
            print(f"Analiz Sonucu:")
            print(f"  - Sahte mi: {result['is_fake']}")
            print(f"  - Güven Skoru: {result['confidence']:.3f}")
            print(f"  - Yüz Tespit Edildi: {result['face_detected']}")
            print(f"  - Analiz Süresi: {analysis_time:.3f} saniye")
            
            if 'faces_analyzed' in result:
                print(f"  - Analiz Edilen Yüz Sayısı: {result['faces_analyzed']}")
            
            if 'error' in result:
                print(f"  - Hata: {result['error']}")
                
        except Exception as e:
            print(f"  Hata: {e}")

def test_video_detection():
    """Video tespit testi"""
    print("\n=== Video Tespit Testi ===")
    
    # Test video oluştur
    test_video_path = "test_video.mp4"
    if not os.path.exists(test_video_path):
        create_test_video(test_video_path)
    
    # Farklı modelleri test et
    models = ["XceptionNet", "FaceForensics++", "GANDetector", "Hibrit Model"]
    
    for model_type in models:
        print(f"\n--- {model_type} Modeli Video Test Ediliyor ---")
        
        try:
            # Detector'ı başlat
            detector = DeepfakeDetector(model_type)
            
            # Video analizi
            start_time = time.time()
            result = detector.analyze_video(test_video_path, max_frames=10)
            analysis_time = time.time() - start_time
            
            # Sonuçları göster
            print(f"Video Analiz Sonucu:")
            print(f"  - Sahte mi: {result['is_fake']}")
            print(f"  - Sahte Frame Yüzdesi: {result['fake_percentage']:.1f}%")
            print(f"  - Ortalama Güven Skoru: {result['overall_confidence']:.3f}")
            print(f"  - Analiz Edilen Frame: {result['total_frames_analyzed']}")
            print(f"  - Toplam Video Frame: {result['total_video_frames']}")
            print(f"  - Video Süresi: {result['duration']:.2f} saniye")
            print(f"  - FPS: {result['fps']:.1f}")
            print(f"  - Analiz Süresi: {analysis_time:.3f} saniye")
            
            if 'error' in result:
                print(f"  - Hata: {result['error']}")
                
        except Exception as e:
            print(f"  Hata: {e}")

def test_face_detection():
    """Yüz tespit testi"""
    print("\n=== Yüz Tespit Testi ===")
    
    # Test görüntüsü oluştur
    test_image = create_test_image()
    
    try:
        detector = DeepfakeDetector("XceptionNet")
        
        # Yüz tespiti
        faces = detector.detect_faces(test_image)
        
        print(f"Tespit Edilen Yüz Sayısı: {len(faces)}")
        
        for i, face in enumerate(faces):
            print(f"  Yüz {i+1}:")
            print(f"    - Bounding Box: {face['bbox']}")
            print(f"    - Güven: {face['confidence']:.3f}")
            
    except Exception as e:
        print(f"Yüz tespit hatası: {e}")

def test_model_loading():
    """Model yükleme testi"""
    print("\n=== Model Yükleme Testi ===")
    
    models = ["XceptionNet", "FaceForensics++", "GANDetector", "Hibrit Model"]
    
    for model_type in models:
        print(f"\n--- {model_type} Yükleniyor ---")
        
        try:
            detector = DeepfakeDetector(model_type)
            model_info = detector.get_model_info()
            
            print(f"  ✅ Başarıyla yüklendi")
            print(f"  - Yüklenen Modeller: {model_info['loaded_models']}")
            print(f"  - Cihaz: {model_info['device']}")
            print(f"  - Yüz Tespit: {model_info['face_detection_available']}")
            
        except Exception as e:
            print(f"  ❌ Yükleme hatası: {e}")

def cleanup_test_files():
    """Test dosyalarını temizle"""
    test_files = ["test_video.mp4"]
    
    for file_path in test_files:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Test dosyası silindi: {file_path}")
            except Exception as e:
                print(f"Dosya silme hatası {file_path}: {e}")

def main():
    """Ana test fonksiyonu"""
    print("🔍 FakeBuster Deepfake Detector Test Script")
    print("=" * 50)
    
    try:
        # Model yükleme testi
        test_model_loading()
        
        # Yüz tespit testi
        test_face_detection()
        
        # Fotoğraf tespit testi
        test_photo_detection()
        
        # Video tespit testi
        test_video_detection()
        
        print("\n" + "=" * 50)
        print("✅ Tüm testler tamamlandı!")
        
    except KeyboardInterrupt:
        print("\n\n❌ Test kullanıcı tarafından durduruldu")
    except Exception as e:
        print(f"\n\n❌ Test hatası: {e}")
    finally:
        # Test dosyalarını temizle
        print("\n🧹 Test dosyaları temizleniyor...")
        cleanup_test_files()

if __name__ == "__main__":
    main()
