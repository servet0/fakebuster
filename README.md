# FakeBuster - Deepfake ve AI Medya Tespit Aracı

Bu proje, deepfake ve AI ile oluşturulmuş video ve fotoğrafları tespit etmek için geliştirilmiş yerel bir prototiptir.

https://github.com/user-attachments/assets/66f835a7-8e66-402c-8b3d-9804764717dc

## Özellikler

- 📸 Fotoğraf analizi (Deepfake tespiti)
- 🎥 Video analizi (Frame-by-frame tespit)
- 🎯 Güven skoru hesaplama
- 🖥️ Streamlit web arayüzü
- 🔍 Çoklu model desteği

## Kurulum

1. Sanal ortamı aktifleştirin:
```bash
venv\Scripts\activate
```

2. Gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```

3. Uygulamayı çalıştırın:
```bash
streamlit run app.py
```

## Kullanım

1. Web arayüzünü açın (genellikle http://localhost:8501)
2. Fotoğraf veya video dosyası yükleyin
3. Analiz sonuçlarını görüntüleyin

## Model Bilgileri

- **XceptionNet**: Deepfake video tespiti için
- **FaceForensics++**: Yüz manipülasyonu tespiti
- **GANDetector**: AI üretimi görsel tespiti

## Geliştirme

Bu proje sürekli geliştirilmektedir. Yeni modeller ve özellikler eklenmektedir.
