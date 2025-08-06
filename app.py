import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from deepfake_detector import DeepfakeDetector
import matplotlib.pyplot as plt
import seaborn as sns

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="FakeBuster - Deepfake Tespit Aracı",
    page_icon="🔍",
    layout="wide"
)

# CSS stilleri
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .fake-result {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .real-result {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
    }
    .confidence-high {
        color: #4caf50;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ff9800;
        font-weight: bold;
    }
    .confidence-low {
        color: #f44336;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Ana başlık
    st.markdown('<h1 class="main-header">🔍 FakeBuster</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center; color: #666;">Deepfake ve AI Medya Tespit Aracı</h3>', unsafe_allow_html=True)
    
    # Sabit güven eşiği (threshold)
    confidence_threshold = 0.50

    # Ana içerik
    tab1, tab2, tab3 = st.tabs(["📸 Fotoğraf Analizi", "🎥 Video Analizi", "ℹ️ Hakkında"])
    
    with tab1:
        photo_analysis_tab(confidence_threshold)
    with tab2:
        video_analysis_tab(confidence_threshold)
    
    with tab3:
        about_tab()

def photo_analysis_tab(confidence_threshold):
    st.header("📸 Fotoğraf Analizi")
    
    # Fotoğraf analizi için model seçimi
    st.sidebar.title("⚙️ Fotoğraf Analizi Ayarları")
    model_type = st.sidebar.selectbox(
        "Model Türü",
        [
            "GANDetector (foto)",
            "Hibrit Model (foto & video)"
        ],
        key="photo_model"
    )
    # Model adını sadeleştir (analiz fonksiyonları için)
    model_type_clean = model_type.split(' (')[0]
    
    uploaded_file = st.file_uploader(
        "Analiz edilecek fotoğrafı yükleyin",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Desteklenen formatlar: PNG, JPG, JPEG, BMP, TIFF"
    )
    
    if uploaded_file is not None:
        # Fotoğrafı göster
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Yüklenen Fotoğraf")
            image = Image.open(uploaded_file)
            st.image(image, caption="Analiz edilecek fotoğraf", use_container_width=True)
        
        with col2:
            st.subheader("🔍 Analiz Sonuçları")
            
            # Progress bar
            with st.spinner("Fotoğraf analiz ediliyor..."):
                # Simüle edilmiş analiz (gerçek model entegrasyonu için placeholder)
                result = analyze_photo(image, model_type_clean)
                
                # Sonuçları göster
                display_photo_results(result, confidence_threshold)

def video_analysis_tab(confidence_threshold):
    st.header("🎥 Video Analizi")
    
    # Video analizi için model seçimi
    st.sidebar.title("⚙️ Video Analizi Ayarları")
    model_type = st.sidebar.selectbox(
        "Model Türü",
        [
            "XceptionNet (video)",
            "FaceForensics++ (video)",
            "Hibrit Model (foto & video)"
        ],
        key="video_model"
    )
    # Model adını sadeleştir (analiz fonksiyonları için)
    model_type_clean = model_type.split(' (')[0]
    
    uploaded_video = st.file_uploader(
        "Analiz edilecek videoyu yükleyin",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Desteklenen formatlar: MP4, AVI, MOV, MKV"
    )
    
    if uploaded_video is not None:
        # Video bilgilerini göster
        st.subheader("📹 Video Bilgileri")
        
        # Geçici dosya oluştur
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_video.getvalue())
            video_path = tmp_file.name
        
        # Video analizi
        with st.spinner("Video analiz ediliyor (bu işlem biraz zaman alabilir)..."):
            results = analyze_video(video_path, model_type_clean)
            
            # Sonuçları göster
            display_video_results(results, confidence_threshold)
        
        # Geçici dosyayı sil
        os.unlink(video_path)

def about_tab():
    st.header("ℹ️ FakeBuster Hakkında")
    
    st.markdown("""
    ### 🎯 Proje Amacı
    FakeBuster, deepfake ve AI ile oluşturulmuş medya içeriklerini tespit etmek için geliştirilmiş 
    açık kaynaklı bir araçtır.
    
    ### 🔬 Kullanılan Teknolojiler
    - **OpenCV**: Video ve görüntü işleme
    - **TensorFlow/PyTorch**: Derin öğrenme modelleri
    - **Streamlit**: Web arayüzü
    - **MediaPipe**: Yüz tespiti ve analizi
    
    ### 📊 Desteklenen Modeller
    1. **XceptionNet**: Deepfake video tespiti için optimize edilmiş
    2. **FaceForensics++**: Yüz manipülasyonu tespiti
    3. **GANDetector**: AI üretimi görsel tespiti
    4. **Hibrit Model**: Çoklu model kombinasyonu
    
    ### 🚀 Gelecek Özellikler
    - Gerçek zamanlı analiz
    - Daha fazla model desteği
    - API entegrasyonu
    - Batch işleme
    """)

def analyze_photo(image, model_type):
    """Fotoğraf analizi"""
    try:
        # PIL Image'i numpy array'e çevir
        if hasattr(image, 'convert'):
            image = np.array(image)
        
        # BGR'den RGB'ye çevir (gerekirse)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # DeepfakeDetector'ı başlat
        detector = DeepfakeDetector(model_type)
        
        # Görüntüyü analiz et
        result = detector.analyze_image(image)
        
        # Sonuç formatını uyumlu hale getir
        return {
            'is_fake': result['is_fake'],
            'confidence': result['confidence'],
            'model_type': model_type,
            'analysis_time': result['analysis_time'],
            'face_detected': result['face_detected'],
            'manipulation_score': result.get('manipulation_score', result['confidence']),
            'faces_analyzed': result.get('faces_analyzed', 0),
            'fake_faces': result.get('fake_faces', 0)
        }
        
    except Exception as e:
        st.error(f"Fotoğraf analiz hatası: {e}")
        # Fallback sonuç
        return {
            'is_fake': False,
            'confidence': 0.5,
            'model_type': model_type,
            'analysis_time': 0.0,
            'face_detected': False,
            'manipulation_score': 0.5,
            'error': str(e)
        }

def analyze_video(video_path, model_type):
    """Video analizi"""
    try:
        # DeepfakeDetector'ı başlat
        detector = DeepfakeDetector(model_type)
        
        # Video analizi
        result = detector.analyze_video(video_path, max_frames=30)
        
        # Sonuç formatını uyumlu hale getir
        return {
            'frame_results': result['frame_results'],
            'total_frames': result['total_frames_analyzed'],
            'fake_percentage': result['fake_percentage'],
            'overall_confidence': result['overall_confidence'],
            'is_fake': result['is_fake'],
            'duration': result['duration'],
            'fps': result['fps'],
            'model_type': model_type,
            'total_video_frames': result['total_video_frames']
        }
        
    except Exception as e:
        st.error(f"Video analiz hatası: {e}")
        # Fallback sonuç
        return {
            'frame_results': [],
            'total_frames': 0,
            'fake_percentage': 0,
            'overall_confidence': 0.5,
            'is_fake': False,
            'duration': 0,
            'fps': 0,
            'model_type': model_type,
            'error': str(e)
        }

def display_photo_results(result, threshold):
    """Fotoğraf analiz sonuçlarını göster"""
    
    is_fake = result['confidence'] > threshold
    if is_fake:
        st.markdown(f"""
        <div class="result-box fake-result">
            <h3>🚨 Bu görsel büyük olasılıkla <b>SAHTE</b>!</h3>
            <p>Yapay zeka analizine göre bu fotoğrafın üzerinde oynama veya sahtecilik yapılmış olabilir.</p>
            <p><b>Model:</b> {result['model_type']}</p>
            <p><b>İşlem Süresi:</b> {result['analysis_time']:.2f} sn</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-box real-result">
            <h3>✅ Bu görsel <b>GERÇEK</b> görünüyor.</h3>
            <p>Yapay zeka analizine göre bu fotoğrafta sahtecilik tespit edilmedi.</p>
            <p><b>Model:</b> {result['model_type']}</p>
            <p><b>İşlem Süresi:</b> {result['analysis_time']:.2f} sn</p>
        </div>
        """, unsafe_allow_html=True)
    


def display_video_results(results, threshold):
    """Video analiz sonuçlarını göster"""
    
    is_fake = results['overall_confidence'] > threshold
    if is_fake:
        st.markdown(f"""
        <div class="result-box fake-result">
            <h3>🚨 Bu video büyük olasılıkla <b>SAHTE</b>!</h3>
            <p>Yapay zeka analizine göre bu videoda sahtecilik veya oynama tespit edildi.</p>
            <p><b>Model:</b> {results['model_type']}</p>
            <p><b>İşlem Süresi:</b> {results['duration']:.2f} sn</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-box real-result">
            <h3>✅ Bu video <b>GERÇEK</b> görünüyor.</h3>
            <p>Yapay zeka analizine göre bu videoda sahtecilik tespit edilmedi.</p>
            <p><b>Model:</b> {results['model_type']}</p>
            <p><b>İşlem Süresi:</b> {results['duration']:.2f} sn</p>
        </div>
        """, unsafe_allow_html=True)
    


if __name__ == "__main__":
    main()
