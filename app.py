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
        
        # Yeni gelişmiş sonuç formatını döndür
        return {
            'is_fake': result['is_fake'],
            'confidence': result['confidence'],
            'adjusted_confidence': result.get('adjusted_confidence', result['confidence']),
            'uncertainty': result.get('uncertainty', 0.0),
            'reliability_score': result.get('reliability_score', 1.0),
            'scale_consistency': result.get('scale_consistency', 1.0),
            'decision_threshold': result.get('decision_threshold', 0.5),
            'result_category': result.get('result_category', 'Bilinmiyor'),
            'category_icon': result.get('category_icon', '❓'),
            'model_type': model_type,
            'analysis_time': result['analysis_time'],
            'face_detected': result['face_detected'],
            'face_count': result.get('face_count', 0),
            'quality_metrics': result.get('quality_metrics', {}),
            'analysis_methods': result.get('analysis_methods', {}),
            'technical_details': result.get('technical_details', {}),
            'manipulation_score': result.get('adjusted_confidence', result['confidence'])
        }
        
    except Exception as e:
        st.error(f"Fotoğraf analiz hatası: {e}")
        # Fallback sonuç
        return {
            'is_fake': False,
            'confidence': 0.3,
            'adjusted_confidence': 0.3,
            'uncertainty': 1.0,
            'reliability_score': 0.0,
            'scale_consistency': 0.0,
            'decision_threshold': 0.5,
            'result_category': 'Hata',
            'category_icon': '❌',
            'model_type': model_type,
            'analysis_time': 0.0,
            'face_detected': False,
            'face_count': 0,
            'quality_metrics': {'certainty': 0.0, 'consistency': 0.0, 'reliability': 0.0},
            'manipulation_score': 0.3,
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
    """Gelişmiş fotoğraf analiz sonuçlarını göster"""
    
    # Sonuç kutusu
    category_icon = result.get('category_icon', '❓')
    result_category = result.get('result_category', 'Bilinmiyor')
    reliability_score = result.get('reliability_score', 0.0)
    
    if result['is_fake']:
        st.markdown(f"""
        <div class="result-box fake-result">
            <h3>🚨 SAHTE TESPİT EDİLDİ {category_icon}</h3>
            <p><strong>Model:</strong> {result['model_type']}</p>
            <p><strong>Güvenilirlik:</strong> {result_category} ({reliability_score:.1%})</p>
            <p><strong>Güven Skoru:</strong> <span class="confidence-high">{result['confidence']:.2%}</span></p>
            <p><strong>Ayarlı Skor:</strong> <span class="confidence-high">{result.get('adjusted_confidence', result['confidence']):.2%}</span></p>
            <p><strong>Analiz Süresi:</strong> {result['analysis_time']:.2f} saniye</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-box real-result">
            <h3>✅ GERÇEK TESPİT EDİLDİ {category_icon}</h3>
            <p><strong>Model:</strong> {result['model_type']}</p>
            <p><strong>Güvenilirlik:</strong> {result_category} ({reliability_score:.1%})</p>
            <p><strong>Güven Skoru:</strong> <span class="confidence-high">{result['confidence']:.2%}</span></p>
            <p><strong>Ayarlı Skor:</strong> <span class="confidence-high">{result.get('adjusted_confidence', result['confidence']):.2%}</span></p>
            <p><strong>Analiz Süresi:</strong> {result['analysis_time']:.2f} saniye</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Gelişmiş detaylı bilgiler
    st.subheader("📊 Gelişmiş Analiz Metrikleri")
    
    # İlk satır - Ana metrikler
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Güven Skoru", f"{result['confidence']:.1%}")
        st.caption("Ensemble algoritmaların ortalama skoru")
    
    with col2:
        st.metric("Belirsizlik", f"{result.get('uncertainty', 0.0):.1%}")
        st.caption("Sonucun belirsizlik seviyesi")
    
    with col3:
        st.metric("Güvenilirlik", f"{result.get('reliability_score', 1.0):.1%}")
        st.caption("Analizin genel güvenilirliği")
    
    with col4:
        st.metric("Kesinlik", f"{result.get('quality_metrics', {}).get('certainty', 1.0):.1%}")
        st.caption("1 - Belirsizlik")
    
    # İkinci satır - Teknik metrikler
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric("Eşik Değeri", f"{result.get('decision_threshold', 0.5):.1%}")
        st.caption("Kullanılan karar eşiği")
    
    with col6:
        st.metric("Ölçek Tutarlılığı", f"{result.get('scale_consistency', 1.0):.1%}")
        st.caption("Çoklu ölçek analiz tutarlılığı")
    
    with col7:
        st.metric("Yüz Sayısı", f"{result.get('face_count', 0)}")
        st.caption("Tespit edilen yüz sayısı")
    
    with col8:
        quality_metrics = result.get('quality_metrics', {})
        consistency = quality_metrics.get('consistency', 1.0)
        st.metric("Tutarlılık", f"{consistency:.1%}")
        st.caption("Yöntemler arası tutarlılık")
    
    # Ensemble yöntemleri detayları
    st.subheader("🎯 Ensemble Analiz Detayları")
    
    analysis_methods = result.get('analysis_methods', {})
    ensemble_data = analysis_methods.get('ensemble', {})
    
    if 'methods' in ensemble_data:
        st.write("**Kullanılan Algoritmalar:**")
        
        # Ensemble yöntemlerini göster
        col_methods = st.columns(4)
        methods = ensemble_data['methods']
        
        method_names = {
            'feature_based': 'Özellik Tabanlı',
            'statistical': 'İstatistiksel',
            'anomaly': 'Anomali Tespiti',
            'entropy': 'Entropi Analizi'
        }
        
        for i, (method, confidence) in enumerate(methods.items()):
            with col_methods[i % 4]:
                st.metric(
                    method_names.get(method, method), 
                    f"{confidence:.1%}",
                    delta=f"{confidence - result['confidence']:.1%}" if confidence != result['confidence'] else None
                )
    
    # Çoklu ölçek analizi
    multi_scale_data = analysis_methods.get('multi_scale', {})
    if multi_scale_data:
        st.write("**Çoklu Ölçek Analizi:**")
        scale_cols = st.columns(len(multi_scale_data))
        
        for i, (scale_name, scale_data) in enumerate(multi_scale_data.items()):
            with scale_cols[i]:
                scale_confidence = scale_data.get('confidence', 0.0)
                st.metric(
                    f"{scale_name.replace('scale_', '').replace('x', '×')}",
                    f"{scale_confidence:.1%}"
                )
    
    # Teknik detaylar (genişletilebilir)
    with st.expander("🔧 Teknik Detaylar"):
        technical_details = result.get('technical_details', {})
        
        col_tech1, col_tech2 = st.columns(2)
        
        with col_tech1:
            st.write("**Analiz Bilgileri:**")
            st.write(f"- Çıkarılan özellik sayısı: {technical_details.get('features_extracted', 'N/A')}")
            st.write(f"- Belirsizlik ayarlaması: {'✅' if technical_details.get('adjustment_applied', False) else '❌'}")
            st.write(f"- Eşik yükseltme: {'✅' if technical_details.get('threshold_elevated', False) else '❌'}")
        
        with col_tech2:
            st.write("**Kullanılan Yöntemler:**")
            methods_used = technical_details.get('methods_used', [])
            for method in methods_used:
                method_names_tech = {
                    'ensemble': '🎯 Ensemble Analizi',
                    'multi_scale': '📏 Çoklu Ölçek',
                    'uncertainty_estimation': '📊 Belirsizlik Tahmini'
                }
                st.write(f"- {method_names_tech.get(method, method)}")
    
    # Güven skoru grafiği
    st.subheader("📈 Güven Skoru Analizi")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Çok metrikli grafik
    metrics = {
        'Güven Skoru': result['confidence'],
        'Ayarlı Skor': result.get('adjusted_confidence', result['confidence']),
        'Güvenilirlik': result.get('reliability_score', 1.0),
        'Kesinlik': result.get('quality_metrics', {}).get('certainty', 1.0)
    }
    
    bars = ax.bar(metrics.keys(), metrics.values(), alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Skor')
    ax.set_title('Gelişmiş Analiz Skorları')
    
    # Eşik çizgisi
    threshold_used = result.get('decision_threshold', threshold)
    ax.axhline(y=threshold_used, color='red', linestyle='--', alpha=0.7, label=f'Karar Eşiği ({threshold_used:.1%})')
    ax.legend()
    
    # Bar değerlerini göster
    for bar, value in zip(bars, metrics.values()):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.1%}', ha='center', va='bottom')
    
    st.pyplot(fig)

def display_video_results(results, threshold):
    """Video analiz sonuçlarını göster"""
    
    # Genel sonuç
    if results['is_fake']:
        st.markdown(f"""
        <div class="result-box fake-result">
            <h3>🚨 SAHTE VİDEO TESPİT EDİLDİ</h3>
            <p><strong>Sahte Frame Oranı:</strong> {results['fake_percentage']:.1f}%</p>
            <p><strong>Ortalama Güven Skoru:</strong> {results['overall_confidence']:.2%}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-box real-result">
            <h3>✅ GERÇEK VİDEO TESPİT EDİLDİ</h3>
            <p><strong>Sahte Frame Oranı:</strong> {results['fake_percentage']:.1f}%</p>
            <p><strong>Ortalama Güven Skoru:</strong> {results['overall_confidence']:.2%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Video bilgileri
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Video Süresi", f"{results['duration']:.2f} saniye")
    
    with col2:
        st.metric("FPS", f"{results['fps']:.1f}")
    
    with col3:
        st.metric("Analiz Edilen Frame", results['total_frames'])
    
    with col4:
        st.metric("Sahte Frame %", f"{results['fake_percentage']:.1f}%")
    st.caption("Ortalama güven skoru, analiz edilen tüm frame'lerin sahte olma olasılığının ortalamasıdır. Yüksek skor, videonun sahte olma ihtimalinin yüksek olduğunu gösterir. Bu skor, güven eşiği ile karşılaştırılır.")
    
    # Frame analiz grafiği
    st.subheader("📈 Frame-by-Frame Analiz")
    
    frame_numbers = [r['frame_number'] for r in results['frame_results']]
    confidences = [r['confidence'] for r in results['frame_results']]
    is_fake = [r['is_fake'] for r in results['frame_results']]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Güven skoru grafiği
    colors = ['green' if not fake else 'red' for fake in is_fake]
    ax1.scatter(frame_numbers, confidences, c=colors, alpha=0.7)
    ax1.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, label=f'Eşik ({threshold:.2%})')
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
    st.pyplot(fig)

if __name__ == "__main__":
    main()
