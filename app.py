import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import time

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PotatoScan · Disease Detection",
    page_icon="🥔",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ─── Google Fonts ─────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ─── Tokens ────────────────────────────────────────────────────────────────── */
:root {
    --green-dark:   #0d2b1a;
    --green-mid:    #1a4731;
    --green-accent: #2e7d52;
    --green-light:  #4caf50;
    --gold:         #c9a84c;
    --cream:        #f5f0e8;
    --text-main:    #1c2b1e;
    --text-muted:   #5a7460;
    --red-blight:   #c0392b;
    --amber-blight: #d4810a;
    --card-bg:      #ffffff;
    --border:       #dde8dd;
}

/* ─── Global reset ──────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--text-main);
}

.stApp {
    background: var(--cream);
    background-image:
        radial-gradient(circle at 10% 20%, rgba(46,125,82,0.06) 0%, transparent 50%),
        radial-gradient(circle at 90% 80%, rgba(201,168,76,0.05) 0%, transparent 50%);
}

/* ─── Sidebar ───────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--green-dark) !important;
    border-right: 1px solid var(--green-mid);
}

[data-testid="stSidebar"] * {
    color: var(--cream) !important;
}

[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--gold) !important;
    font-family: 'DM Serif Display', serif;
}

[data-testid="stSidebar"] hr {
    border-color: var(--green-mid) !important;
}

/* ─── Hero header ───────────────────────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, var(--green-dark) 0%, var(--green-mid) 60%, var(--green-accent) 100%);
    border-radius: 20px;
    padding: 3rem 3.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(13,43,26,0.18);
}

.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(201,168,76,0.15) 0%, transparent 70%);
    border-radius: 50%;
}

.hero::after {
    content: '🥔';
    position: absolute;
    font-size: 9rem;
    right: 3rem; top: 50%;
    transform: translateY(-50%);
    opacity: 0.12;
    filter: grayscale(1);
}

.hero-badge {
    display: inline-block;
    background: rgba(201,168,76,0.2);
    border: 1px solid rgba(201,168,76,0.4);
    color: var(--gold) !important;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.3rem 0.9rem;
    border-radius: 30px;
    margin-bottom: 1rem;
}

.hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    color: #ffffff !important;
    line-height: 1.15;
    margin: 0.4rem 0;
}

.hero p {
    font-size: 1rem;
    color: rgba(245,240,232,0.75) !important;
    margin-top: 0.6rem;
    max-width: 500px;
    line-height: 1.6;
}

/* ─── Metric cards ──────────────────────────────────────────────────────────── */
.metric-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.2rem;
    margin-bottom: 2rem;
}

.metric-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 2px 12px rgba(13,43,26,0.06);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 24px rgba(13,43,26,0.12);
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--accent-color, var(--green-light));
    border-radius: 16px 16px 0 0;
}

.metric-icon {
    font-size: 2rem;
    margin-bottom: 0.6rem;
}

.metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.3rem;
}

.metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: var(--text-main);
    line-height: 1;
}

.metric-sub {
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-top: 0.3rem;
}

/* ─── Upload zone ───────────────────────────────────────────────────────────── */
.upload-section {
    background: var(--card-bg);
    border: 2px dashed var(--border);
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    margin-bottom: 2rem;
    transition: border-color 0.2s ease, background 0.2s ease;
}

.upload-section:hover {
    border-color: var(--green-accent);
    background: rgba(46,125,82,0.02);
}

.upload-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    color: var(--text-main);
    margin-bottom: 0.5rem;
}

.upload-sub {
    font-size: 0.9rem;
    color: var(--text-muted);
}

/* ─── Result card ───────────────────────────────────────────────────────────── */
.result-card {
    background: var(--green-light);
    border-radius: 20px;
    padding: 2rem;
    margin-top: 1.5rem;
    border: 1px solid var(--border);
    box-shadow: 0 4px 20px rgba(13,43,26,0.08);
}

.result-header {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.8rem;
}

.result-disease {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    margin-bottom: 0.4rem;
}

.disease-early  { color: var(--amber-blight); }
.disease-late   { color: var(--red-blight); }
.disease-healthy { color: var(--green-accent); }

/* ─── Confidence bar ────────────────────────────────────────────────────────── */
.conf-bar-wrap {
    margin-top: 1.2rem;
}

.conf-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    color: var(--text-muted);
    margin-bottom: 0.4rem;
}

.conf-bar-bg {
    height: 8px;
    background: #eef2ee;
    border-radius: 8px;
    overflow: hidden;
}

.conf-bar-fill {
    height: 100%;
    border-radius: 8px;
    background: linear-gradient(90deg, var(--green-accent), var(--green-light));
    transition: width 0.8s cubic-bezier(.4,0,.2,1);
}

/* ─── Pill badge ────────────────────────────────────────────────────────────── */
.status-pill {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 30px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-top: 0.6rem;
}

.pill-healthy { background: #e8f5e9; color: #2e7d32; border: 1px solid #a5d6a7; }
.pill-early   { background: #fff3e0; color: #e65100; border: 1px solid #ffcc80; }
.pill-late    { background: #fce4ec; color: #b71c1c; border: 1px solid #ef9a9a; }

/* ─── Info panel ────────────────────────────────────────────────────────────── */
.info-panel {
    background: var(--card-bg);
    border-left: 4px solid var(--green-accent);
    border-radius: 0 12px 12px 0;
    padding: 1.2rem 1.4rem;
    margin-top: 1.2rem;
    font-size: 0.9rem;
    line-height: 1.7;
    color: var(--text-muted);
}

.info-panel strong {
    color: var(--text-main);
}

/* ─── Section title ─────────────────────────────────────────────────────────── */
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: var(--text-main);
    margin-bottom: 0.3rem;
}

.section-sub {
    font-size: 0.85rem;
    color: var(--text-muted);
    margin-bottom: 1.2rem;
}

/* ─── Streamlit overrides ───────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    border: none !important;
    background: transparent !important;
}

[data-testid="stFileUploader"] > div {
    background: transparent !important;
}

.stButton button {
    background: var(--green-accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.8rem !important;
    transition: background 0.2s ease, transform 0.15s ease !important;
}

.stButton button:hover {
    background: var(--green-dark) !important;
    transform: translateY(-1px) !important;
}

/* hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load model ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_disease_model():
    return load_model("potato_disease_model.h5")

model = load_disease_model()
class_names = ['Early Blight', 'Late Blight', 'Healthy']

DISEASE_INFO = {
    'Early Blight': {
        'icon': '🟡',
        'css': 'disease-early',
        'pill': 'pill-early',
        'severity': 'Moderate',
        'desc': 'Caused by <strong>Alternaria solani</strong>. Appears as dark brown spots with concentric rings. Usually affects older leaves first.',
        'action': 'Apply copper-based fungicide. Remove and destroy infected leaves. Improve air circulation.',
    },
    'Late Blight': {
        'icon': '🔴',
        'css': 'disease-late',
        'pill': 'pill-late',
        'severity': 'Severe',
        'desc': 'Caused by <strong>Phytophthora infestans</strong>. Highly destructive — responsible for the Irish Famine. Water-soaked lesions on leaves.',
        'action': 'Apply fungicides immediately. Destroy all infected plants. Avoid overhead irrigation.',
    },
    'Healthy': {
        'icon': '🟢',
        'css': 'disease-healthy',
        'pill': 'pill-healthy',
        'severity': 'None',
        'desc': 'The leaf appears <strong>healthy</strong> with no visible signs of disease. Keep monitoring regularly for early detection.',
        'action': 'Continue regular monitoring and preventive care. Maintain soil health and proper irrigation.',
    },
}

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🥔 PotatoScan")
    st.markdown("---")
    st.markdown("#### How it works")
    st.markdown("""
1. **Upload** a potato leaf photo  
2. The AI model **analyzes** the image  
3. View your **diagnosis** and recommendations
    """)
    st.markdown("---")
    st.markdown("#### Detectable Diseases")
    st.markdown("🟡 **Early Blight** — *Alternaria solani*")
    st.markdown("🔴 **Late Blight** — *Phytophthora infestans*")
    st.markdown("🟢 **Healthy** — No disease detected")
    st.markdown("---")
    st.markdown("#### Model Info")
    st.markdown("""
- **Input size:** 224 × 224 px  
- **Classes:** 3  
- **Framework:** TensorFlow / Keras
    """)
    st.markdown("---")
    st.caption("PotatoScan v1.0 · Agricultural AI")

# ── Hero ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">AI-Powered · Plant Pathology</div>
    <h1>Potato Leaf<br>Disease Detection</h1>
    <p>Upload a leaf image and our deep learning model will identify Early Blight, Late Blight, or confirm a Healthy plant in seconds.</p>
</div>
""", unsafe_allow_html=True)

# ── Metric cards ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="metric-row">
  <div class="metric-card" style="--accent-color:#4caf50">
    <div class="metric-icon">🌿</div>
    <div class="metric-label">Detectable Classes</div>
    <div class="metric-value">3</div>
    <div class="metric-sub">Blight variants + Healthy</div>
  </div>
  <div class="metric-card" style="--accent-color:#c9a84c">
    <div class="metric-icon">⚡</div>
    <div class="metric-label">Inference Speed</div>
    <div class="metric-value">&lt;1s</div>
    <div class="metric-sub">Per image prediction</div>
  </div>
  <div class="metric-card" style="--accent-color:#2e7d52">
    <div class="metric-icon">📐</div>
    <div class="metric-label">Input Resolution</div>
    <div class="metric-value">224²</div>
    <div class="metric-sub">Pixels — RGB channels</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Main columns ──────────────────────────────────────────────────────────────
col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown('<p class="section-title">Upload Leaf Image</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Supported formats: JPG, PNG, JPEG</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "",
        type=["jpg", "png", "jpeg"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        img_display = Image.open(uploaded_file)
        st.image(img_display, caption="Uploaded leaf image")
    else:
        st.markdown("""
        <div class="upload-section">
            <div style="font-size:3rem;margin-bottom:0.8rem">🌱</div>
            <div class="upload-title">Drop your image here</div>
            <div class="upload-sub">or use the uploader above · JPG, PNG, JPEG</div>
        </div>
        """, unsafe_allow_html=True)

with col_result:
    st.markdown('<p class="section-title">Diagnosis Result</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">AI prediction will appear here after upload</p>', unsafe_allow_html=True)

    if uploaded_file is not None:
        # Pre-process
        img = Image.open(uploaded_file).resize((224, 224))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        with st.spinner("Analyzing leaf..."):
            time.sleep(0.4)   # brief pause for UX feel
            prediction = model.predict(img_array)

        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))
        info = DISEASE_INFO[predicted_class]

        # All-class confidence breakdown
        scores = {class_names[i]: float(prediction[0][i]) for i in range(len(class_names))}

        conf_pct = int(confidence * 100)

        st.markdown(f"""
        <div class="result-card">
            <div class="result-header">Predicted Condition</div>
            <div class="result-disease {info['css']}">{info['icon']} {predicted_class}</div>
            <span class="status-pill {info['pill']}">Severity: {info['severity']}</span>

            <div class="conf-bar-wrap">
                <div class="conf-label">
                    <span>Confidence</span>
                    <span><strong>{conf_pct}%</strong></span>
                </div>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill" style="width:{conf_pct}%"></div>
                </div>
            </div>

            <div class="info-panel" style="margin-top:1.4rem">
                <strong>About:</strong><br>{info['desc']}
            </div>
            <div class="info-panel" style="border-left-color:#c9a84c">
                <strong>Recommended Action:</strong><br>{info['action']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Score breakdown
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-title" style="font-size:1.1rem">All Class Scores</p>', unsafe_allow_html=True)
        for cls, score in scores.items():
            pct = int(score * 100)
            color = {"Early Blight": "#d4810a", "Late Blight": "#c0392b", "Healthy": "#2e7d52"}[cls]
            st.markdown(f"""
            <div style="margin-bottom:0.7rem">
                <div style="display:flex;justify-content:space-between;font-size:0.83rem;margin-bottom:0.3rem">
                    <span style="font-weight:500">{cls}</span>
                    <span style="color:{color};font-weight:600">{pct}%</span>
                </div>
                <div style="height:6px;background:#eef2ee;border-radius:6px;overflow:hidden">
                    <div style="height:100%;width:{pct}%;background:{color};border-radius:6px;transition:width 0.8s ease"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="result-card" style="text-align:center;padding:3rem 2rem;border:2px dashed #dde8dd">
            <div style="font-size:3rem;margin-bottom:1rem;opacity:0.4">🔬</div>
            <div style="font-family:'DM Serif Display',serif;font-size:1.3rem;color:#5a7460">
                Awaiting image upload
            </div>
            <div style="font-size:0.85rem;color:#5a7460;margin-top:0.5rem">
                Upload a potato leaf photo to see the AI diagnosis
            </div>
        </div>
        """, unsafe_allow_html=True)