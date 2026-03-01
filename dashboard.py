"""
Sign Language MNIST – Streamlit Dashboard
Upload a hand-sign PNG/JPG and see the predicted ASL letter + confidence.

Model priority (first match wins):
  1. multilayer_model/multilayer_model.h5  – Keras MLP
  2. baseline_model/baseline_model.pkl     – Random Forest (from notebook)
  3. Auto-train a lightweight Random Forest on the fly and cache it.
"""

import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Sign-Language MNIST labels: 0-24, skipping 9 (J requires motion)
# Maps integer label → uppercase letter
LABEL_TO_LETTER: dict[int, str] = {
    i: chr(ord("A") + i) for i in range(25) if i != 9
}

TRAIN_CSV = os.path.join(
    os.path.dirname(__file__),
    "data", "sign-language-mnist",
    "sign_mnist_train", "sign_mnist_train.csv",
)
RF_MODEL_PATH  = os.path.join(os.path.dirname(__file__), "baseline_model",   "baseline_model.pkl")
RF_SCALER_PATH = os.path.join(os.path.dirname(__file__), "baseline_model",   "scaler.pkl")
KERAS_MODEL_PATH = os.path.join(os.path.dirname(__file__), "multilayer_model", "multilayer_model.h5")

IMAGE_SIZE = 28  # Sign-Language MNIST pixel dimensions

# ---------------------------------------------------------------------------
# Model loading / training (cached so it runs once per session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="⏳ Loading model…")
def load_keras_model():
    """Load a saved Keras model."""
    import tensorflow as tf  # lazy import – avoid cost when not needed
    return tf.keras.models.load_model(KERAS_MODEL_PATH)


@st.cache_resource(show_spinner="⏳ Loading / training Random Forest…")
def load_rf_model():
    """
    Load a saved RF + scaler if they exist, otherwise train a quick one
    on the training CSV and persist it for future runs.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    # --- load saved model ---
    if os.path.exists(RF_MODEL_PATH) and os.path.exists(RF_SCALER_PATH):
        with open(RF_MODEL_PATH,  "rb") as f:
            model = pickle.load(f)
        with open(RF_SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        return model, scaler

    # --- auto-train ---
    if not os.path.exists(TRAIN_CSV):
        st.info("📥 Training data not found locally — downloading from Kaggle …")
        try:
            import kagglehub, shutil
            path = kagglehub.dataset_download("datamunge/sign-language-mnist")
            train_dest = os.path.join(os.path.dirname(__file__),
                                      "data", "sign-language-mnist", "sign_mnist_train")
            test_dest  = os.path.join(os.path.dirname(__file__),
                                      "data", "sign-language-mnist", "sign_mnist_test")
            os.makedirs(train_dest, exist_ok=True)
            os.makedirs(test_dest,  exist_ok=True)
            for f in os.listdir(path):
                src = os.path.join(path, f)
                if "train" in f and f.endswith(".csv"):
                    shutil.copy(src, os.path.join(train_dest, f))
                elif "test" in f and f.endswith(".csv"):
                    shutil.copy(src, os.path.join(test_dest, f))
        except Exception as e:
            st.error(
                f"Could not auto-download the dataset: {e}\n\n"
                "Please run the data-download cell in `baseline_model/model.ipynb` first, "
                "or install kagglehub (`pip install kagglehub`) and set up your Kaggle API key."
            )
            st.stop()

    df = pd.read_csv(TRAIN_CSV)
    X  = df.drop("label", axis=1).values.astype(np.float32)
    y  = df["label"].values

    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=30,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled, y)

    # Persist so next run is instant
    os.makedirs(os.path.dirname(RF_MODEL_PATH), exist_ok=True)
    with open(RF_MODEL_PATH,  "wb") as f:
        pickle.dump(model,  f)
    with open(RF_SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    return model, scaler


# ---------------------------------------------------------------------------
# Image pre-processing
# ---------------------------------------------------------------------------

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Convert an uploaded image to a (1, 784) float32 array."""
    img = image.convert("L")                        # grayscale
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))       # 28 × 28
    arr = np.array(img, dtype=np.float32).flatten()  # 784,
    return arr.reshape(1, -1)


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def predict_rf(image: Image.Image, model, scaler):
    arr = preprocess_image(image)
    arr_scaled = scaler.transform(arr)
    proba   = model.predict_proba(arr_scaled)[0]
    classes = model.classes_
    return classes, proba


def predict_keras(image: Image.Image, model):
    arr = preprocess_image(image) / 255.0            # normalise to [0, 1]
    proba   = model.predict(arr, verbose=0)[0]
    classes = np.arange(len(proba))
    return classes, proba


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ASL Sign Classifier",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #6C63FF 0%, #48CAE4 50%, #06D6A0 100%);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    color: white;
    text-align: center;
    box-shadow: 0 8px 32px rgba(108,99,255,0.25);
}
.hero h1 { font-size: 3rem; font-weight: 900; margin: 0 0 0.5rem 0; letter-spacing: -1px; }
.hero p  { font-size: 1.15rem; opacity: 0.92; margin: 0; }

/* ── Upload zone ── */
.upload-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 2px dashed #6C63FF;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
}

/* ── Result letter card ── */
.letter-card {
    background: linear-gradient(135deg, #6C63FF, #48CAE4);
    border-radius: 20px;
    padding: 1.5rem;
    text-align: center;
    color: white;
    box-shadow: 0 8px 24px rgba(108,99,255,0.35);
    margin-bottom: 1rem;
}
.letter-card .big-letter {
    font-size: 7rem;
    font-weight: 900;
    line-height: 1;
    text-shadow: 0 4px 14px rgba(0,0,0,0.25);
}
.letter-card .label  { font-size: 0.85rem; opacity: 0.85; text-transform: uppercase; letter-spacing: 2px; }

/* ── Confidence badge ── */
.conf-badge {
    display: inline-block;
    border-radius: 50px;
    padding: 0.5rem 1.5rem;
    font-size: 1.6rem;
    font-weight: 700;
    color: white;
    text-align: center;
    width: 100%;
    margin-bottom: 1rem;
    box-shadow: 0 4px 14px rgba(0,0,0,0.2);
}
.conf-high   { background: linear-gradient(90deg, #06D6A0, #0CB87C); }
.conf-medium { background: linear-gradient(90deg, #FFB703, #FB8500); }
.conf-low    { background: linear-gradient(90deg, #EF476F, #D62246); }

/* ── Top-5 row ── */
.top5-row {
    display: flex;
    gap: 0.6rem;
    margin-top: 0.5rem;
    flex-wrap: wrap;
}
.top5-chip {
    flex: 1 1 0;
    border-radius: 12px;
    padding: 0.6rem 0.3rem;
    text-align: center;
    font-weight: 700;
    font-size: 1rem;
    color: white;
    min-width: 48px;
}
.chip-1 { background: linear-gradient(135deg,#6C63FF,#48CAE4); }
.chip-2 { background: linear-gradient(135deg,#48CAE4,#06D6A0); box-shadow:none; opacity:.9; }
.chip-3 { background: linear-gradient(135deg,#FFB703,#FB8500); opacity:.85; }
.chip-4 { background: linear-gradient(135deg,#EF476F,#D62246); opacity:.75; }
.chip-5 { background: #444; opacity:.65; }

/* ── Section headings ── */
.section-head {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #6C63FF;
    margin: 1.2rem 0 0.4rem 0;
}

/* ── Image frame ── */
.img-frame {
    border-radius: 16px;
    overflow: hidden;
    border: 3px solid #6C63FF44;
    box-shadow: 0 8px 24px rgba(108,99,255,0.2);
}

/* ── Sidebar styling ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%) !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-bottom: 130px !important;
}
[data-testid="stSidebar"] * { color: #e0e0ff !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 0.95rem !important; }

/* ── About – fixed to bottom of sidebar ── */
.about-bottom {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 21rem;
    padding: 1rem 1.5rem 1.4rem 1.5rem;
    background: linear-gradient(180deg, transparent 0%, #16213e 28%, #16213e 100%);
    border-top: 1px solid #6C63FF55;
    z-index: 100;
    box-sizing: border-box;
}
.about-bottom h4 {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #6C63FF !important;
    margin: 0 0 0.45rem 0;
}
.about-bottom p   { font-size: 0.82rem; color: #b0b0cc !important; margin: 0.15rem 0; }
.about-bottom a   { color: #48CAE4 !important; text-decoration: none; }
.about-bottom a:hover { text-decoration: underline; }
.about-bottom .tag {
    display: inline-block;
    margin-top: 0.5rem;
    background: linear-gradient(90deg,#6C63FF33,#48CAE433);
    border: 1px solid #6C63FF55;
    border-radius: 20px;
    padding: 0.2rem 0.75rem;
    font-size: 0.72rem;
    color: #a0a0cc !important;
    font-weight: 600;
    letter-spacing: 0.5px;
}

/* ── Streamlit metric tweaks ── */
[data-testid="stMetricValue"] { font-size: 2.4rem !important; font-weight: 800 !important; }

/* ── Progress bars for top-5 ── */
.bar-row { margin: 0.3rem 0; }
.bar-label { display: inline-block; width: 2rem; font-weight: 700; color: #6C63FF; }
.bar-bg { display: inline-block; background: #22223b; border-radius: 6px; height: 14px; width: 60%; vertical-align: middle; margin: 0 0.5rem; }
.bar-fill { height: 14px; border-radius: 6px; background: linear-gradient(90deg,#6C63FF,#48CAE4); }
.bar-pct { font-size: 0.85rem; color: #aaa; }
</style>
""", unsafe_allow_html=True)


# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>ASL Sign Language Detector</h1>
  <p>Upload any hand-sign photo — get the predicted <strong>American Sign Language</strong> letter and confidence score instantly.</p>
</div>
""", unsafe_allow_html=True)



with st.sidebar:

    model_options = []
    if os.path.exists(KERAS_MODEL_PATH):
        model_options.append("🧠  Multilayer MLP (Keras)")
    model_options.append("🌲  Random Forest (baseline)")

    chosen_model = st.radio("**Select model**", model_options, index=0)

    # Load model
    if chosen_model.startswith("🧠"):
        keras_model = load_keras_model()
        backend = "keras"
        st.success("Keras MLP loaded ✓")
    else:
        rf_model, rf_scaler = load_rf_model()
        backend = "rf"
        st.success("Random Forest loaded ✓")

    st.markdown("""
    <div class="about-bottom">
        <h4>📖 About</h4>
        <p>Dataset: <a href="https://www.kaggle.com/datamunge/sign-language-mnist" target="_blank">Sign Language MNIST</a></p>
        <p>24 ASL letters (J &amp; Z excluded — require motion)</p>
        <div class="tag">CSCI 4083 · Mid-Course Project</div>
    </div>
    """, unsafe_allow_html=True)


# ── Main layout ───────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1.1, 0.9], gap="large")

with left_col:
    st.markdown('<p class="section-head">📤 Upload Image</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop a PNG / JPG hand-sign image",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.markdown('<div class="img-frame">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.caption(f"📐 Original: {image.width} × {image.height} px  →  resized to 28 × 28 for inference")
    else:
        # Show example ASL chart
        example_dir = os.path.join(os.path.dirname(__file__), "data", "sign-language-mnist")
        shown = False
        for img_name in ["amer_sign2.png", "amer_sign3.png", "american_sign_language.PNG"]:
            img_path = os.path.join(example_dir, img_name)
            if os.path.exists(img_path):
                st.markdown('<p class="section-head">🔤 ASL Alphabet Reference</p>', unsafe_allow_html=True)
                st.image(img_path, use_container_width=True)
                shown = True
                break
        if not shown:
            st.markdown("""
            <div style="border:2px dashed #6C63FF44; border-radius:16px; padding:3rem; text-align:center; color:#888;">
                <div style="font-size:3rem">🤟</div>
                <div style="margin-top:0.5rem">Drag & drop an image to get started</div>
            </div>
            """, unsafe_allow_html=True)


with right_col:
    if uploaded_file is not None:
        # ── Run inference ──
        if backend == "keras":
            classes, proba = predict_keras(image, keras_model)
        else:
            classes, proba = predict_rf(image, rf_model, rf_scaler)

        top_idx     = int(np.argmax(proba))
        pred_label  = int(classes[top_idx])
        confidence  = float(proba[top_idx])
        pred_letter = LABEL_TO_LETTER.get(pred_label, "?")

        # ── Letter card ──
        st.markdown('<p class="section-head">🎯 Prediction</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="letter-card">
            <div class="label">Predicted Letter</div>
            <div class="big-letter">{pred_letter}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Confidence badge ──
        if confidence >= 0.6:
            conf_cls, conf_icon = "conf-high",   "✅"
        elif confidence >= 0.35:
            conf_cls, conf_icon = "conf-medium", "⚠️"
        else:
            conf_cls, conf_icon = "conf-low",    "❓"

        st.markdown(f"""
        <div class="conf-badge {conf_cls}">
            {conf_icon}  Confidence: {confidence:.1%}
        </div>
        """, unsafe_allow_html=True)

        # ── Top-5 chips ──
        st.markdown('<p class="section-head">🏆 Top-5 Candidates</p>', unsafe_allow_html=True)

        top5_idx     = np.argsort(proba)[::-1][:5]
        top5_letters = [LABEL_TO_LETTER.get(int(classes[i]), "?") for i in top5_idx]
        top5_proba   = [float(proba[i]) for i in top5_idx]

        chip_classes = ["chip-1", "chip-2", "chip-3", "chip-4", "chip-5"]
        chips_html = '<div class="top5-row">'
        for rank, (letter, prob, chip_cls) in enumerate(zip(top5_letters, top5_proba, chip_classes)):
            chips_html += f"""
            <div class="top5-chip {chip_cls}">
                <div style="font-size:1.6rem">{letter}</div>
                <div style="font-size:0.72rem;opacity:.85">{prob:.0%}</div>
            </div>"""
        chips_html += "</div>"
        st.markdown(chips_html, unsafe_allow_html=True)

        # ── Progress bars ──
        st.markdown('<p class="section-head">📊 Confidence Breakdown</p>', unsafe_allow_html=True)
        bars_html = ""
        max_prob = max(top5_proba) if top5_proba else 1
        for letter, prob in zip(top5_letters, top5_proba):
            fill_pct = int((prob / max_prob) * 100)
            bars_html += f"""
            <div class="bar-row">
                <span class="bar-label">{letter}</span>
                <span class="bar-bg"><span class="bar-fill" style="width:{fill_pct}%"></span></span>
                <span class="bar-pct">{prob:.1%}</span>
            </div>"""
        st.markdown(bars_html, unsafe_allow_html=True)

    else:
        # No image yet – placeholder
        st.markdown("""
        <div style="
            background: linear-gradient(135deg,#1a1a2e,#16213e);
            border-radius: 20px;
            padding: 3.5rem 2rem;
            text-align: center;
            color: #6C63FF;
            border: 2px solid #6C63FF33;
            margin-top: 2.5rem;
        ">
            <div style="font-size:4rem">🖼️</div>
            <div style="font-size:1.2rem;font-weight:700;margin-top:1rem;color:#e0e0ff">
                Upload an image to see the prediction
            </div>
            <div style="font-size:0.95rem;color:#888;margin-top:0.5rem">
                Supports PNG, JPG, JPEG — any resolution
            </div>
        </div>
        """, unsafe_allow_html=True)
