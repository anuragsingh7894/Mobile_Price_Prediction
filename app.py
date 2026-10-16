import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Mobile Price Prediction AI",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================
# 2. HERO IMAGE URL
# =========================================================
# Direct image URL from your Kommodo share link.
# Note: the provider says this link expires on November 15, 2026.
HERO_IMAGE_URL = "https://plain-apac-prod-public.komododecks.com/202610/16/k8sVsf4t5cCYVrjo9FMT/image.png"

# =========================================================
# 3. LOAD MODEL AND SCALER
# =========================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("mobile_price_prediction_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


try:
    Mobile_price_predictor, Mobile_price_predictor_scaler = load_artifacts()
except Exception as e:
    st.error(f"Unable to load model or scaler: {e}")
    st.stop()

# =========================================================
# 4. CUSTOM CSS
# =========================================================
st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(ellipse at 8% 10%, rgba(0,126,255,.16), transparent 34%),
            radial-gradient(ellipse at 90% 25%, rgba(153,55,255,.15), transparent 32%),
            linear-gradient(145deg, #030817 0%, #07152d 52%, #020611 100%);
        color: #f4f7ff;
    }
    header[data-testid="stHeader"] { background: transparent; }
    #MainMenu, footer { visibility: hidden; }
    .block-container { max-width: 1600px; padding-top: 1.5rem; padding-bottom: 2rem; }
    html, body, [class*="css"] { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    p, label, .stMarkdown { color: #e9efff; }

    /* Hero text and layout */
    .hero-copy { padding: 24px 8px 22px 4px; }
    .ai-badge {
        display: inline-block; border: 1px solid rgba(129,115,255,.55);
        border-radius: 30px; padding: 8px 15px; color: #d9ccff;
        background: rgba(94,73,255,.13); font-size: 13px; margin-bottom: 15px;
    }
    .hero-title {
        font-size: clamp(30px, 3.3vw, 54px); font-weight: 800;
        letter-spacing: -1.8px; line-height: 1.13; color: #fff; margin: 0 0 14px 0;
    }
    .gradient-text {
        background: linear-gradient(90deg, #b9a4ff, #8c73ff, #39d8ff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero-description { font-size: 16px; color: #bdcce5; line-height: 1.7; margin-bottom: 18px; }
    .hero-note {
        display: inline-block; padding: 12px 16px; border: 1px solid rgba(66,153,255,.34);
        border-radius: 12px; background: rgba(25,85,163,.15); color: #64c8ff; font-size: 13px;
    }

    /* Hero image: no decorative rounded-square wrapper */
    [data-testid="stImage"] img {
        object-fit: contain;
    }

    .section-heading { font-size: 22px; font-weight: 750; color: #fff; margin: 10px 0 5px 0; }
    .section-subtitle { color: #94a9cb; font-size: 13px; margin-bottom: 18px; }

    /* Specification cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(145deg, rgba(26,43,72,.83), rgba(10,20,41,.91));
        border: 1px solid rgba(129,165,220,.23); border-radius: 17px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,.035), 0 8px 24px rgba(0,0,0,.12);
        transition: border-color .2s ease, box-shadow .2s ease;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(77,170,255,.55); box-shadow: 0 0 20px rgba(41,124,255,.10);
    }
    div[data-testid="stWidgetLabel"] p { color: #e6edff !important; font-size: 13px !important; font-weight: 650 !important; }
    div[data-baseweb="input"] > div, div[data-baseweb="select"] > div,
    div[data-testid="stNumberInput"] button {
        background-color: rgba(37,52,76,.90) !important;
        border-color: rgba(125,158,207,.30) !important; color: #fff !important; border-radius: 10px !important;
    }
    div[data-baseweb="input"] input, div[data-baseweb="select"] div,
    div[data-baseweb="input"] input::placeholder { color: #fff !important; -webkit-text-fill-color: #fff; }
    div[data-baseweb="input"] > div:focus-within, div[data-baseweb="select"] > div:focus-within {
        border-color: #36c9ff !important; box-shadow: 0 0 0 1px #36c9ff !important;
    }
    div[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
        background: #c99aff !important; border: 2px solid #fff !important;
        box-shadow: 0 0 12px rgba(173,98,255,.8);
    }

    /* Predict button */
    div.stButton > button[kind="primary"] {
        width: 100%; min-height: 58px; border: 1px solid rgba(135,200,255,.8);
        border-radius: 50px; background: linear-gradient(100deg,#00b9ff 0%,#2463ff 35%,#822dff 72%,#da3dff 100%);
        color: #fff; font-size: 17px; font-weight: 800;
        box-shadow: 0 0 18px rgba(0,146,255,.30), 0 0 35px rgba(145,51,255,.20);
        transition: all .25s ease;
    }
    div.stButton > button[kind="primary"]:hover {
        border-color: #fff; box-shadow: 0 0 25px rgba(0,176,255,.45), 0 0 45px rgba(145,51,255,.32);
        transform: translateY(-2px);
    }

    .result-card {
        border: 1px solid rgba(83,211,255,.45); border-radius: 18px; padding: 24px; margin-top: 18px;
        background: radial-gradient(ellipse at top right, rgba(127,64,255,.18), transparent 55%),
                    linear-gradient(135deg, rgba(8,45,75,.88), rgba(25,16,58,.90));
        box-shadow: 0 0 25px rgba(48,156,255,.12);
    }
    .result-label { font-size: 13px; color: #9db7dd; margin-bottom: 7px; }
    .result-price { font-size: clamp(30px,4vw,44px); font-weight: 800; letter-spacing: -1px; color: #70e5ff; }
    .result-caption { color: #b8c9e7; font-size: 13px; margin-top: 8px; }

    @media (max-width: 700px) {
        .hero-copy { padding-top: 12px; }
        .hero-title { letter-spacing: -.8px; }
        .block-container { padding-left: 1rem; padding-right: 1rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 5. HERO SECTION: TEXT + YOUR IMAGE
# =========================================================
hero_text, hero_image = st.columns([1.1, 1], gap="large", vertical_alignment="center")

with hero_text:
    st.markdown(
        """
        <div class="hero-copy">
            <div class="ai-badge">✦ &nbsp; AI POWERED</div>
            <h1 class="hero-title">Mobile Price<br><span class="gradient-text">Prediction App</span></h1>
            <p class="hero-description">
                Discover your smartphone's estimated value using its
                hardware specifications and a machine learning model.
            </p>
            <div class="hero-note">ⓘ &nbsp; Enter all 12 specifications to estimate the price.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with hero_image:
    st.image(
        HERO_IMAGE_URL,
        width=400,  # Slightly smaller image; try 380 or 420 if preferred
        caption="Mobile hardware specifications",
    )

# =========================================================
# 6. SPECIFICATION INPUTS
# =========================================================
st.markdown(
    """
    <div class="section-heading">⚙️ Smartphone Specifications</div>
    <div class="section-subtitle">Configure your smartphone's hardware details below.</div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    with st.container(border=True):
        st.markdown("#### 📏 Physical & Battery")
        Thickness = st.number_input(
            "Thickness (mm)", min_value=0.1, max_value=50.0,
            value=8.0, step=0.1, format="%.2f",
            help="Smartphone thickness in millimeters."
        )
        Battery_power = st.number_input(
            "Battery power (mAh)", min_value=1, max_value=30000,
            value=4000, step=100
        )

    with st.container(border=True):
        st.markdown("#### 📸 Camera Specifications")
        Front_camera = st.number_input(
            "Front camera resolution (MP)", min_value=0, max_value=300,
            value=13, step=1
        )
        Rear_camera = st.number_input(
            "Rear camera resolution (MP)", min_value=0, max_value=300,
            value=13, step=1
        )

with col2:
    with st.container(border=True):
        st.markdown("#### 💾 Memory")
        Ram = st.number_input(
            "RAM (as used in training data)", min_value=0.0,
            value=6.0, step=1.0, format="%.2f"
        )
        Internal_memory = st.number_input(
            "Internal memory (GB)", min_value=0.0,
            value=64.0, step=16.0
        )

    with st.container(border=True):
        st.markdown("#### ⚡ Processor")
        Cpu_frequency = st.number_input(
            "CPU frequency (GHz)", min_value=0.0, max_value=10.0,
            value=2.0, step=0.1, format="%.2f"
        )
        Cpu_cores = st.number_input(
            "Number of CPU cores", min_value=0, max_value=128,
            value=8, step=1
        )

with col3:
    with st.container(border=True):
        st.markdown("#### 🖥️ Display")
        PPI = st.number_input(
            "Pixel density (PPI)", min_value=0, max_value=2000,
            value=400, step=10
        )
        Resolution = st.number_input(
            "Screen size (inches)", min_value=0.1, max_value=30.0,
            value=5.0, step=0.1, format="%.2f",
            help=(
                "Confirm that this field's meaning and units match the "
                "'resoloution' feature used to train your model."
            ),
        )

    with st.container(border=True):
        st.markdown("#### ⚖️ Weight & Sales")
        weight = st.number_input(
            "Smartphone weight (grams)", min_value=1.0, max_value=5000.0,
            value=150.6, step=1.0, format="%.2f"
        )
        Sales = st.number_input(
            "Number of sales (units)", min_value=0,
            value=100, step=10
        )

# =========================================================
# 7. PREDICTION
# =========================================================
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
predict_clicked = st.button("✦   Predict Price   ➜", type="primary", use_container_width=True)

if predict_clicked:
    # Keep exactly the feature names and order used during training.
    feature_names = [
        "Sale", "weight", "resoloution", "ppi", "cpu core", "cpu freq",
        "internal mem", "ram", "RearCam", "Front_Cam", "battery", "thickness"
    ]

    input_values = [[
        Sales, weight, Resolution, PPI, Cpu_cores, Cpu_frequency,
        Internal_memory, Ram, Rear_camera, Front_camera, Battery_power, Thickness
    ]]
    input_data = pd.DataFrame(input_values, columns=feature_names)

    try:
        scaled_input_data = Mobile_price_predictor_scaler.transform(input_data)
        prediction = Mobile_price_predictor.predict(scaled_input_data)
        predicted_price = float(np.asarray(prediction).ravel()[0])

        if not np.isfinite(predicted_price):
            st.error("The model returned an invalid prediction. Please verify the inputs and trained model.")
        else:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">✦ &nbsp; ESTIMATED SMARTPHONE PRICE</div>
                    <div class="result-price">{predicted_price:,.2f}</div>
                    <div class="result-caption">Price predicted by your trained machine learning model.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander("📊 View submitted specifications"):
                st.dataframe(input_data, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# =========================================================
# 8. FOOTER
# =========================================================
st.markdown(
    """
    <div style="text-align:center;padding:25px 0 5px;color:#7185a8;font-size:12px;">
        POWERED BY MACHINE LEARNING &nbsp; • &nbsp; MOBILE PRICE PREDICTION
    </div>
    """,
    unsafe_allow_html=True,
)
