import io
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

import engine
from disease_info import INFO, DISCLAIMER

HERE = Path(__file__).resolve().parent
st.set_page_config(page_title="Grape Leaf Disease Detector", page_icon="🍇", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def get_model(name):
    return engine.load_model(name)


def pretty(cls):
    return INFO[cls]["display"]


# ── sidebar ──────────────────────────────────────────────────
models_ok = engine.available_models()
if not models_ok:
    st.error("No model weights found in the 'weights' folder. See README.")
    st.stop()
default = engine.CFG.get("default_model", models_ok[0])
with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Model", models_ok,
                              index=models_ok.index(default) if default in models_ok else 0)
    threshold = st.slider("Low-confidence warning below", 0.50, 0.95, 0.80, 0.05)
    alpha = st.slider("Heat-map opacity", 0.1, 0.9, 0.45, 0.05)
    st.caption(f"Running on: {engine.DEVICE}")
    st.divider()
    st.caption(DISCLAIMER)

st.title("🍇 Grape Leaf Disease Detector")
st.write("Upload a photo of **a single grape leaf** (ideally on a plain background). "
         "The app predicts the condition, shows its confidence, highlights the image regions "
         "that influenced the decision (Grad-CAM) and gives basic disease information.")

# ── image source ─────────────────────────────────────────────
samples = sorted((HERE / "samples").glob("*.*")) if (HERE / "samples").exists() else []
options = ["Upload", "Camera"] + (["Sample images"] if samples else [])
source = st.radio("Image source", options, horizontal=True)

pil = None
if source == "Upload":
    f = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if f is not None:
        pil = Image.open(f)
elif source == "Camera":
    f = st.camera_input("Take a photo of the leaf")
    if f is not None:
        pil = Image.open(f)
else:
    choice = st.selectbox("Sample", [p.name for p in samples])
    pil = Image.open(HERE / "samples" / choice)

if pil is None:
    st.info("Waiting for an image...")
    st.stop()

if min(pil.size) < 150:
    st.warning("This image is very small; predictions may be unreliable.")

# ── inference ────────────────────────────────────────────────
model = get_model(model_name)
rgb = engine.preprocess_image(pil)
probs = engine.predict(model, rgb)
order = np.argsort(-probs)
top = int(order[0]); conf = float(probs[top]); cls = engine.CLASSES[top]

c1, c2, c3 = st.columns([1, 1, 1.1])
with c1:
    st.subheader("Your photo")
    st.image(rgb, caption="After the same preprocessing used in training", width="stretch")

explain_label = None
with c3:
    st.subheader("Result")
    st.metric("Prediction", pretty(cls), f"{conf*100:.1f}% confidence", delta_color="off")
    if conf < threshold:
        st.warning("Low confidence. Retake the photo: one leaf, in focus, good light, "
                   "plain background. The model only knows 4 conditions and cannot tell "
                   "when an image is not a grape leaf.")
    elif cls != "Healthy Leaves":
        st.error(f"Likely **{pretty(cls)}**. See the details below.")
    else:
        st.success("No disease pattern detected on this leaf.")
    st.write("**All class probabilities**")
    for i in order:
        st.write(f"{pretty(engine.CLASSES[i])} - {probs[i]*100:.1f}%")
        st.progress(float(probs[i]))

with c2:
    st.subheader("Grad-CAM")
    target_names = [pretty(c) for c in engine.CLASSES]
    explain_label = st.selectbox("Explain class", target_names, index=top)
    t_idx = target_names.index(explain_label)
    cam, _ = engine.gradcam(model, model_name, rgb, class_idx=t_idx)
    ov = engine.overlay(rgb, cam, alpha)
    st.image(ov, width="stretch",
             caption="Warm colours = regions that pushed the model towards this class. "
                     "A model explanation, not a pixel-accurate lesion map.")
    buf = io.BytesIO(); Image.fromarray(ov).save(buf, format="PNG")
    st.download_button("Download heat-map", buf.getvalue(), "gradcam.png", "image/png")

# ── disease information ──────────────────────────────────────
info = INFO[cls]
st.divider()
st.header(f"About: {info['display']}")
a, b = st.columns(2)
with a:
    st.write(info["about"])
    st.write(f"**Severity / spread:** {info['severity']}")
    st.write("**What to look for**")
    for s in info["symptoms"]:
        st.write(f"- {s}")
with b:
    st.write("**General management ideas**")
    for s in info["management"]:
        st.write(f"- {s}")
st.caption(DISCLAIMER)