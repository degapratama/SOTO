import io
from typing import Tuple, List

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from timm import create_model

# Konfigurasi
CLASS_NAMES = [
    "coto_makassar",
    "soto_bandung",
    "soto_betawi",
    "soto_lamongan",
    "soto_medan",
    "soto_padang",
]

MODEL_CONFIGS = {
    "DeiT Base": {
        "arch": "deit_base_patch16_224",
        "path": "Model/Deit_Base.pth",
    },
    "DeiT Tiny": {
        "arch": "deit_tiny_patch16_224",
        "path": "Model/Deit_Tiny.pth",
    },
}

IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource(show_spinner=False)
def load_model(model_name: str) -> torch.nn.Module:
    cfg = MODEL_CONFIGS[model_name]
    model = create_model(cfg["arch"], pretrained=False, num_classes=len(CLASS_NAMES))
    state = torch.load(cfg["path"], map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model

def preprocess_image(pil_img: Image.Image) -> Tuple[torch.Tensor, Image.Image]:
    """
    Preprocess image: resize, center crop, normalize.
    Returns tensor and also the cropped PIL image for display.
    """
    img = pil_img.convert("RGB")
    w, h = img.size
    scale = 256 / min(w, h)
    img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
    w, h = img.size
    left = (w - IMG_SIZE) // 2
    top = (h - IMG_SIZE) // 2
    img_cropped = img.crop((left, top, left + IMG_SIZE, top + IMG_SIZE))
    
    arr = np.array(img_cropped, dtype=np.float32) / 255.0
    norm = (arr - MEAN) / STD
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).float()
    return tensor.unsqueeze(0).to(DEVICE), img_cropped

@torch.no_grad()
def predict_image(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
) -> Tuple[int, float, List[float]]:
    logits = model(input_tensor)
    probs = F.softmax(logits, dim=1).squeeze().cpu().tolist()
    pred_idx = int(np.argmax(probs))
    return pred_idx, probs[pred_idx], probs

def format_label_name(name: str) -> str:
    return name.replace("_", " ").title()

def render_confidence_bars(probs: List[float]) -> None:
    for i, prob in enumerate(probs):
        col1, col2 = st.columns([3, 7])
        col1.caption(format_label_name(CLASS_NAMES[i]))
        col2.progress(prob, text=f"{prob * 100:.1f}%")

def main():
    st.set_page_config(page_title="Soto Classifier", page_icon="🍜", layout="centered")
    st.title("Soto Classifier")

    with st.sidebar:
        st.header("Pengaturan Model")
        model_name = st.radio(
            "Pilih Model:",
            options=list(MODEL_CONFIGS.keys()),
            index=0,
        )
        st.caption(f"Berjalan menggunakan: **{str(DEVICE).upper()}**")
        st.divider()
        st.markdown("**Daftar Kelas (Kategori):**")
        for cls in CLASS_NAMES:
            st.markdown(f"- {format_label_name(cls)}")

    with st.spinner(f"Memuat model {model_name}…"):
        try:
            model = load_model(model_name)
        except FileNotFoundError:
            st.error(
                f"File model tidak ditemukan: `{MODEL_CONFIGS[model_name]['path']}`\n\n"
                "Pastikan file `.pth` berada di direktori yang tepat."
            )
            st.stop()

    uploaded_file = st.file_uploader(
        "Unggah gambar soto",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )
    if not uploaded_file:
        st.info("Silakan unggah gambar soto terlebih dahulu.", icon="📂")
        st.stop()

    pil_img = Image.open(io.BytesIO(uploaded_file.read()))
    input_tensor, processed_img = preprocess_image(pil_img)

    with st.spinner("Sedang memproses gambar…"):
        pred_idx, confidence, probs = predict_image(model, input_tensor)

    pred_label = format_label_name(CLASS_NAMES[pred_idx])
    st.subheader(f"Prediksi: **{pred_label}**")
    st.metric("Tingkat Keyakinan", f"{confidence * 100:.2f}%")
    st.divider()

    # Tampilkan gambar
    st.image(processed_img, caption="Gambar yang diproses (224x224)", use_container_width=True)

    st.divider()
    st.subheader("Probabilitas Kelas")
    render_confidence_bars(probs)

if __name__ == "__main__":
    main()
