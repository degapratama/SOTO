import io
from typing import Tuple, List

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from timm import create_model
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ---------- Konfigurasi ----------
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

# ---------- Fungsi Pemrosesan & Model ----------
@st.cache_resource(show_spinner=False)
def load_model(model_name: str) -> torch.nn.Module:
    """Memuat model DeiT dari penyimpanan lokal."""
    cfg = MODEL_CONFIGS[model_name]
    model = create_model(cfg["arch"], pretrained=False, num_classes=len(CLASS_NAMES))
    state = torch.load(cfg["path"], map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model

def preprocess_image(pil_img: Image.Image) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Resize dan center-crop ke 224x224, lalu normalisasi untuk input model.
    Mengembalikan tensor dan array RGB (0-1) untuk visualisasi.
    """
    img = pil_img.convert("RGB")
    # Resize sisi terpendek ke 256
    w, h = img.size
    scale = 256 / min(w, h)
    img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
    # Center crop
    w, h = img.size
    left = (w - IMG_SIZE) // 2
    top = (h - IMG_SIZE) // 2
    img = img.crop((left, top, left + IMG_SIZE, top + IMG_SIZE))

    arr = np.array(img, dtype=np.float32) / 255.0   # 0-1 RGB
    norm = (arr - MEAN) / STD
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).float()
    return tensor.unsqueeze(0).to(DEVICE), arr

def reshape_transform(tensor: torch.Tensor, height: int = 14, width: int = 14) -> torch.Tensor:
    """Ubah token ViT menjadi feature map spatial untuk Grad-CAM."""
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return result.transpose(2, 3).transpose(1, 2)

def generate_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    pred_class: int,
) -> np.ndarray:
    """Hasilkan heatmap Grad-CAM++."""
    target_layers = [model.blocks[-1].norm1]
    with GradCAMPlusPlus(
        model=model,
        target_layers=target_layers,
        reshape_transform=reshape_transform,
    ) as cam:
        heatmap = cam(
            input_tensor=input_tensor,
            targets=[ClassifierOutputTarget(pred_class)],
        )[0]
    return heatmap

@torch.no_grad()
def predict_image(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
) -> Tuple[int, float, List[float]]:
    """Prediksi kelas dan probabilitas."""
    logits = model(input_tensor)
    probs = F.softmax(logits, dim=1).squeeze().cpu().tolist()
    pred_idx = int(np.argmax(probs))
    return pred_idx, probs[pred_idx], probs

# ---------- Fungsi Overlay Grad-CAM (tanpa OpenCV) ----------
def overlay_heatmap_on_image(
    rgb_img: np.ndarray,      # shape (H,W,3) dalam rentang 0-1
    heatmap: np.ndarray,      # shape (H,W) nilai 0-1
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Overlay heatmap pada gambar RGB menggunakan PIL.
    Mengembalikan array uint8 (0-255) untuk ditampilkan di Streamlit.
    """
    # Resize heatmap ke ukuran gambar asli
    h, w = rgb_img.shape[:2]
    heatmap_resized = Image.fromarray((heatmap * 255).astype(np.uint8), mode='L')
    heatmap_resized = heatmap_resized.resize((w, h), Image.Resampling.BILINEAR)
    heatmap_resized = np.array(heatmap_resized) / 255.0  # 0-1

    # Buat colormap (merah) sederhana: (R=1, G=heatmap, B=heatmap) -> efek merah
    # Atau bisa juga gunakan jet colormap, tapi kita buat sederhana
    heatmap_colored = np.zeros((h, w, 3), dtype=np.float32)
    heatmap_colored[..., 0] = 1.0          # R = 1
    heatmap_colored[..., 1] = heatmap_resized
    heatmap_colored[..., 2] = heatmap_resized

    # Overlay
    overlay = (1 - alpha) * rgb_img + alpha * heatmap_colored
    overlay = np.clip(overlay, 0, 1)
    overlay_uint8 = (overlay * 255).astype(np.uint8)
    return overlay_uint8

# ---------- UI ----------
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

    # Sidebar
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

    # Muat model
    with st.spinner(f"Memuat model {model_name}…"):
        try:
            model = load_model(model_name)
        except FileNotFoundError:
            st.error(
                f"File model tidak ditemukan: `{MODEL_CONFIGS[model_name]['path']}`\n\n"
                "Pastikan file `.pth` berada di direktori yang tepat."
            )
            st.stop()

    # Unggah gambar
    uploaded_file = st.file_uploader(
        "Unggah gambar soto",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )
    if not uploaded_file:
        st.info("Silakan unggah gambar soto terlebih dahulu.", icon="📂")
        st.stop()

    # Proses
    pil_img = Image.open(io.BytesIO(uploaded_file.read()))
    input_tensor, rgb_arr = preprocess_image(pil_img)

    with st.spinner("Sedang memproses gambar…"):
        pred_idx, confidence, probs = predict_image(model, input_tensor)

    with st.spinner("Membuat visualisasi Grad-CAM…"):
        heatmap = generate_gradcam(model, input_tensor, pred_idx)
        cam_img = overlay_heatmap_on_image(rgb_arr, heatmap)

    # Tampilkan hasil
    pred_label = format_label_name(CLASS_NAMES[pred_idx])
    st.subheader(f"Prediksi: **{pred_label}**")
    st.metric("Tingkat Keyakinan", f"{confidence * 100:.2f}%")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.image(pil_img, caption="Gambar Asli", use_container_width=True)
    with col2:
        st.image(cam_img, caption="Grad-CAM Overlay", use_container_width=True)

    st.divider()
    st.subheader("Probabilitas Kelas")
    render_confidence_bars(probs)

if __name__ == "__main__":
    main()
