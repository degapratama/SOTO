import io
from typing import Tuple, List

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from timm import create_model

# Untuk heatmap
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── Konfigurasi & Konstanta ───────────────────────────────────────────────────

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

# ── Fungsi Pemrosesan & Model (Core Logic) ────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model(model_name: str) -> torch.nn.Module:
    """Memuat dan menyimpan model DeiT dari penyimpanan lokal ke dalam cache."""
    cfg = MODEL_CONFIGS[model_name]
    model = create_model(cfg["arch"], pretrained=False, num_classes=len(CLASS_NAMES))
    state = torch.load(cfg["path"], map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model

def preprocess_image(pil_img: Image.Image) -> Tuple[torch.Tensor, Image.Image]:
    """
    Preprocess image: resize, center crop, normalize.
    Returns tensor and cropped PIL image for display.
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
def predict_with_attention(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
) -> Tuple[int, float, List[float], torch.Tensor]:
    """
    Forward pass dengan hook untuk menangkap attention dari blok terakhir.
    Menghitung attention weights secara manual dari Q dan K.
    """
    attn_weights = None

    def hook_fn(module, input, output):
        # input adalah tuple (x,) ; x shape (batch, tokens, dim)
        x = input[0]  # (B, N, C)
        B, N, C = x.shape
        
        # Dapatkan qkv dari modul attention
        qkv = module.qkv(x)  # (B, N, 3*C)
        # Reshape menjadi (B, N, 3, num_heads, head_dim)
        qkv = qkv.reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # masing-masing (B, heads, N, head_dim)
        
        # Hitung attention scores
        attn = (q @ k.transpose(-2, -1)) * (C // module.num_heads) ** -0.5
        attn = F.softmax(attn, dim=-1)  # (B, heads, N, N)
        
        # Simpan ke variabel luar
        nonlocal attn_weights
        attn_weights = attn.detach()

    # Pasang hook pada modul attention di blok terakhir
    handle = model.blocks[-1].attn.register_forward_hook(hook_fn)

    # Forward pass
    logits = model(input_tensor)
    probs = F.softmax(logits, dim=1).squeeze().cpu().tolist()
    pred_idx = int(np.argmax(probs))

    # Lepas hook
    handle.remove()

    return pred_idx, probs[pred_idx], probs, attn_weights

def generate_heatmap(
    attn: torch.Tensor,
    img_pil: Image.Image,
) -> Tuple[Image.Image, Image.Image]:
    """
    Membuat heatmap dan overlay dari attention blok terakhir.
    Mengembalikan (heatmap_image, overlay_image) dalam format PIL.
    """
    # attn shape: (batch, heads, tokens, tokens)
    attn = attn[0]  # (heads, tokens, tokens)
    
    # Rata-rata antar kepala
    attn = attn.mean(dim=0)  # (tokens, tokens)
    
    # Ambil perhatian dari class token ke semua patch (indeks 0 ke 1..end)
    attn_class = attn[0, 1:]  # (num_patches,)
    
    # Reshape ke grid 14x14 (patch size 16)
    num_patches = attn_class.shape[0]
    grid_size = int(np.sqrt(num_patches))  # 14
    attn_map = attn_class.reshape(grid_size, grid_size).cpu().numpy()
    
    # Normalisasi
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    
    # Interpolasi ke 224x224
    attn_tensor = torch.from_numpy(attn_map).float().unsqueeze(0).unsqueeze(0)
    attn_tensor = F.interpolate(
        attn_tensor, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False
    )
    attn_map_resized = attn_tensor.squeeze().cpu().numpy()
    
    # Buat heatmap berwarna dengan colormap 'jet'
    norm = plt.Normalize(vmin=0, vmax=1)
    heatmap_rgba = cm.jet(norm(attn_map_resized))  # (224,224,4)
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap_rgb, 'RGB')
    
    # Pastikan gambar asli ukuran 224x224
    if img_pil.size != (IMG_SIZE, IMG_SIZE):
        img_pil = img_pil.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    
    # Overlay dengan alpha 0.5
    overlay = Image.blend(img_pil, heatmap_pil, alpha=0.5)
    
    return heatmap_pil, overlay

def format_label_name(name: str) -> str:
    return name.replace("_", " ").title()

def render_confidence_bars(probs: List[float]) -> None:
    for i, prob in enumerate(probs):
        col1, col2 = st.columns([3, 7])
        col1.caption(format_label_name(CLASS_NAMES[i]))
        col2.progress(prob, text=f"{prob * 100:.1f}%")

def main():
    st.set_page_config(
        page_title="Soto Classifier",
        page_icon="🍜",
        layout="centered",
    )

    st.title("Soto Classifier")

    # Konfigurasi Sidebar
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

    # Memuat Model
    with st.spinner(f"Memuat model {model_name}…"):
        try:
            model = load_model(model_name)
        except FileNotFoundError:
            st.error(
                f"File model tidak ditemukan: `{MODEL_CONFIGS[model_name]['path']}`\n\n"
                "Pastikan file `.pth` berada di direktori yang tepat."
            )
            st.stop()

    # Unggah Gambar
    uploaded_file = st.file_uploader(
        "Unggah gambar soto",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if not uploaded_file:
        st.info("Silakan unggah gambar soto terlebih dahulu.", icon="📂")
        st.stop()

    # ── Proses Inferensi & Heatmap ──
    pil_img = Image.open(io.BytesIO(uploaded_file.read()))
    input_tensor, processed_img = preprocess_image(pil_img)

    with st.spinner("Sedang memproses gambar…"):
        pred_idx, confidence, probs, attn = predict_with_attention(model, input_tensor)

    pred_label = format_label_name(CLASS_NAMES[pred_idx])

    st.subheader(f"Prediksi: **{pred_label}**")
    st.metric("Tingkat Keyakinan", f"{confidence * 100:.2f}%")
    st.divider()

    # Tampilkan gambar asli dan heatmap overlay dalam 2 kolom
    col1, col2 = st.columns(2)
    with col1:
        st.image(
            processed_img,
            caption="Gambar Asli",
            use_container_width=True,
        )
    with col2:
        if attn is not None:
            try:
                _, overlay_img = generate_heatmap(attn, processed_img)
                st.image(
                    overlay_img,
                    caption="Attention Heatmap Overlay",
                    use_container_width=True,
                )
            except Exception as e:
                st.warning(f"Tidak dapat menampilkan heatmap: {e}")
                st.image(processed_img, caption="Gambar Asli (Heatmap gagal)", use_container_width=True)
        else:
            st.info("Attention tidak tersedia untuk model ini.")
            st.image(processed_img, caption="Gambar Asli (tanpa heatmap)", use_container_width=True)

    st.divider()
    st.subheader("Probabilitas Kelas")
    render_confidence_bars(probs)

if __name__ == "__main__":
    main()
