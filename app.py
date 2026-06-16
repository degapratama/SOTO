import io
from typing import Tuple, List

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from timm import create_model
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── Konfigurasi ────────────────────────────────────────────────────────────────
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
STD  = np.array([0.229, 0.224, 0.225])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Fungsi Inti ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model(model_name: str) -> torch.nn.Module:
    cfg = MODEL_CONFIGS[model_name]
    model = create_model(cfg["arch"], pretrained=False, num_classes=len(CLASS_NAMES))
    state = torch.load(cfg["path"], map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


def preprocess_image(pil_img: Image.Image) -> Tuple[torch.Tensor, Image.Image]:
    img = pil_img.convert("RGB")
    w, h = img.size
    scale = 256 / min(w, h)
    img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
    w, h = img.size
    left = (w - IMG_SIZE) // 2
    top  = (h - IMG_SIZE) // 2
    img_cropped = img.crop((left, top, left + IMG_SIZE, top + IMG_SIZE))

    arr  = np.array(img_cropped, dtype=np.float32) / 255.0
    norm = (arr - MEAN) / STD
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).float()
    return tensor.unsqueeze(0).to(DEVICE), img_cropped


def attention_rollout(all_attn_weights: List[torch.Tensor]) -> np.ndarray:
    """
    Attention Rollout: menggabungkan attention dari SEMUA layer secara rekursif.
    Menghasilkan peta perhatian yang lebih akurat untuk model besar (DeiT Base).
    """
    result = torch.eye(all_attn_weights[0].size(-1)).to(all_attn_weights[0].device)

    for attn in all_attn_weights:
        # attn shape: (B, heads, tokens, tokens)
        # Ambil max antar heads (lebih fokus dibanding mean)
        attn_fused = attn[0].max(dim=0).values          # (tokens, tokens)

        # Tambahkan residual connection
        attn_fused = attn_fused + torch.eye(attn_fused.size(-1)).to(attn_fused.device)

        # Normalisasi per baris
        attn_fused = attn_fused / attn_fused.sum(dim=-1, keepdim=True)

        result = attn_fused @ result

    cls_attn = result[0, 1:]   # (num_patches,)
    return cls_attn.cpu().numpy()


@torch.no_grad()
def predict_with_attention(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
) -> Tuple[int, float, List[float], List[torch.Tensor]]:
    """
    Forward pass dengan hook untuk menangkap attention dari SEMUA blok transformer.
    """
    all_attn_weights = []
    hooks = []

    def make_hook(idx):
        def hook_fn(module, input, output):
            x = input[0]
            B, N, C = x.shape
            head_dim = C // module.num_heads
            qkv = module.qkv(x)
            qkv = qkv.reshape(B, N, 3, module.num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k, _ = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
            attn = F.softmax(attn, dim=-1)
            all_attn_weights.append(attn.detach())
        return hook_fn

    for i, block in enumerate(model.blocks):
        h = block.attn.register_forward_hook(make_hook(i))
        hooks.append(h)

    logits = model(input_tensor)
    probs  = F.softmax(logits, dim=1).squeeze().cpu().tolist()
    pred_idx = int(np.argmax(probs))

    for h in hooks:
        h.remove()

    return pred_idx, probs[pred_idx], probs, all_attn_weights


def attn_to_map(cls_attn: np.ndarray) -> np.ndarray:
    """Reshape attention 1D → 2D → resize ke 224x224."""
    num_patches = cls_attn.shape[0]
    grid_size   = int(np.sqrt(num_patches))
    attn_map    = cls_attn.reshape(grid_size, grid_size)
    attn_map    = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    attn_tensor = torch.from_numpy(attn_map).float().unsqueeze(0).unsqueeze(0)
    attn_tensor = F.interpolate(attn_tensor, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
    return attn_tensor.squeeze().cpu().numpy()


def make_overlay(attn_map_resized: np.ndarray, img_pil: Image.Image, alpha: float = 0.5) -> Image.Image:
    """Buat overlay heatmap + gambar asli."""
    norm         = plt.Normalize(vmin=0, vmax=1)
    heatmap_rgba = cm.jet(norm(attn_map_resized))
    heatmap_rgb  = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
    heatmap_pil  = Image.fromarray(heatmap_rgb, 'RGB')

    if img_pil.size != (IMG_SIZE, IMG_SIZE):
        img_pil = img_pil.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)

    return Image.blend(img_pil, heatmap_pil, alpha=alpha)


def generate_heatmap_overlay(
    all_attn_weights: List[torch.Tensor],
    img_pil: Image.Image,
    method: str = "rollout",
    threshold_pct: int = 0,
) -> Image.Image:
    """
    Menghasilkan overlay heatmap.
    method: 'rollout' | 'mean' | 'max'
    threshold_pct: 0 = tanpa threshold, >0 = persentil cutoff
    """
    last_attn = all_attn_weights[-1]   # (B, heads, tokens, tokens)

    if method == "rollout":
        cls_attn = attention_rollout(all_attn_weights)
    elif method == "mean":
        attn_avg = last_attn[0].mean(dim=0)   # (tokens, tokens)
        cls_attn = attn_avg[0, 1:].cpu().numpy()
    else:  # max
        attn_max = last_attn[0].max(dim=0).values
        cls_attn = attn_max[0, 1:].cpu().numpy()

    attn_map = attn_to_map(cls_attn)

    if threshold_pct > 0:
        threshold = np.percentile(attn_map, threshold_pct)
        attn_map  = np.where(attn_map >= threshold, attn_map, attn_map * 0.1)
        attn_map  = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    return make_overlay(attn_map, img_pil)


def generate_per_head_overlays(
    all_attn_weights: List[torch.Tensor],
    img_pil: Image.Image,
) -> List[Tuple[str, Image.Image]]:
    """
    Buat overlay heatmap untuk SETIAP head di layer terakhir.
    Return list of (label, overlay_image).
    """
    last_attn  = all_attn_weights[-1]   # (B, heads, tokens, tokens)
    num_heads  = last_attn.shape[1]
    results    = []

    for h_idx in range(num_heads):
        attn_head = last_attn[0, h_idx]       # (tokens, tokens)
        cls_attn  = attn_head[0, 1:].cpu().numpy()   # (num_patches,)
        attn_map  = attn_to_map(cls_attn)
        overlay   = make_overlay(attn_map, img_pil, alpha=0.5)
        results.append((f"Head {h_idx + 1}", overlay))

    return results


# ── UI Helpers ──────────────────────────────────────────────────────────────

def format_label_name(name: str) -> str:
    return name.replace("_", " ").title()


def render_confidence_bars(probs: List[float]) -> None:
    for i, prob in enumerate(probs):
        col1, col2 = st.columns([3, 7])
        col1.caption(format_label_name(CLASS_NAMES[i]))
        col2.progress(prob, text=f"{prob * 100:.1f}%")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Soto Classifier", page_icon="🍜", layout="wide")
    st.title("Soto Classifier")

    with st.sidebar:
        st.header("Pengaturan Model")
        model_name = st.radio("Pilih Model:", options=list(MODEL_CONFIGS.keys()), index=0)
        st.caption(f"Berjalan menggunakan: **{str(DEVICE).upper()}**")

        st.divider()
        st.header("Pengaturan Heatmap")
        heatmap_method = st.selectbox(
            "Metode Agregasi:",
            options=["rollout", "max", "mean"],
            format_func=lambda x: {
                "rollout": "Attention Rollout (direkomendasikan)",
                "max":     "Max antar heads (terfokus)",
                "mean":    "Mean antar heads (tersebar)",
            }[x],
        )
        threshold_pct = st.slider(
            "Threshold (%)",
            min_value=0,
            max_value=90,
            value=0,
            step=5,
            help="0 = tanpa threshold (tampilkan semua). Semakin tinggi = semakin fokus ke area penting saja.",
        )
        show_per_head = st.checkbox("Tampilkan semua head secara terpisah", value=False)

        st.divider()
        st.markdown("**Daftar Kelas:**")
        for cls in CLASS_NAMES:
            st.markdown(f"- {format_label_name(cls)}")

    with st.spinner(f"Memuat model {model_name}…"):
        try:
            model = load_model(model_name)
        except FileNotFoundError:
            st.error(f"File model tidak ditemukan: `{MODEL_CONFIGS[model_name]['path']}`")
            st.stop()

    uploaded_file = st.file_uploader(
        "Unggah gambar soto",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )
    if not uploaded_file:
        st.info("Silakan unggah gambar soto terlebih dahulu.", icon="📂")
        st.stop()

    pil_img_original = Image.open(io.BytesIO(uploaded_file.read()))
    input_tensor, pil_img_cropped = preprocess_image(pil_img_original)

    with st.spinner("Sedang memproses gambar…"):
        pred_idx, confidence, probs, all_attn_weights = predict_with_attention(model, input_tensor)

    pred_label = format_label_name(CLASS_NAMES[pred_idx])
    st.subheader(f"Prediksi: **{pred_label}**")
    st.metric("Tingkat Keyakinan", f"{confidence * 100:.2f}%")
    st.divider()

    # ── Gambar Asli + Heatmap Utama ──
    col1, col2 = st.columns(2)
    with col1:
        st.image(pil_img_original, caption="Gambar Asli", use_container_width=True)
    with col2:
        if all_attn_weights:
            try:
                overlay_img = generate_heatmap_overlay(
                    all_attn_weights,
                    pil_img_cropped,
                    method=heatmap_method,
                    threshold_pct=threshold_pct,
                )
                method_label = {
                    "rollout": "Attention Rollout",
                    "max":     "Max Heads",
                    "mean":    "Mean Heads",
                }[heatmap_method]
                thr_label = f" | Threshold {threshold_pct}%" if threshold_pct > 0 else " | Tanpa Threshold"
                st.image(overlay_img, caption=f"Heatmap Overlay ({method_label}{thr_label})", use_container_width=True)
            except Exception as e:
                st.warning(f"Gagal membuat heatmap: {e}")
                st.image(pil_img_cropped, caption="Gambar yang diproses (tanpa heatmap)", use_container_width=True)
        else:
            st.info("Attention tidak tersedia.")
            st.image(pil_img_cropped, caption="Gambar yang diproses", use_container_width=True)

    # ── Tampilkan Semua Head ──
    if show_per_head and all_attn_weights:
        st.divider()
        num_heads = all_attn_weights[-1].shape[1]
        st.subheader(f"Visualisasi Per Head — Layer Terakhir ({num_heads} heads)")
        st.caption("Setiap head menunjukkan 'sudut pandang' perhatian yang berbeda. Tanpa threshold.")

        with st.spinner("Membuat visualisasi per head…"):
            per_head_imgs = generate_per_head_overlays(all_attn_weights, pil_img_cropped)

        # Tampilkan dalam grid 4 kolom
        cols_per_row = 4
        for row_start in range(0, len(per_head_imgs), cols_per_row):
            row_imgs = per_head_imgs[row_start : row_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for col, (label, img) in zip(cols, row_imgs):
                with col:
                    st.image(img, caption=label, use_container_width=True)

    st.divider()
    st.subheader("Probabilitas Kelas")
    render_confidence_bars(probs)


if __name__ == "__main__":
    main()
