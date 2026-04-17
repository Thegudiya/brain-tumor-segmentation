import streamlit as st
import torch
import cv2
import numpy as np

# ===== IMPORT MODEL =====
from src.model import UNet

# ===== IMPORT UTILS =====
from src.utils import calculate_tumor_percentage, classify_severity

# ===== LOAD MODEL =====
model = UNet()
model.load_state_dict(torch.load("outputs/models/unet.pth", map_location=torch.device('cpu')))
model.eval()

# ===== UI =====
st.title("Brain Tumor Segmentation")
st.markdown("Upload an MRI image to detect tumor region")

# ===== FILE UPLOAD =====
uploaded_file = st.file_uploader("Upload MRI Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:

    # ===== READ IMAGE =====
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # ===== PREPROCESS =====
    img_resized = cv2.resize(img, (128,128))
    img_norm = img_resized / 255.0
    img_tensor = torch.tensor(img_norm).permute(2,0,1).unsqueeze(0).float()

    # ===== PREDICTION =====
    with torch.no_grad():
        pred = model(img_tensor)

    # ✅ Apply sigmoid
    pred = torch.sigmoid(pred)

    # ===== THRESHOLD =====
    pred_img = (pred[0][0].numpy() > 0.5).astype("float32")

    # ===== NOISE REMOVAL =====
    pred_img = (pred_img * 255).astype("uint8")
    pred_img = cv2.medianBlur(pred_img, 5)

    # ===== KEEP ONLY LARGEST COMPONENT (IMPORTANT FIX) =====
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred_img)

    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        clean_mask = np.zeros_like(pred_img)
        clean_mask[labels == largest_label] = 255
        pred_img = clean_mask

    # ===== RESIZE BACK =====
    pred_img = cv2.resize(pred_img, (img.shape[1], img.shape[0]))

    # ===== CALCULATE TUMOR % =====
    percentage = calculate_tumor_percentage(pred_img)
    severity = classify_severity(percentage)

    # ===== DISPLAY =====
    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Input MRI", width=300)

    with col2:
        st.image(pred_img, caption="Tumor Prediction", width=300)

    # ===== RESULT =====
    if np.sum(pred_img) > 0:
        st.success("Tumor Detected ✅")

        st.metric("Tumor %", f"{percentage}%")
        st.metric("Severity", severity)

    else:
        st.info("No Tumor Detected")