import os
# ** 在导入任何其他库之前，首先设置环境变量 **
os.environ['TRANSFORMERS_NO_TENSORFLOW'] = 'true'

import streamlit as st
import torch
import torch.nn as nn
import pickle
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from torchvision import transforms
from torchvision.models import resnet50
from ultralytics import YOLO
from scipy.spatial.distance import cosine
import io
import easyocr
from sentence_transformers import SentenceTransformer

# --- 配置 ---
# 使用相对路径，确保项目可移植
LIBRARY_FILE = Path('sku_library_multimodal.pkl')
FINETUNED_VISUAL_MODEL_PATH = Path('finetuned_resnet50_aug.pt')
YOLO_MODEL_PATH = 'runs/train/oriental_leaf_exp1/weights/best.pt'  # 更新为我们刚刚训练好的模型
TEXT_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
# --- (结束) ---

# --- 模型定义与加载 ---

class EmbeddingNet(nn.Module):
    """视觉特征提取模型结构"""
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        resnet = resnet50(weights=None)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
    def forward(self, x):
        x = self.features(x)
        return torch.flatten(x, 1)

class VisualFeatureExtractor:
    """封装了微调后的视觉模型"""
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EmbeddingNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    def get_feature_vector(self, image):
        image = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(image)
        return embedding.cpu().numpy().squeeze()

@st.cache_resource
def load_all_models():
    """一次性加载所有需要的模型"""
    print("正在加载所有模型...")
    detector = YOLO(YOLO_MODEL_PATH)
    visual_extractor = VisualFeatureExtractor(FINETUNED_VISUAL_MODEL_PATH)
    text_encoder = SentenceTransformer(TEXT_MODEL_NAME)
    ocr_reader = easyocr.Reader(['ch_sim', 'en'])
    print("所有模型加载完成。")
    return detector, visual_extractor, text_encoder, ocr_reader

@st.cache_resource
def load_prototype_library():
    """加载多模态原型知识库"""
    if not LIBRARY_FILE.exists():
        st.error(f"错误: 多模态知识库 {LIBRARY_FILE} 不存在！请先运行 build_prototype_library.py。")
        return None, None
    print("正在加载【多模态】SKU原型知识库...")
    with open(LIBRARY_FILE, 'rb') as f:
        prototype_library = pickle.load(f)
    class_names = list(prototype_library.keys())
    prototypes = np.array(list(prototype_library.values()))
    print("【多模态】SKU原型知识库加载完成。")
    return class_names, prototypes

# --- 主识别函数 ---
def run_recognition(image: Image.Image):
    detector, visual_extractor, text_encoder, ocr_reader = load_all_models()
    class_names, prototypes = load_prototype_library()

    if class_names is None: return None

    results = detector(image, classes=[39], verbose=False)
    detected_boxes = results[0].boxes.xyxy.cpu().numpy()

    if len(detected_boxes) == 0:
        st.warning("在图片中未能检测到任何'瓶子'。")
        return image

    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("Arial Unicode MS", 20)
    except IOError:
        font = ImageFont.load_default()

    for box in detected_boxes:
        cropped_image = image.crop(box)
        
        # --- **多模态特征提取** ---
        visual_embedding = visual_extractor.get_feature_vector(cropped_image)
        ocr_result = ocr_reader.readtext(np.array(cropped_image), detail=0, paragraph=True)
        ocr_text = " ".join(ocr_result)
        text_embedding = text_encoder.encode(ocr_text, convert_to_numpy=True)
        query_embedding = np.concatenate([visual_embedding, text_embedding])
        # --- **提取结束** ---

        similarities = 1 - np.array([cosine(query_embedding, p) for p in prototypes])
        
        best_match_index = np.argmax(similarities)
        best_match_label = class_names[best_match_index]
        best_match_score = similarities[best_match_index]
        
        label_text = f"{best_match_label} ({best_match_score:.2f})"
        
        draw.rectangle(box, outline="cyan", width=3)
        text_bbox = draw.textbbox((box[0], box[1]), label_text, font=font)
        draw.rectangle(text_bbox, fill="cyan")
        draw.text((box[0], box[1]), label_text, fill="black", font=font)
        
    return image

# --- Streamlit 应用界面 ---
st.set_page_config(layout="wide", page_title="多模态SKU识别系统")
st.title("👑 多模态SKU识别系统 (最终版)")
st.info("上传一张货架图片，系统将结合【视觉】与【文字】进行识别。")

uploaded_file = st.file_uploader("上传您的货架图片...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
    st.image(image, caption="您上传的图片", use_column_width=True)
    
    if st.button("开始识别", type="primary", use_container_width=True):
        with st.spinner('正在调用【多模态模型】进行联合分析...'):
            result_image = run_recognition(image)
        
        if result_image:
            st.success("识别完成！")
            st.image(result_image, caption="识别结果", use_column_width=True)
else:
    st.warning("请上传一张图片以开始。")
