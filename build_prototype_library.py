import os
# ** 在导入任何其他库之前，首先设置环境变量 **
# ** 这会告诉transformers库不要加载TensorFlow **
os.environ['TRANSFORMERS_NO_TENSORFLOW'] = 'true'

import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision.models import resnet50
import pickle
from tqdm import tqdm
import easyocr
from sentence_transformers import SentenceTransformer

# --- 配置 ---
# 使用相对路径，确保项目可移植
ANNOTATIONS_CSV = Path('bbox_labels.csv')
FINETUNED_VISUAL_MODEL_PATH = Path('finetuned_resnet50_aug.pt')
# ** 输出最终的多模态知识库 **
OUTPUT_LIBRARY_FILE = Path('sku_library_multimodal.pkl')
# ** Sentence Transformer 模型，用于文本编码 **
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

def main():
    print("--- 开始构建【多模态】SKU原型知识库 ---")

    # 1. 加载所有需要的模型
    print("正在加载所有模型...")
    visual_extractor = VisualFeatureExtractor(FINETUNED_VISUAL_MODEL_PATH)
    text_encoder = SentenceTransformer(TEXT_MODEL_NAME)
    ocr_reader = easyocr.Reader(['ch_sim', 'en'])
    print("所有模型加载完成。")

    # 2. 加载标注数据
    df = pd.read_csv(ANNOTATIONS_CSV)
    print(f"成功加载 {len(df)} 条标注。")

    # 3. 计算多模态原型
    multimodal_library = {}
    for label, group in tqdm(df.groupby('label'), desc="计算多模态原型"):
        multimodal_embeddings = []
        for _, row in group.iterrows():
            try:
                img_path = Path(row['image_path'])
                image = Image.open(img_path).convert("RGB")
                bbox = (row['xmin'], row['ymin'], row['xmax'], row['ymax'])
                cropped_image = image.crop(bbox)
                
                # 提取视觉特征
                visual_embedding = visual_extractor.get_feature_vector(cropped_image)
                
                # 提取文本特征
                ocr_result = ocr_reader.readtext(np.array(cropped_image), detail=0, paragraph=True)
                ocr_text = " ".join(ocr_result)
                text_embedding = text_encoder.encode(ocr_text, convert_to_numpy=True)
                
                # ** 拼接视觉和文本特征，形成“超级指纹” **
                multimodal_embedding = np.concatenate([visual_embedding, text_embedding])
                multimodal_embeddings.append(multimodal_embedding)

            except Exception as e:
                print(f"警告: 处理图片 {row['image_path']} 时出错: {e}")
                continue
        
        if multimodal_embeddings:
            prototype = np.mean(multimodal_embeddings, axis=0)
            multimodal_library[label] = prototype
    
    # 4. 保存最终的知识库
    with open(OUTPUT_LIBRARY_FILE, 'wb') as f:
        pickle.dump(multimodal_library, f)

    print("\n--- 【多模态】知识库构建完成！ ---")
    print(f"已将最终版的多模态知识库保存至: {OUTPUT_LIBRARY_FILE}")

if __name__ == '__main__':
    main()