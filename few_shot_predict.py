import torch
import pickle
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from ultralytics import YOLO
from scipy.spatial.distance import cosine
import argparse

# --- 配置 ---
# 知识库文件路径
LIBRARY_FILE = Path('/Users/guohongbin/projects/识别/sku_library.pkl')
# 用于检测的YOLO模型
YOLO_MODEL_PATH = 'yolov8n.pt'
# --- (结束) ---

class FeatureExtractor:
    """
    一个封装了预训练模型以提取特征的类。
    (与 build_prototype_library.py 中的类相同)
    """
    def __init__(self):
        weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=weights)
        self.model.eval()
        self.feature_layer = self.model._modules.get('avgpool')
        self.transform = weights.transforms()
        self.embedding = None
        self.feature_layer.register_forward_hook(self.copy_embedding)

    def copy_embedding(self, module, input, output):
        self.embedding = output.clone().detach().squeeze()

    def get_feature_vector(self, image):
        image = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            self.model(image)
        return self.embedding.numpy()

def main(image_path: Path):
    """
    主函数，执行完整的检测和识别流程。
    """
    print("--- 开始执行小样本SKU识别 ---")

    # 1. 加载所有需要的模型和数据
    print("正在加载工具...")
    try:
        # 加载通用物体检测器
        detector = YOLO(YOLO_MODEL_PATH)
        # 加载特征提取器
        feature_extractor = FeatureExtractor()
        # 加载原型知识库
        with open(LIBRARY_FILE, 'rb') as f:
            prototype_library = pickle.load(f)
        
        class_names = list(prototype_library.keys())
        prototypes = np.array(list(prototype_library.values()))
        print("所有工具加载成功！")
    except Exception as e:
        print(f"错误: 加载模型或知识库失败: {e}")
        return

    # 2. 读取并处理输入图片
    if not image_path.exists():
        print(f"错误: 输入图片 {image_path} 不存在。")
        return
    
    image = Image.open(image_path).convert("RGB")

    # 3. 阶段一：通用物体检测
    print("阶段一: 正在检测图片中的物体...")
    # 我们只关心 'bottle' 类别 (在COCO数据集中类别号为39)
    results = detector(image, classes=[39], verbose=False)
    
    detected_boxes = results[0].boxes.xyxy.cpu().numpy()
    if len(detected_boxes) == 0:
        print("未检测到任何'瓶子'。程序结束。")
        return
    print(f"检测到 {len(detected_boxes)} 个潜在目标。")

    # 准备在图片上绘制结果
    draw = ImageDraw.Draw(image)
    try:
        # 使用系统中的一个常见中文字体
        font = ImageFont.truetype("Arial Unicode MS", 20)
    except IOError:
        print("警告: 未找到 Arial Unicode MS 字体，将使用默认字体。")
        font = ImageFont.load_default()

    # 4. 阶段二：小样本识别
    print("阶段二: 正在识别每个目标...")
    for box in detected_boxes:
        # 裁剪出检测到的物体
        cropped_image = image.crop(box)
        
        # 提取特征"指纹"
        query_embedding = feature_extractor.get_feature_vector(cropped_image)
        
        # 计算与知识库中所有原型的相似度 (1 - 余弦距离)
        similarities = 1 - np.array([cosine(query_embedding, p) for p in prototypes])
        
        # 找到最相似的原型
        best_match_index = np.argmax(similarities)
        best_match_label = class_names[best_match_index]
        best_match_score = similarities[best_match_index]
        
        # 在图片上绘制边界框和标签
        label_text = f"{best_match_label} ({best_match_score:.2f})"
        
        draw.rectangle(box, outline="red", width=3)
        
        # 绘制文本背景
        text_bbox = draw.textbbox((box[0], box[1]), label_text, font=font)
        draw.rectangle(text_bbox, fill="red")
        draw.text((box[0], box[1]), label_text, fill="white", font=font)

    # 5. 显示结果
    print("--- 识别完成！---")
    image.show()
    
    # 保存结果图片
    output_path = image_path.parent / f"{image_path.stem}_result.jpg"
    image.save(output_path)
    print(f"结果已保存至: {output_path}")


if __name__ == '__main__':
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="小样本SKU识别工具")
    parser.add_argument("image_path", type=str, help="需要识别的图片路径")
    args = parser.parse_args()
    
    main(Path(args.image_path))
