# SKU 识别系统

这是一个基于多模态（视觉与文本）的 SKU（库存单位）识别系统。它结合了先进的深度学习模型，包括 YOLOv8 进行目标检测、微调后的 ResNet50 进行视觉特征提取、EasyOCR 进行文本识别以及 Sentence Transformers 进行文本嵌入，最终通过多模态特征匹配实现对商品 SKU 的精准识别。

## 主要功能

*   **多模态 SKU 识别**: 结合图像的视觉特征和 OCR 提取的文本信息，对检测到的商品进行识别和分类。
*   **通用目标检测**: 使用 YOLOv8 模型在图像中自动检测出潜在的商品（如瓶子）。
*   **视觉特征提取**: 利用微调后的 ResNet50 模型从商品图像中提取高维视觉特征。
*   **文本识别 (OCR)**: 集成 EasyOCR 库，从商品图像的局部区域识别文字信息。
*   **文本特征嵌入**: 使用 Sentence Transformers 将 OCR 识别的文本转换为语义向量。
*   **原型库匹配**: 将提取到的多模态特征与预先构建的 SKU 原型知识库进行匹配，找出最相似的 SKU。
*   **数据标注工具**: 提供基于 Streamlit 的交互式工具，辅助进行图像标注和数据集准备。
*   **模型训练与微调**: 包含用于训练 YOLO 模型和微调 ResNet50 特征提取器的脚本。

## 项目结构

```
.
├── app.py                          # 主 Streamlit 应用，实现多模态 SKU 识别
├── requirements.txt                # 项目依赖库列表
├── train.py                        # YOLOv8 模型训练脚本
├── finetune_recognizer.py          # ResNet50 特征提取器微调脚本 (基于三元组损失)
├── yolo_dataset_converter.py       # 将标注数据转换为 YOLO 格式的脚本
├── streamlit_tagger.py             # 基于 Streamlit 的图像标注工具 (用于生成 labels.csv)
├── few_shot_predict.py             # 小样本 SKU 识别示例脚本 (早期版本或替代方案)
├── bbox_labels.csv                 # 边界框标注文件 (用于 yolo_dataset_converter.py 和 finetune_recognizer.py)
├── labels.csv                      # 图像标签文件 (由 streamlit_tagger.py 生成)
├── sku_library_multimodal.pkl      # 多模态 SKU 原型知识库 (由 build_prototype_library.py 生成，供 app.py 使用)
├── finetuned_resnet50_aug.pt       # 微调后的 ResNet50 视觉特征提取模型
├── yolov8n.pt                      # 预训练的 YOLOv8n 模型权重
├── yolo_dataset/                   # YOLO 格式的数据集目录 (由 yolo_dataset_converter.py 生成)
│   ├── images/
│   └── labels/
├── 东方树叶/                       # 示例图片目录
├── 可口可乐/                       # 示例图片目录
└── 饮料-700张/                     # 原始图片数据集目录
```

## 安装

1.  **克隆仓库**:
    ```bash
    git clone https://github.com/guohongbin-git/sku.git
    cd sku
    ```

2.  **创建并激活虚拟环境** (推荐):
    ```bash
    python -m venv venv
    # macOS/Linux
    source venv/bin/activate
    # Windows
    .\venv\Scripts\activate
    ```

3.  **安装依赖**:
    ```bash
    pip install -r requirements.txt
    ```

## 使用指南

### 1. 数据准备

*   **原始图片**: 将您的原始商品图片放置在如 `饮料-700张/` 这样的目录下。
*   **边界框标注**: 使用 `streamlit_box_tagger.py` (如果存在) 或其他工具生成包含边界框信息的 `bbox_labels.csv` 文件。该文件应包含 `image_path`, `xmin`, `ymin`, `xmax`, `ymax`, `label` 等列。
*   **YOLO 数据集转换**: 运行 `yolo_dataset_converter.py` 将 `bbox_labels.csv` 转换为 YOLO 训练所需的格式。
    ```bash
    python yolo_dataset_converter.py
    ```
    这将在 `yolo_dataset/` 目录下生成 `images/` 和 `labels/` 子目录，以及 `data.yaml` 配置文件。
*   **文本标签标注**: 使用 `streamlit_tagger.py` 工具，通过 OCR 辅助为图片添加文本标签，生成 `labels.csv`。
    ```bash
    streamlit run streamlit_tagger.py
    ```

### 2. 模型训练与微调

*   **训练 YOLOv8 检测器**:
    ```bash
    python train.py
    ```
    此脚本将使用 `yolo_dataset/` 中的数据微调 YOLOv8n 模型，训练结果将保存在 `runs/train/` 目录下。

*   **微调 ResNet50 视觉特征提取器**:
    ```bash
    python finetune_recognizer.py
    ```
    此脚本将使用 `bbox_labels.csv` 中的数据，通过三元组损失微调 ResNet50 模型，以提高视觉特征的区分度。微调后的模型将保存为 `finetuned_resnet50_aug.pt`。

### 3. 构建 SKU 原型知识库

在运行主识别系统之前，您需要构建一个包含已知 SKU 多模态特征的原型知识库。这通常通过 `build_prototype_library.py` 脚本完成（如果该脚本尚未提供，您需要根据 `app.py` 中 `load_prototype_library` 的逻辑自行实现）。

该脚本会遍历您的已知 SKU 样本，提取它们的视觉特征和 OCR 文本特征，并将这些多模态特征存储在 `sku_library_multimodal.pkl` 文件中。

### 4. 运行多模态 SKU 识别系统

启动主 Streamlit 应用：

```bash
streamlit run app.py
```

在浏览器中打开 Streamlit 应用界面，上传您的货架图片，系统将自动检测商品并进行多模态识别。

## 使用的模型

*   **目标检测**: YOLOv8n (预训练模型)
*   **视觉特征提取**: ResNet50 (预训练并经过微调)
*   **文本识别 (OCR)**: EasyOCR (`ch_sim`, `en` 语言模型)
*   **文本嵌入**: Sentence Transformers (`paraphrase-multilingual-MiniLM-L12-v2`)

## 配置

主要配置项位于 `app.py` 文件的顶部：

```python
# --- 配置 ---
LIBRARY_FILE = Path('/Users/guohongbin/projects/识别/sku_library_multimodal.pkl')
FINETUNED_VISUAL_MODEL_PATH = Path('/Users/guohongbin/projects/识别/finetuned_resnet50_aug.pt')
YOLO_MODEL_PATH = 'yolov8n.pt'
TEXT_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
# --- (结束) ---
```
您可以根据实际部署路径修改这些配置。