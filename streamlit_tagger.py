import streamlit as st
import easyocr
import csv
from pathlib import Path
from PIL import Image
import re

# --- 配置 ---
SOURCE_DIR = Path('/Users/guohongbin/projects/识别/饮料-700张')
OUTPUT_CSV = Path('/Users/guohongbin/projects/识别/labels.csv')
# --- (结束) ---

# 使用 Streamlit 的缓存功能加载模型，避免每次交互都重新加载
@st.cache_resource
def load_ocr_model():
    """加载 EasyOCR 模型并缓存"""
    print("正在加载 EasyOCR 模型...")
    reader = easyocr.Reader(['ch_sim', 'en'])
    print("模型加载完成。")
    return reader

def get_best_guess(ocr_texts):
    """从OCR结果中猜测一个最可能的标签 (与之前脚本逻辑相同)"""
    if not ocr_texts:
        return ""
    ignore_keywords = ['净含量', 'ml', '毫升', 'L', '升', '公司', '地址', '电话', '官网', '营养成分表', '客服热线']
    candidates = [
        text.strip() for text in ocr_texts 
        if not any(keyword in text for keyword in ignore_keywords) 
        and not re.match(r'^[\d\s\.\%]+$', text.strip())
    ]
    return max(candidates, key=len) if candidates else ocr_texts[0]

def save_label(image_path, label):
    """将标签追加到CSV文件"""
    # 确保文件存在并有表头
    if not OUTPUT_CSV.exists():
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['image_path', 'label'])
            
    with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([str(image_path), label])

# --- 主应用 ---
st.set_page_config(layout="wide")
st.title("图像智能标注工具 (EasyOCR + Streamlit)")

# 加载模型
reader = load_ocr_model()

# 加载图片列表并过滤已处理的
all_images = sorted([p.relative_to(Path.cwd()) for p in SOURCE_DIR.glob('**/*.jpg')])
processed_images = set()
if OUTPUT_CSV.exists():
    with open(OUTPUT_CSV, 'r', encoding='utf-8') as f:
        reader_csv = csv.reader(f)
        next(reader_csv, None)
        processed_images = {row[0] for row in reader_csv if row}

unprocessed_images = [p for p in all_images if str(p) not in processed_images]

# 初始化 session_state
if 'image_index' not in st.session_state:
    st.session_state.image_index = 0

# --- UI 布局 ---
if not unprocessed_images or st.session_state.image_index >= len(unprocessed_images):
    st.success("所有图片都已标注完成！🎉")
    st.balloons()
else:
    current_image_path = unprocessed_images[st.session_state.image_index]
    
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"#### 进度: {len(processed_images) + 1} / {len(all_images)}")
        st.image(str(current_image_path), use_column_width=True)

    with col2:
        st.write("#### 标注操作")
        
        try:
            # OCR 处理
            result = reader.readtext(str(current_image_path))
            ocr_texts = [item[1] for item in result]
            
            with st.expander("查看 OCR 识别结果"):
                if ocr_texts:
                    st.json(ocr_texts)
                else:
                    st.warning("未能识别出任何文本。")

            # 获取推荐标签
            best_guess = get_best_guess(ocr_texts) if ocr_texts else ""
            
            # 用户输入
            label = st.text_input("请输入或确认标签:", value=best_guess, key=f"label_input_{st.session_state.image_index}")

            # 操作按钮
            c1, c2, _ = st.columns([1, 1, 4])
            save_button = c1.button("保存并下一张", use_container_width=True, type="primary")
            skip_button = c2.button("跳过", use_container_width=True)

            if save_button:
                if label:
                    save_label(current_image_path, label)
                    st.session_state.image_index += 1
                    st.rerun()
                else:
                    st.error("标签不能为空！")

            if skip_button:
                st.session_state.image_index += 1
                st.rerun()

        except Exception as e:
            st.error(f"处理图片时发生错误: {e}")
            if st.button("跳过这张错误的图片"):
                st.session_state.image_index += 1
                st.rerun()
