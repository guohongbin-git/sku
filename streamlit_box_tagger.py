import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import glob

# --- 配置 ---
# ** 现在可以扫描多个文件夹 **
IMAGE_DIRS = [
    '/Users/guohongbin/projects/识别/东方树叶',
    '/Users/guohongbin/projects/识别/可口可乐'
]
ANNOTATIONS_CSV = Path('/Users/guohongbin/projects/识别/bbox_labels.csv')
# --- (结束) ---

st.set_page_config(layout="wide", page_title="一体化物体标注工具")

# --- Session State 初始化 ---
if 'image_index' not in st.session_state:
    st.session_state.image_index = 0
if 'annotations' not in st.session_state:
    st.session_state.annotations = []
if 'current_image_annotations' not in st.session_state:
    st.session_state.current_image_annotations = []
if 'class_names' not in st.session_state:
    if ANNOTATIONS_CSV.exists():
        df = pd.read_csv(ANNOTATIONS_CSV)
        st.session_state.class_names = sorted(df['label'].unique().tolist())
    else:
        st.session_state.class_names = []

# --- 数据加载与保存函数 ---
@st.cache_data
def get_image_paths():
    """从所有指定的目录中获取图片路径"""
    all_files = []
    for d in IMAGE_DIRS:
        # 使用glob递归搜索所有jpg文件
        image_extensions = ['*.jpg', '*.jpeg', '*.png']
    all_files = []
    for d in IMAGE_DIRS:
        for ext in image_extensions:
            all_files.extend(glob.glob(str(Path(d) / '**' / ext), recursive=True))
    
    # 转换为相对路径并去重
    return sorted(list(set([Path(p).relative_to(Path.cwd()) for p in all_files])))

def save_annotations():
    if not st.session_state.current_image_annotations:
        return
    df_to_save = pd.DataFrame(st.session_state.current_image_annotations)
    if ANNOTATIONS_CSV.exists():
        df_existing = pd.read_csv(ANNOTATIONS_CSV)
        df_final = pd.concat([df_existing, df_to_save], ignore_index=True)
    else:
        df_final = df_to_save
    df_final.drop_duplicates(inplace=True)
    df_final.to_csv(ANNOTATIONS_CSV, index=False)

# --- 主应用 ---
all_images = get_image_paths()

processed_images = set()
if ANNOTATIONS_CSV.exists():
    df_processed = pd.read_csv(ANNOTATIONS_CSV)
    if not df_processed.empty:
        processed_images = set(df_processed['image_path'].unique())

unprocessed_images = [p for p in all_images if str(p) not in processed_images]

st.title("一体化物体标注工具 (画框 & 创建标签)")

if not unprocessed_images or st.session_state.image_index >= len(unprocessed_images):
    st.success("所有图片都已标注完成！🎉")
    st.balloons()
    st.stop()

current_image_path = unprocessed_images[st.session_state.image_index]
bg_image = Image.open(current_image_path)

# --- 侧边栏控制器 ---
with st.sidebar:
    st.header("标注控制器")
    st.write(f"**进度:** {len(processed_images) + 1} / {len(all_images)}")
    st.write(f"**当前图片:** `{str(current_image_path)}`")

    st.subheader("1. 选择或创建类别")
    ADD_NEW_CLASS_OPTION = "+ 添加新类别..."
    class_options = st.session_state.class_names + [ADD_NEW_CLASS_OPTION]
    selected_class = st.selectbox("选择类别", class_options, key="class_selector")

    if selected_class == ADD_NEW_CLASS_OPTION:
        new_class_name = st.text_input("输入新类别名称:")
        if st.button("确认添加类别", use_container_width=True):
            if new_class_name and new_class_name not in st.session_state.class_names:
                st.session_state.class_names.append(new_class_name)
                st.session_state.class_names.sort()
                st.session_state.selected_class_for_rerun = new_class_name
                st.rerun()
            else:
                st.warning("类别名称不能为空或重复！")
    else:
        st.session_state.selected_class_for_rerun = selected_class

    st.subheader("2. 画框并添加")
    st.write(f"当前选定标签: **{st.session_state.get('selected_class_for_rerun', '')}**")
    
    st.subheader("3. 本图已标注内容")
    st.dataframe(pd.DataFrame(st.session_state.current_image_annotations), use_container_width=True)

    undo_col, reset_col = st.columns(2)
    with undo_col:
        if st.button("撤销上一个", use_container_width=True) and st.session_state.current_image_annotations:
            st.session_state.current_image_annotations.pop()
            st.rerun()
    with reset_col:
        if st.button("重置本图", use_container_width=True, type="secondary"):
            st.session_state.current_image_annotations = []
            st.rerun()

    st.subheader("4. 导航")
    col1, col2 = st.columns(2)
    if col1.button("保存并下一张", type="primary", use_container_width=True):
        save_annotations()
        st.session_state.image_index += 1
        st.session_state.current_image_annotations = []
        st.rerun()

    if col2.button("跳过", use_container_width=True):
        st.session_state.image_index += 1
        st.session_state.current_image_annotations = []
        st.rerun()

# --- 主画布 ---
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=2,
    stroke_color="#FFA500",
    background_image=bg_image,
    update_streamlit=True,
    height=bg_image.height,
    width=bg_image.width,
    drawing_mode="rect",
    key=f"canvas_{st.session_state.image_index}",
)

if canvas_result.json_data and canvas_result.json_data["objects"]:
    if st.sidebar.button("确认添加该框", use_container_width=True):
        obj = canvas_result.json_data["objects"][-1]
        xmin, ymin = obj["left"], obj["top"]
        xmax, ymax = xmin + obj["width"], ymin + obj["height"]
        
        new_annotation = {
            "image_path": str(current_image_path),
            "label": st.session_state.get('selected_class_for_rerun', selected_class),
            "xmin": int(xmin),
            "ymin": int(ymin),
            "xmax": int(xmax),
            "ymax": int(ymax)
        }
        st.session_state.current_image_annotations.append(new_annotation)
        st.rerun()
