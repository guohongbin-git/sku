import pandas as pd
import yaml
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
from PIL import Image

# --- 配置 ---
# 输入文件
ANNOTATIONS_CSV = Path('bbox_labels.csv')
# 输出目录
DATASET_ROOT = Path('yolo_dataset_sanitized') # 使用一个全新的、不带中文的目录
# --- (结束) ---

def convert_bbox_to_yolo(img_width, img_height, bbox):
    """
    将 (xmin, ymin, xmax, ymax) 格式的边界框转换为YOLO格式。
    YOLO格式: (x_center_norm, y_center_norm, width_norm, height_norm)
    """
    xmin, ymin, xmax, ymax = bbox
    
    dw = 1.0 / img_width
    dh = 1.0 / img_height
    
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    width = xmax - xmin
    height = ymax - ymin
    
    x_center_norm = x_center * dw
    y_center_norm = y_center * dh
    width_norm = width * dw
    height_norm = height * dw
    
    return x_center_norm, y_center_norm, width_norm, height_norm

def main():
    """
    主函数，执行整个转换流程。
    """
    print("--- 开始将标注数据转换为YOLO格式 ---")

    # 1. 读取并验证标注文件
    if not ANNOTATIONS_CSV.exists():
        print(f"错误: 标注文件 {ANNOTATIONS_CSV} 不存在。")
        return
        
    df = pd.read_csv(ANNOTATIONS_CSV)
    if df.empty:
        print("错误: 标注文件为空。")
        return

    # 2. 准备目录结构
    print(f"正在创建新的数据集目录: {DATASET_ROOT}")
    # if DATASET_ROOT.exists():
    #     shutil.rmtree(DATASET_ROOT) # 注释掉此行以避免操作原始的、有问题的目录
    
    # 创建所有需要的子目录
    train_img_dir = DATASET_ROOT / 'images' / 'train'
    val_img_dir = DATASET_ROOT / 'images' / 'val'
    train_label_dir = DATASET_ROOT / 'labels' / 'train'
    val_label_dir = DATASET_ROOT / 'labels' / 'val'
    
    for p in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
        p.mkdir(parents=True, exist_ok=True)

    # 3. 创建类别映射
    class_names = sorted(df['label'].unique())
    class_to_id = {name: i for i, name in enumerate(class_names)}
    print(f"发现 {len(class_names)} 个类别: {class_names}")

    # 4. 划分训练集和验证集
    # 以图片为单位进行划分
    image_paths = df['image_path'].unique()
    train_paths, val_paths = train_test_split(image_paths, test_size=0.2, random_state=42)
    print(f"数据集划分: {len(train_paths)} 张训练图片, {len(val_paths)} 张验证图片。")

    # 5. 处理并保存数据 (包含文件名清理逻辑)
    file_counter = 0
    def process_split(paths, img_dir, label_dir, split_name):
        nonlocal file_counter
        print(f"\n正在处理 {split_name} 集...")
        for img_path_str in paths:
            img_path = Path(img_path_str)
            
            # --- 文件名清理 ---
            # 创建一个纯ASCII、唯一的、从0开始递增的文件名
            new_stem = f"img_{file_counter:05d}"
            file_counter += 1
            new_img_path = img_dir / f"{new_stem}{img_path.suffix}"
            new_label_path = label_dir / f"{new_stem}.txt"
            # --- 清理结束 ---

            # 复制并重命名图片文件
            shutil.copy(img_path, new_img_path)
            
            # 获取图片尺寸
            with Image.open(img_path) as img:
                img_width, img_height = img.size
            
            # 获取该图片的所有标注
            annotations = df[df['image_path'] == img_path_str]
            
            # 使用新的、清理过的文件名创建对应的标签文件
            with open(new_label_path, 'w', encoding='utf-8') as f:
                for _, row in annotations.iterrows():
                    class_id = class_to_id[row['label']]
                    bbox = (row['xmin'], row['ymin'], row['xmax'], row['ymax'])
                    yolo_bbox = convert_bbox_to_yolo(img_width, img_height, bbox)
                    
                    f.write(f"{class_id} {' '.join(map(str, yolo_bbox))}\n")
        print(f"成功处理 {len(paths)} 张图片及其标注。")

    process_split(train_paths, train_img_dir, train_label_dir, "训练")
    process_split(val_paths, val_img_dir, val_label_dir, "验证")

    # 6. 创建 data.yaml 文件
    # 路径应该是相对于 data.yaml 文件本身
    yaml_data = {
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(class_names),
        'names': class_names
    }
    
    yaml_path = DATASET_ROOT / 'data.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    print(f"\n已生成YOLO配置文件: {yaml_path}")

    print("\n--- 数据集准备完成！ ---")
    print("现在可以开始训练YOLO模型了。")

if __name__ == '__main__':
    main()
