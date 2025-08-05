from ultralytics import YOLO

def main():
    # 加载一个预训练的YOLOv8n模型
    # 'n' 代表 nano，是YOLOv8系列中最小、最快的模型
    model = YOLO('yolov8n.pt')

    # 使用我们准备好的数据集对模型进行微调
    # 训练结果，包括模型权重和日志，将保存在 'runs/train' 目录下
    results = model.train(
        data='/Users/guohongbin/projects/识别/yolo_dataset/data.yaml',
        epochs=50,          # 训练50个周期
        imgsz=640,          # 输入图像大小为640x640
        project='runs/train', # 将训练结果保存在 'runs/train' 目录下
        name='oriental_leaf_exp1' # 本次实验的名称
    )

    print("--- 训练完成 ---")
    print("模型和训练结果已保存在 'runs/train/oriental_leaf_exp1' 目录下。")
    print("效果最好的模型权重是 'best.pt'。")

if __name__ == '__main__':
    main()
