import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from pathlib import Path
import random
from tqdm import tqdm

# --- 配置 ---
ANNOTATIONS_CSV = Path('/Users/guohongbin/projects/识别/bbox_labels.csv')
OUTPUT_MODEL_PATH = Path('/Users/guohongbin/projects/识别/finetuned_resnet50_aug.pt') # 新模型文件名
# 训练参数
NUM_EPOCHS = 50 # 增加训练周期以适应更复杂的数据
BATCH_SIZE = 8
LEARNING_RATE = 0.0001
MARGIN = 0.2
# --- (结束) ---

class TripletSKUDataset(Dataset):
    def __init__(self, annotations_df, transform=None):
        self.annotations_df = annotations_df
        self.transform = transform
        self.labels = list(self.annotations_df['label'].unique())
        self.grouped = self.annotations_df.groupby('label')
        self.label_to_indices = {label: group.index.tolist() for label, group in self.grouped}

    def __len__(self):
        return len(self.annotations_df)

    def __getitem__(self, index):
        anchor_row = self.annotations_df.iloc[index]
        anchor_label = anchor_row['label']
        
        positive_indices = self.label_to_indices[anchor_label]
        positive_index = index
        while positive_index == index:
            positive_index = random.choice(positive_indices)
        positive_row = self.annotations_df.iloc[positive_index]

        negative_label = random.choice([l for l in self.labels if l != anchor_label])
        negative_index = random.choice(self.label_to_indices[negative_label])
        negative_row = self.annotations_df.iloc[negative_index]

        anchor_img = self._load_and_crop(anchor_row)
        positive_img = self._load_and_crop(positive_row)
        negative_img = self._load_and_crop(negative_row)

        # 对所有三张图片应用相同的变换
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img

    def _load_and_crop(self, row):
        img_path = Path(row['image_path'])
        image = Image.open(img_path).convert("RGB")
        bbox = (row['xmin'], row['ymin'], row['xmax'], row['ymax'])
        return image.crop(bbox)

class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=2048):
        super(EmbeddingNet, self).__init__()
        weights = ResNet50_Weights.DEFAULT
        resnet = resnet50(weights=weights)
        self.features = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x

def main():
    print("--- 开始使用【图像增强】来微调特征提取器 ---")
    
    df = pd.read_csv(ANNOTATIONS_CSV)
    label_counts = df['label'].value_counts()
    problematic_labels = label_counts[label_counts < 2]
    if not problematic_labels.empty:
        print("\n【错误】训练无法开始！请为以下类别添加更多标注（每个至少2个）:")
        print(problematic_labels)
        return

    # --- **新增：定义带图像增强的训练变换** ---
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)), # 确保输入尺寸一致
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # --- **变换定义结束** ---

    dataset = TripletSKUDataset(df, transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"将使用 {device} 进行训练。")

    model = EmbeddingNet().to(device)
    loss_fn = nn.TripletMarginLoss(margin=MARGIN, p=2)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"数据校验通过，开始训练 {NUM_EPOCHS} 个周期...")
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for anchor, positive, negative in progress_bar:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            
            optimizer.zero_grad()
            
            anchor_embedding = model(anchor)
            positive_embedding = model(positive)
            negative_embedding = model(negative)
            
            loss = loss_fn(anchor_embedding, positive_embedding, negative_embedding)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, 平均损失: {epoch_loss:.4f}")

    torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
    print("\n--- 训练完成！ ---")
    print(f"已将【带图像增强】的专家模型保存至: {OUTPUT_MODEL_PATH}")

if __name__ == '__main__':
    main()
