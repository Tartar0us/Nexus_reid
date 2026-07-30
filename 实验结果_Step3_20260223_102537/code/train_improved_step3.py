"""
实验改进 - 第三步
优化数据增强和训练策略，进一步提升泛化能力
- 增强的数据增强
- Warmup学习率
- Label Smoothing
- Mixup（可选）
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
import numpy as np

from mars_video_dataset import MARSVideoDataset


# =====================
# 增强的数据增强（在Dataset中使用）
# =====================
from torchvision import transforms

def get_augmented_transforms(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(10),
            transforms.RandomCrop((256, 128)),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


# =====================
# 增强的MARS数据集
# =====================
from torch.utils.data import Dataset
from PIL import Image
from collections import defaultdict

class MARSVideoDatasetAugmented(Dataset):
    def __init__(self, root, seq_len=8, is_train=True):
        super().__init__()
        self.root = os.path.normpath(root)
        self.seq_len = seq_len
        self.transform = get_augmented_transforms(is_train)
        
        print(f"[INFO] 加载MARS数据集（增强版）...")
        raw_samples, _ = self._load_raw_tracklets()
        self.pid2label, self.label2pid = self._build_pid_mapping(raw_samples)
        self.samples = self._convert_pid_to_label(raw_samples)
        self.samples = [s for s in self.samples if len(s[1]) > 0]
        
        print(f"[INFO] 加载完成：{len(self.samples)} 个tracklet | PID数：{len(self.pid2label)}")
    
    def _load_raw_tracklets(self):
        samples = []
        bad_files = []
        pid_folders = [p for p in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, p))]
        
        from tqdm import tqdm
        for pid in tqdm(pid_folders, desc="加载PID文件夹", ncols=80):
            pid_dir = os.path.join(self.root, pid)
            tracklet_frames = defaultdict(list)
            
            for f in os.listdir(pid_dir):
                if not f.lower().endswith(".jpg"):
                    continue
                img_path = os.path.join(pid_dir, f)
                try:
                    tracklet_id = f[6:11]
                except:
                    bad_files.append(img_path)
                    continue
                if os.path.exists(img_path):
                    tracklet_frames[tracklet_id].append(f)
            
            for tid, frames in tracklet_frames.items():
                if frames:
                    frames.sort()
                    samples.append((pid_dir, frames, int(pid)))
        
        return samples, bad_files
    
    def _build_pid_mapping(self, raw_samples):
        all_pids = sorted(list(set([s[2] for s in raw_samples])))
        pid2label = {pid: idx for idx, pid in enumerate(all_pids)}
        label2pid = {idx: pid for idx, pid in enumerate(all_pids)}
        return pid2label, label2pid
    
    def _convert_pid_to_label(self, raw_samples):
        return [(pid_dir, frames, self.pid2label[raw_pid]) 
                for pid_dir, frames, raw_pid in raw_samples]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        pid_dir, frames, label = self.samples[idx]
        
        # 采样帧
        if len(frames) >= self.seq_len:
            select_frames = frames[:self.seq_len]
        else:
            select_frames = frames + [frames[-1]] * (self.seq_len - len(frames))
        
        # 加载并增强
        video_tensor = []
        for f in select_frames:
            img_path = os.path.join(pid_dir, f)
            try:
                with open(img_path, 'rb') as fobj:
                    img = Image.open(fobj).convert('RGB')
                video_tensor.append(self.transform(img))
            except:
                video_tensor.append(torch.randn(3, 256, 128))
        
        if len(video_tensor) < self.seq_len:
            video_tensor += [torch.randn(3, 256, 128)] * (self.seq_len - len(video_tensor))
        
        return torch.stack(video_tensor[:self.seq_len]), label


# =====================
# 模型（使用Step1的架构）
# =====================
class VideoReIDAttentionImproved(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        for name, module in self.backbone.named_children():
            if name in ['0', '1', '2', '3', '4', '5', '6']:
                for param in module.parameters():
                    param.requires_grad = False
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )
        self.embedding = nn.Linear(2048, 512)
        self.bn = nn.BatchNorm1d(512)
        self.classifier = nn.Linear(512, num_classes)
    
    def forward(self, x, return_embedding=False):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feat = self.backbone(x)
        feat = self.gap(feat).view(B, T, 2048)
        attn = self.attention(feat)
        attn = torch.softmax(attn, dim=1)
        video_feat = (feat * attn).sum(dim=1)
        embedding = self.embedding(video_feat)
        embedding = self.bn(embedding)
        logits = self.classifier(embedding)
        
        if return_embedding:
            embedding_norm = torch.nn.functional.normalize(embedding, p=2, dim=1)
            return logits, embedding_norm
        else:
            return logits


# =====================
# Label Smoothing Cross Entropy
# =====================
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_class
        log_prob = torch.nn.functional.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prob).sum(dim=1).mean()
        return loss


# =====================
# Triplet Loss
# =====================
class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
    
    def forward(self, embeddings, labels):
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)
        batch_size = embeddings.size(0)
        loss = 0.0
        num_triplets = 0
        
        for i in range(batch_size):
            pos_mask = (labels == labels[i]) & (torch.arange(batch_size, device=labels.device) != i)
            neg_mask = labels != labels[i]
            
            if not pos_mask.any() or not neg_mask.any():
                continue
            
            pos_dist = dist_matrix[i][pos_mask].max()
            neg_dist = dist_matrix[i][neg_mask].min()
            triplet_loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
            loss += triplet_loss
            num_triplets += 1
        
        if num_triplets > 0:
            loss = loss / num_triplets
        
        return loss


# =====================
# Warmup学习率调度器
# =====================
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0
    
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            # Warmup阶段：线性增长
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            # Cosine退火
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1
        return lr


# =====================
# 训练函数
# =====================
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 超参数
    num_classes = 625
    seq_len = 8
    batch_size = 16
    epochs = 60
    base_lr = 3e-4
    warmup_epochs = 5
    triplet_weight = 0.5
    label_smoothing = 0.1
    save_dir = "./trained_models_improved"
    os.makedirs(save_dir, exist_ok=True)
    
    # 数据集（使用增强版）
    data_root = r"G:\行人重识别\时序建模\Video-Person-ReID-master\data\mars\bbox_train"
    print(f"\n加载训练集（增强版）...")
    train_dataset = MARSVideoDatasetAugmented(root=data_root, seq_len=seq_len, is_train=True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device == "cuda" else False,
        drop_last=True
    )
    
    # 模型
    model = VideoReIDAttentionImproved(num_classes=num_classes).to(device)
    
    # 损失函数（使用Label Smoothing）
    ce_criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
    triplet_criterion = TripletLoss(margin=0.3)
    
    # 优化器
    params = [
        {'params': model.backbone[7].parameters(), 'lr': base_lr * 0.1},
        {'params': model.attention.parameters(), 'lr': base_lr},
        {'params': model.embedding.parameters(), 'lr': base_lr},
        {'params': model.bn.parameters(), 'lr': base_lr},
        {'params': model.classifier.parameters(), 'lr': base_lr}
    ]
    optimizer = optim.AdamW(params, weight_decay=5e-4)  # 使用AdamW
    
    # 学习率调度（Warmup + Cosine）
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, epochs, base_lr)
    
    # 训练
    print("="*70)
    print("开始训练 - 第三步（增强数据增强 + 优化训练策略）")
    print("="*70)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_triplet_loss = 0.0
        
        # 更新学习率
        current_lr = scheduler.step()
        
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", ncols=120)
        
        for videos, labels in pbar:
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            logits, embeddings = model(videos, return_embedding=True)
            
            ce_loss = ce_criterion(logits, labels)
            triplet_loss = triplet_criterion(embeddings, labels)
            loss = ce_loss + triplet_weight * triplet_loss
            
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_triplet_loss += triplet_loss.item() if isinstance(triplet_loss, torch.Tensor) else triplet_loss
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'ce': f'{ce_loss.item():.4f}',
                'tri': f'{triplet_loss.item() if isinstance(triplet_loss, torch.Tensor) else triplet_loss:.4f}',
                'lr': f'{current_lr:.6f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        avg_ce_loss = total_ce_loss / len(train_loader)
        avg_triplet_loss = total_triplet_loss / len(train_loader)
        
        print(f"\nEpoch [{epoch+1}/{epochs}] 平均损失:")
        print(f"  Total: {avg_loss:.4f} | CE: {avg_ce_loss:.4f} | Triplet: {avg_triplet_loss:.4f} | LR: {current_lr:.6f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss
            }, os.path.join(save_dir, "best_model_step3.pth"))
            print(f"  ✅ 保存最优模型 (loss: {best_loss:.4f})")
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), 
                      os.path.join(save_dir, f"model_step3_epoch{epoch+1}.pth"))
        
        print()
    
    torch.save(model.state_dict(), os.path.join(save_dir, "final_model_step3.pth"))
    print("="*70)
    print("训练完成！")
    print("="*70)


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"\n❌ 训练错误: {str(e)}")
        import traceback
        traceback.print_exc()
