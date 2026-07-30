"""
快速MARS评估 - 使用合理的采样策略
每个PID随机选择1个tracklet作为query，其余作为gallery
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from collections import defaultdict
import random


# 模型定义
class VideoReIDAttentionImproved(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )
        self.embedding = nn.Linear(2048, 512)
        self.bn = nn.BatchNorm1d(512)
        self.classifier = nn.Linear(512, num_classes)
    
    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feat = self.backbone(x)
        feat = self.gap(feat).view(B, T, 2048)
        attn = self.attention(feat)
        attn = torch.softmax(attn, dim=1)
        video_feat = (feat * attn).sum(dim=1)
        embedding = self.embedding(video_feat)
        embedding = self.bn(embedding)
        embedding_norm = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding_norm


# MARS测试集 - Query/Gallery分割
class MARSQueryGalleryDataset(Dataset):
    def __init__(self, root, seq_len=8, max_gallery_per_pid=5):
        super().__init__()
        self.root = os.path.normpath(root)
        self.seq_len = seq_len
        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"加载MARS测试集（Query/Gallery分割）")
        
        # 第一步：收集所有tracklet
        pid_tracklets = defaultdict(list)
        pid_folders = sorted([p for p in os.listdir(root) if os.path.isdir(os.path.join(root, p))])
        
        for pid_str in tqdm(pid_folders, desc="扫描PID"):
            pid_dir = os.path.join(root, pid_str)
            try:
                pid = int(pid_str)
            except:
                continue
            
            tracklet_frames = defaultdict(list)
            frames = sorted([f for f in os.listdir(pid_dir) if f.lower().endswith('.jpg')])
            
            for frame in frames:
                try:
                    camid = int(frame[5])
                    tracklet_id = frame[4:11]
                    tracklet_frames[tracklet_id].append(frame)
                except:
                    continue
            
            for tid, frames_list in sorted(tracklet_frames.items()):
                if len(frames_list) >= seq_len:
                    camid = int(tid[1])
                    pid_tracklets[pid].append((pid_dir, frames_list[:seq_len], pid, camid, tid))
        
        # 第二步：分割Query和Gallery
        self.query_samples = []
        self.gallery_samples = []
        self.query_info = []  # (pid, camid)
        self.gallery_info = []
        
        random.seed(42)  # 固定随机种子保证可复现
        
        for pid, tracklets in pid_tracklets.items():
            if len(tracklets) < 2:
                continue  # 至少需要2个tracklet（1个query + 1个gallery）
            
            # 随机选1个作为query
            query_tracklet = random.choice(tracklets)
            self.query_samples.append(query_tracklet[:3])  # (pid_dir, frames, pid)
            self.query_info.append((query_tracklet[2], query_tracklet[3]))  # (pid, camid)
            
            # 其余作为gallery（最多max_gallery_per_pid个）
            gallery_tracklets = [t for t in tracklets if t != query_tracklet]
            if len(gallery_tracklets) > max_gallery_per_pid:
                gallery_tracklets = random.sample(gallery_tracklets, max_gallery_per_pid)
            
            for g_tracklet in gallery_tracklets:
                self.gallery_samples.append(g_tracklet[:3])
                self.gallery_info.append((g_tracklet[2], g_tracklet[3]))
        
        print(f"Query集: {len(self.query_samples)} 个tracklet")
        print(f"Gallery集: {len(self.gallery_samples)} 个tracklet")
        print(f"唯一PID数: {len(pid_tracklets)}")
    
    def get_query_loader(self, batch_size):
        return DataLoader(
            QueryDataset(self.query_samples, self.seq_len, self.transform),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
    
    def get_gallery_loader(self, batch_size):
        return DataLoader(
            GalleryDataset(self.gallery_samples, self.seq_len, self.transform),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )


class QueryDataset(Dataset):
    def __init__(self, samples, seq_len, transform):
        self.samples = samples
        self.seq_len = seq_len
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        pid_dir, frames, pid = self.samples[idx]
        video_tensor = []
        
        for frame in frames:
            img_path = os.path.join(pid_dir, frame)
            try:
                with open(img_path, 'rb') as f:
                    img = Image.open(f).convert('RGB')
                video_tensor.append(self.transform(img))
            except:
                if video_tensor:
                    video_tensor.append(video_tensor[-1])
                else:
                    video_tensor.append(torch.randn(3, 256, 128))
        
        while len(video_tensor) < self.seq_len:
            video_tensor.append(video_tensor[-1] if video_tensor else torch.randn(3, 256, 128))
        
        return torch.stack(video_tensor[:self.seq_len]), pid, idx


class GalleryDataset(Dataset):
    def __init__(self, samples, seq_len, transform):
        self.samples = samples
        self.seq_len = seq_len
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        pid_dir, frames, pid = self.samples[idx]
        video_tensor = []
        
        for frame in frames:
            img_path = os.path.join(pid_dir, frame)
            try:
                with open(img_path, 'rb') as f:
                    img = Image.open(f).convert('RGB')
                video_tensor.append(self.transform(img))
            except:
                if video_tensor:
                    video_tensor.append(video_tensor[-1])
                else:
                    video_tensor.append(torch.randn(3, 256, 128))
        
        while len(video_tensor) < self.seq_len:
            video_tensor.append(video_tensor[-1] if video_tensor else torch.randn(3, 256, 128))
        
        return torch.stack(video_tensor[:self.seq_len]), pid, idx


def extract_features(model, data_loader, device):
    model.eval()
    all_feats = []
    all_pids = []
    
    with torch.no_grad():
        for videos, pids, _ in tqdm(data_loader, desc="提取特征"):
            videos = videos.to(device, non_blocking=True)
            embeddings = model(videos)
            all_feats.append(embeddings.cpu().numpy())
            all_pids.append(pids.numpy())
    
    all_feats = np.concatenate(all_feats, axis=0)
    all_pids = np.concatenate(all_pids, axis=0)
    
    return all_feats, all_pids


def evaluate_query_gallery(query_feats, query_pids, gallery_feats, gallery_pids):
    """
    Query-Gallery评估
    """
    print("\n计算距离矩阵...")
    # query_feats: [num_query, feat_dim]
    # gallery_feats: [num_gallery, feat_dim]
    dist_matrix = np.sqrt(np.sum((query_feats[:, np.newaxis] - gallery_feats) ** 2, axis=2))
    
    num_queries = len(query_feats)
    cmc = np.zeros(len(gallery_feats))
    ap_list = []
    
    print("评估检索性能...")
    for i in tqdm(range(num_queries), desc="Query进度"):
        query_pid = query_pids[i]
        query_dist = dist_matrix[i]
        
        # 排序
        sorted_indices = np.argsort(query_dist)
        sorted_pids = gallery_pids[sorted_indices]
        
        # 计算matches
        matches = (sorted_pids == query_pid)
        
        # CMC
        if np.any(matches):
            first_match_idx = np.where(matches)[0][0]
            cmc[first_match_idx:] += 1
        
        # AP
        num_pos = np.sum(matches)
        if num_pos > 0:
            match_indices = np.where(matches)[0]
            precisions = []
            for j, match_idx in enumerate(match_indices):
                precision = (j + 1) / (match_idx + 1)
                precisions.append(precision)
            ap = np.mean(precisions)
            ap_list.append(ap)
    
    cmc = cmc / num_queries
    mAP = np.mean(ap_list) if ap_list else 0.0
    
    return cmc, mAP


def main():
    import sys
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = "final_model_step1.pth"
    
    MODEL_PATH = f"./trained_models_improved/{model_name}"
    TEST_DATA_ROOT = r"G:\行人重识别\时序建模\Video-Person-ReID-master\data\mars\bbox_test"
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    
    print(f"\n{'='*70}")
    print(f"MARS快速评估（Query/Gallery分割）")
    print(f"{'='*70}")
    print(f"设备: {DEVICE}")
    print(f"模型: {model_name}")
    print(f"{'='*70}\n")
    
    # 加载测试集
    print("加载测试集...")
    test_dataset = MARSQueryGalleryDataset(
        root=TEST_DATA_ROOT,
        seq_len=8,
        max_gallery_per_pid=5
    )
    
    query_loader = test_dataset.get_query_loader(BATCH_SIZE)
    gallery_loader = test_dataset.get_gallery_loader(BATCH_SIZE)
    
    # 加载模型
    print("\n加载模型...")
    model = VideoReIDAttentionImproved(num_classes=625).to(DEVICE)
    
    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        else:
            model.load_state_dict(state_dict)
        print("✅ 模型加载成功")
    else:
        print(f"❌ 模型文件不存在: {MODEL_PATH}")
        return
    
    # 提取Query特征
    print("\n提取Query特征...")
    query_feats, query_pids = extract_features(model, query_loader, DEVICE)
    print(f"Query特征形状: {query_feats.shape}")
    
    # 提取Gallery特征
    print("\n提取Gallery特征...")
    gallery_feats, gallery_pids = extract_features(model, gallery_loader, DEVICE)
    print(f"Gallery特征形状: {gallery_feats.shape}")
    
    # 评估
    print("\n执行Query-Gallery评估...")
    cmc, mAP = evaluate_query_gallery(query_feats, query_pids, gallery_feats, gallery_pids)
    
    # 打印结果
    print(f"\n{'='*70}")
    print(f"MARS评估结果（Query/Gallery分割）")
    print(f"{'='*70}")
    print(f"Query数: {len(query_feats)}")
    print(f"Gallery数: {len(gallery_feats)}")
    print(f"")
    print(f"Rank-1:   {cmc[0] * 100:6.2f}%")
    print(f"Rank-5:   {cmc[4] * 100:6.2f}%")
    print(f"Rank-10:  {cmc[9] * 100:6.2f}%")
    print(f"Rank-20:  {cmc[19] * 100:6.2f}%")
    print(f"mAP:      {mAP * 100:6.2f}%")
    print(f"{'='*70}\n")
    
    # 与baseline对比
    baseline_rank1 = 19.73
    baseline_mAP = 12.90
    
    print("与Baseline对比:")
    print(f"Rank-1: {baseline_rank1:.2f}% → {cmc[0]*100:.2f}% (Δ {cmc[0]*100-baseline_rank1:+.2f}%)")
    print(f"mAP:    {baseline_mAP:.2f}% → {mAP*100:.2f}% (Δ {mAP*100-baseline_mAP:+.2f}%)")
    
    if cmc[0] * 100 > baseline_rank1 + 10:
        print("\n✅ 显著提升！改进有效！")
    elif cmc[0] * 100 > baseline_rank1:
        print("\n⚠️  有提升，但不够显著")
    else:
        print("\n❌ 性能下降，需要调整")


if __name__ == "__main__":
    main()
