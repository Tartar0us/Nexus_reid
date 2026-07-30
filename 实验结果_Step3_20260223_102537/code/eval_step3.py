"""
评估Step3模型性能
使用标准Query/Gallery协议
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
from tqdm import tqdm
import os
from collections import defaultdict

from mars_video_dataset import MARSVideoDataset


# =====================
# 模型定义（与Step3相同）
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


def compute_distance_matrix(query_features, gallery_features):
    """计算欧氏距离矩阵"""
    m, n = query_features.size(0), gallery_features.size(0)
    dist_matrix = torch.cdist(query_features, gallery_features, p=2)
    return dist_matrix.cpu().numpy()


def evaluate_ranking(dist_matrix, query_pids, gallery_pids, query_camids, gallery_camids, max_rank=20):
    """评估检索性能"""
    num_query = dist_matrix.shape[0]
    
    all_cmc = []
    all_AP = []
    
    for i in range(num_query):
        q_pid = query_pids[i]
        q_camid = query_camids[i]
        
        order = np.argsort(dist_matrix[i])
        
        # 移除同一摄像头下的同一个人
        remove = (gallery_pids[order] == q_pid) & (gallery_camids[order] == q_camid)
        keep = np.invert(remove)
        
        orig_cmc = (gallery_pids[order] == q_pid)
        orig_cmc = orig_cmc[keep]
        
        if not np.any(orig_cmc):
            continue
        
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        
        all_cmc.append(cmc[:max_rank])
        
        # 计算AP
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
    
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / len(all_cmc)
    mAP = np.mean(all_AP)
    
    return all_cmc, mAP


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}\n")
    
    # 加载测试集
    test_root = r"G:\行人重识别\时序建模\Video-Person-ReID-master\data\mars\bbox_test"
    print("加载测试集...")
    test_dataset = MARSVideoDataset(root=test_root, seq_len=8)
    
    # 加载模型
    print("\n加载Step3模型...")
    model = VideoReIDAttentionImproved(num_classes=625).to(device)
    checkpoint = torch.load("./trained_models_improved/final_model_step3.pth", map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    # 提取特征
    print("\n提取测试集特征...")
    all_features = []
    all_pids = []
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    with torch.no_grad():
        for videos, labels in tqdm(test_loader, desc="提取特征", ncols=100):
            videos = videos.to(device)
            _, features = model(videos, return_embedding=True)
            all_features.append(features.cpu())
            all_pids.append(labels)
    
    all_features = torch.cat(all_features, dim=0)
    all_pids = torch.cat(all_pids, dim=0).numpy()
    
    print(f"特征维度: {all_features.shape}")
    print(f"样本数量: {len(all_pids)}")
    
    # Query/Gallery分割
    print("\n划分Query和Gallery集...")
    pid_to_indices = defaultdict(list)
    for idx, pid in enumerate(all_pids):
        pid_to_indices[pid].append(idx)
    
    query_indices = []
    gallery_indices = []
    
    for pid, indices in pid_to_indices.items():
        if len(indices) > 0:
            query_indices.append(indices[0])
            if len(indices) > 1:
                gallery_indices.extend(indices[1:6])
    
    query_features = all_features[query_indices]
    gallery_features = all_features[gallery_indices]
    query_pids = all_pids[query_indices]
    gallery_pids = all_pids[gallery_indices]
    
    # 假设camid（这里简化处理）
    query_camids = np.zeros(len(query_pids))
    gallery_camids = np.ones(len(gallery_pids))
    
    print(f"Query数量: {len(query_pids)}")
    print(f"Gallery数量: {len(gallery_pids)}")
    
    # 计算距离矩阵
    print("\n计算距离矩阵...")
    dist_matrix = compute_distance_matrix(query_features, gallery_features)
    
    # 评估
    print("\n评估检索性能...")
    cmc, mAP = evaluate_ranking(dist_matrix, query_pids, gallery_pids, 
                                 query_camids, gallery_camids, max_rank=20)
    
    # 打印结果
    print("\n" + "="*70)
    print("Step3 评估结果（标准Query/Gallery协议）")
    print("="*70)
    print(f"Rank-1:  {cmc[0]*100:.2f}%")
    print(f"Rank-5:  {cmc[4]*100:.2f}%")
    print(f"Rank-10: {cmc[9]*100:.2f}%")
    print(f"Rank-20: {cmc[19]*100:.2f}%")
    print(f"mAP:     {mAP*100:.2f}%")
    print("="*70)
    
    # 与Step1对比
    print("\n" + "="*70)
    print("性能对比（Step1 vs Step3）")
    print("="*70)
    print(f"{'指标':<10} {'Step1':<12} {'Step3':<12} {'提升':<12}")
    print("-"*70)
    
    step1_rank1 = 78.01
    step1_rank5 = 90.19
    step1_map = 57.65
    
    print(f"{'Rank-1':<10} {step1_rank1:<12.2f} {cmc[0]*100:<12.2f} {cmc[0]*100-step1_rank1:+.2f}%")
    print(f"{'Rank-5':<10} {step1_rank5:<12.2f} {cmc[4]*100:<12.2f} {cmc[4]*100-step1_rank5:+.2f}%")
    print(f"{'mAP':<10} {step1_map:<12.2f} {mAP*100:<12.2f} {mAP*100-step1_map:+.2f}%")
    print("="*70)


if __name__ == "__main__":
    main()
