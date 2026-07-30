import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from collections import defaultdict
from tqdm import tqdm

class MARSVideoDataset(Dataset):
    def __init__(self, root, seq_len=8):
        super().__init__()
        self.root = os.path.normpath(root)
        self.seq_len = seq_len

        # 前置路径校验
        if not os.path.exists(self.root):
            raise FileNotFoundError(f"❌ 路径不存在: {self.root}")
        if not os.listdir(self.root):
            raise ValueError(f"❌ 路径为空: {self.root}")

        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print("[INFO] 加载MARS数据集（直接解析bbox_train）...")
        # 第一步：加载所有样本，收集原始PID标签
        raw_samples, _ = self._load_raw_tracklets()
        # 第二步：提取所有唯一PID，构建标签映射（原始PID -> 0~624归一化标签）
        self.pid2label, self.label2pid = self._build_pid_mapping(raw_samples)
        # 第三步：将原始样本的PID替换为归一化标签
        self.samples = self._convert_pid_to_label(raw_samples)
        
        self.samples = [s for s in self.samples if len(s[1]) > 0]
        assert len(self.samples) > 0, "❌ Dataset is empty, check data path!"
        print(f"[INFO] 加载完成：{len(self.samples)} 个有效视频tracklet | 归一化标签范围：0~{len(self.pid2label)-1}")

    def _load_raw_tracklets(self):
        """加载原始样本，保留原始PID标签（未归一化）"""
        samples = []
        bad_files = []
        pid_folders = [p for p in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, p))]
        
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
                    # 保留原始PID（字符串转整数）
                    tracklet_frames[tracklet_id].append(f)
            
            for tid, frames in tracklet_frames.items():
                if frames:
                    frames.sort()
                    # 跳过特殊PID（如"00-1"，MARS数据集中的干扰样本）
                    try:
                        pid_int = int(pid)
                        samples.append((pid_dir, frames, pid_int))  # 原始PID：int(pid)
                    except ValueError:
                        # 跳过无法转换为整数的PID（干扰样本）
                        continue
        return samples, bad_files

    def _build_pid_mapping(self, raw_samples):
        """构建PID-标签映射：提取唯一PID，按顺序映射为0,1,2,..."""
        # 提取所有唯一的原始PID
        all_pids = list(set([s[2] for s in raw_samples]))
        all_pids.sort()  # 排序保证映射稳定
        # 构建双向映射
        pid2label = {pid: idx for idx, pid in enumerate(all_pids)}
        label2pid = {idx: pid for idx, pid in enumerate(all_pids)}
        print(f"[INFO] 构建PID映射：{len(all_pids)} 个唯一PID -> 归一化标签0~{len(all_pids)-1}")
        return pid2label, label2pid

    def _convert_pid_to_label(self, raw_samples):
        """将原始样本的PID替换为归一化标签"""
        converted_samples = []
        for pid_dir, frames, raw_pid in raw_samples:
            normalized_label = self.pid2label[raw_pid]
            converted_samples.append((pid_dir, frames, normalized_label))
        return converted_samples

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        pid_dir, frames, label = self.samples[idx]  # label已为归一化标签（0~624）
        # 补帧：固定为seq_len帧，不足则用最后一帧补齐
        if len(frames) >= self.seq_len:
            select_frames = frames[:self.seq_len]
        else:
            select_frames = frames + [frames[-1]] * (self.seq_len - len(frames))
        
        # 加载帧，失败则填充随机张量
        video_tensor = []
        for f in select_frames:
            img_path = os.path.join(pid_dir, f)
            try:
                with open(img_path, 'rb') as fobj:
                    img = Image.open(fobj).convert('RGB')
                video_tensor.append(self.transform(img))
            except:
                video_tensor.append(torch.randn(3, 256, 128))
        
        # 终极兜底：强制保证张量长度为seq_len
        if len(video_tensor) < self.seq_len:
            video_tensor += [torch.randn(3, 256, 128)] * (self.seq_len - len(video_tensor))
        video_tensor = video_tensor[:self.seq_len]
        
        return torch.stack(video_tensor), label

# 单独测试（验证标签归一化是否生效）
if __name__ == "__main__":
    ds = MARSVideoDataset(
        root=r"G:\行人重识别\时序建模\Video-Person-ReID-master\data\mars\bbox_train",
        seq_len=8
    )
    print(f"测试样本数：{len(ds)}")  # 应输出8298
    x, y = ds[0]
    print(f"测试张量形状：{x.shape}")  # 应输出torch.Size([8, 3, 256, 128])
    print(f"测试归一化标签：{y}（范围0~624）")  # 应输出0~624之间的整数