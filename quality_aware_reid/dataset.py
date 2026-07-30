"""
MARS training dataset for quality-aware video ReID.

This loader reads MARS bbox_train folders directly. It is intentionally kept
separate from the official-protocol evaluator, which uses MARS metadata under
info/ for query/gallery testing.
"""
import os
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from collections import defaultdict
from tqdm import tqdm


def get_transforms(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(10),
            transforms.RandomCrop((256, 128)),
            transforms.ColorJitter(brightness=0.25, contrast=0.25,
                                   saturation=0.25, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


class MARSDataset(Dataset):
    def __init__(self, root, seq_len=8, is_train=True):
        self.root = os.path.normpath(root)
        self.seq_len = seq_len
        self.is_train = is_train
        self.transform = get_transforms(is_train)

        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"MARS bbox directory does not exist: {self.root}")
        if not os.listdir(self.root):
            raise ValueError(f"MARS bbox directory is empty: {self.root}")

        raw_samples, _ = self._load_tracklets()
        self.pid2label = {pid: i for i, pid in
                          enumerate(sorted(set(s[2] for s in raw_samples)))}
        self.samples = [(d, f, self.pid2label[p]) for d, f, p in raw_samples]
        self.samples = [s for s in self.samples if len(s[1]) > 0]
        print(f"[MARS] {'train' if is_train else 'test'}: "
              f"{len(self.samples)} tracklets | {len(self.pid2label)} identities")

    def _load_tracklets(self):
        samples, bad = [], []
        pid_folders = [p for p in os.listdir(self.root)
                       if os.path.isdir(os.path.join(self.root, p))]
        for pid in tqdm(pid_folders, desc="加载MARS", ncols=80):
            pid_dir = os.path.join(self.root, pid)
            tracklet_frames = defaultdict(list)
            try:
                pid_int = int(pid)
            except ValueError:
                continue
            for f in os.listdir(pid_dir):
                if not f.lower().endswith('.jpg'):
                    continue
                try:
                    tid = f[6:11]
                except Exception:
                    bad.append(f)
                    continue
                tracklet_frames[tid].append(f)
            for tid, frames in tracklet_frames.items():
                if frames:
                    frames.sort()
                    samples.append((pid_dir, frames, pid_int))
        return samples, bad

    def _sample_frames(self, frames):
        """Random temporal sampling for train, deterministic uniform sampling for eval."""
        if len(frames) >= self.seq_len:
            if self.is_train:
                return sorted(random.sample(frames, self.seq_len))
            indices = torch.linspace(0, len(frames) - 1, self.seq_len).round().long().tolist()
            return [frames[i] for i in indices]
        return frames + [frames[-1]] * (self.seq_len - len(frames))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pid_dir, frames, label = self.samples[idx]
        select = self._sample_frames(frames)

        tensors = []
        for f in select:
            try:
                with open(os.path.join(pid_dir, f), 'rb') as fobj:
                    img = Image.open(fobj).convert('RGB')
                tensors.append(self.transform(img))
            except Exception:
                tensors.append(torch.zeros(3, 256, 128))

        return torch.stack(tensors[:self.seq_len]), label
