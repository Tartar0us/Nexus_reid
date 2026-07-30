# 视频行人重识别项目

本目录保存行人重识别方向的代码、实验结果和后续论文工作。

## 目录说明

```text
person_reid_project/
├── 实验结果_Step3_20260223_102537/
│   ├── code/        # Step3 历史训练/评估脚本
│   ├── models/      # Step3 历史权重
│   └── documents/   # Step1/Step3 实验文档
└── quality_aware_reid/
    ├── model.py     # 质量感知时序建模候选方法
    ├── baseline_model.py  # mean pooling / semantic attention baselines
    ├── dataset.py   # 当前 MARS bbox_train 加载器
    ├── losses.py
    ├── train.py
    ├── train_baseline.py
    ├── eval_mars_official.py
    ├── eval_mars_degraded.py
    ├── visualize_quality.py
    ├── summarize_metrics.py
    └── run_ablation_eval.ps1
```

## 当前判断

`实验结果_Step3_20260223_102537` 是历史探索结果，主要贡献来自数据增强和训练策略优化。它可以作为 baseline 参考，但当前评估协议不是 MARS 官方 query/gallery 协议，不能直接作为论文主结果。

`quality_aware_reid` 是更适合作为论文主线的方向：显式估计帧质量，并把质量分数注入时序注意力，目标是提升视频 ReID 在模糊、遮挡、低质量帧下的鲁棒性。

## 近期优先级

1. 用 MARS 官方协议 evaluator 复测所有结果。
2. 将 Step3 历史模型接入同一 evaluator。
3. 用同一 evaluator 复测 Step3 baseline。
4. 做质量感知模块消融实验。
5. 做低质量退化鲁棒性实验和可视化。

## 训练入口

`quality_aware_reid/train.py` 已改为命令行参数入口，会在 `--save-dir`
下保存 `config.json`、`train_history.json`、`metrics.json` 和 checkpoint。

示例：

```bash
python quality_aware_reid/train.py ^
  --data-root "G:\行人重识别\时序建模\Video-Person-ReID-master\data\mars\bbox_train" ^
  --save-dir quality_aware_reid\models\mars_quality_aware ^
  --num-classes 625 ^
  --seq-len 8 ^
  --batch-size 16 ^
  --epochs 60
```

训练集现在使用随机时序采样，同一 tracklet 在不同 epoch 会看到不同帧组合；
评估侧保持均匀采样，保证结果可复现。

训练 mean pooling 或 semantic attention baseline：

```bash
python quality_aware_reid/train_baseline.py ^
  --data-root "G:\行人重识别\时序建模\Video-Person-ReID-master\data\mars\bbox_train" ^
  --model-type semantic_attention ^
  --save-dir quality_aware_reid\models\mars_semantic_attention ^
  --num-classes 625 ^
  --seq-len 8 ^
  --batch-size 16 ^
  --epochs 60
```

## 官方协议评估入口

已新增 `quality_aware_reid/eval_mars_official.py`，用于按 MARS 官方 metadata 评估 `quality_aware_reid` 模型。

示例：

```bash
python quality_aware_reid/eval_mars_official.py ^
  --data-root "G:\行人重识别\时序建模\Video-Person-ReID-master\data\mars" ^
  --checkpoint quality_aware_reid\models\best_model.pth ^
  --model-type quality_aware ^
  --num-classes 625 ^
  --seq-len 8
```

它需要 `data_root` 下存在：

```text
bbox_test/
info/test_name.txt
info/tracks_test_info.mat
info/query_IDX.mat
```

评估 Step3 风格 semantic attention baseline：

```bash
python quality_aware_reid/eval_mars_official.py ^
  --data-root "G:\行人重识别\时序建模\Video-Person-ReID-master\data\mars" ^
  --checkpoint "实验结果_Step3_20260223_102537\models\final_model_step3.pth" ^
  --model-type semantic_attention ^
  --num-classes 625 ^
  --seq-len 8
```

`--model-type` 也支持 `mean_pooling`，用于最基础的时序聚合消融。

## 鲁棒性与可解释性入口

低质量退化 stress test：

```bash
python quality_aware_reid/eval_mars_degraded.py ^
  --data-root "G:\行人重识别\时序建模\Video-Person-ReID-master\data\mars" ^
  --checkpoint quality_aware_reid\models\mars_quality_aware\best_model.pth ^
  --model-type quality_aware ^
  --degradation mixed ^
  --severity 0.5 ^
  --output-json quality_aware_reid\models\mars_quality_aware\metrics_degraded_mixed_05.json
```

导出质量分数和注意力权重：

```bash
python quality_aware_reid/visualize_quality.py ^
  --bbox-root "G:\行人重识别\时序建模\Video-Person-ReID-master\data\mars\bbox_test" ^
  --checkpoint quality_aware_reid\models\mars_quality_aware\best_model.pth ^
  --output-json quality_aware_reid\models\mars_quality_aware\quality_attention_samples.json
```

批量跑 clean/degraded 消融评估并汇总表格：

```powershell
powershell -ExecutionPolicy Bypass -File quality_aware_reid\run_ablation_eval.ps1 `
  -DataRoot "G:\行人重识别\时序建模\Video-Person-ReID-master\data\mars" `
  -MeanCheckpoint quality_aware_reid\models\mars_mean_pooling\best_model.pth `
  -SemanticCheckpoint quality_aware_reid\models\mars_semantic_attention\best_model.pth `
  -QualityCheckpoint quality_aware_reid\models\mars_quality_aware\best_model.pth `
  -OutputDir quality_aware_reid\models\ablation_metrics
```

详细路线见根目录 `PROJECT_SEPARATION_AND_REID_ROADMAP.md`。
论文推进清单见 `TODO_PAPER.md`。
