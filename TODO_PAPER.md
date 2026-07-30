# Quality-Aware Video ReID 论文推进清单

## 当前完成度

- 项目已从医学影像代码中物理分离，ReID 主目录为 `person_reid_project/`。
- 历史 Step3 已形成可参考的 attention baseline、训练权重和实验文档。
- 新主线 `quality_aware_reid` 已实现质量估计模块、质量引导时序注意力、CE + Triplet + quality regularization 训练。
- 已有 MARS 官方 query/gallery evaluator，支持从 `info/*.mat` 读取测试协议。
- 训练入口已配置化，可保存配置、训练历史和指标文件。
- 已有 mean pooling 与 semantic attention baseline 训练入口。
- 已有退化鲁棒性 evaluator，可对 blur、brightness、occlusion、mixed 退化做同协议评估。
- 已有质量分数和 attention 权重 JSON 导出工具，可支撑可解释性图表。
- 已有 metrics JSON 聚合工具和批量消融评估 PowerShell 脚本。
- 已有合成退化质量排序监督入口，可用 `--quality-rank-weight` 打开。

## 目前不能直接投稿的原因

- Step3 历史结果来自简化 query/gallery 划分，不是 MARS 官方协议，不能作为论文主结果。
- 质量分数目前主要由 MLP 学习得到，缺少清晰的伪质量监督或退化排序监督。
- 缺少已完成的严格消融结果：mean pooling、semantic attention、QA attention、QA + regularization、QA + degradation supervision。
- 只覆盖 MARS 主线，缺少第二数据集或跨数据集泛化实验。
- 已有质量分数/attention 导出入口，但还没有生成论文可用图表和案例分析。

## P0：可信实验协议

- 用 `quality_aware_reid/eval_mars_official.py` 复测当前 quality-aware checkpoint。
- Step3 历史模型已抽到独立 `baseline_model.py`，用 `--model-type semantic_attention` 接入同一个 MARS 官方 evaluator。
- 保存每次实验的 `config.json`、`metrics.json`、checkpoint、日志。
- 明确报告 Rank-1、Rank-5、Rank-10、Rank-20、mAP。

## P1：论文方法增强

- 增加 synthetic degradation：blur、occlusion、brightness、compression。
- 为 FQE 增加伪质量监督或排序损失：已实现原始视频质量高于退化视频的 ranking loss，下一步需要跑消融。
- 用 `visualize_quality.py` 输出每帧 quality score 和 attention weight，并生成论文图。
- 用 `eval_mars_degraded.py` 做低质量 stress test，验证鲁棒性而不只是整体 Rank-1。

## P2：消融矩阵

- Mean pooling baseline。
- Semantic temporal attention baseline。
- QA attention without quality regularization。
- QA attention + quality regularization。
- QA attention + degradation supervision。
- Additive log-bias vs multiplicative fusion。
- `seq_len=4/8/16`。

## P3：写作主线

暂定题目：

**Quality-Guided Temporal Aggregation for Robust Video Person Re-Identification**

核心贡献：

1. 显式估计视频 tracklet 内帧质量，建模模糊、遮挡、低信息帧对 ReID 的影响。
2. 将质量先验注入时序注意力，使聚合过程降低低质量帧权重。
3. 通过官方协议、消融实验、退化鲁棒性和可视化证明方法有效。
