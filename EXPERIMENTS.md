# Paper Experiment Guide

This guide defines the experiment order for turning the current codebase into
paper-grade evidence. Keep generated checkpoints, logs, and metrics out of Git;
the repository tracks code and documentation only.

## 1. Train Models

Set paths first:

```powershell
$MARS_ROOT = "G:\行人重识别\时序建模\Video-Person-ReID-master\data\mars"
$MARS_TRAIN = "$MARS_ROOT\bbox_train"
```

Mean pooling baseline:

```powershell
python quality_aware_reid\train_baseline.py `
  --data-root $MARS_TRAIN `
  --model-type mean_pooling `
  --save-dir quality_aware_reid\models\mars_mean_pooling
```

Semantic attention baseline:

```powershell
python quality_aware_reid\train_baseline.py `
  --data-root $MARS_TRAIN `
  --model-type semantic_attention `
  --save-dir quality_aware_reid\models\mars_semantic_attention
```

Quality-aware default model:

```powershell
python quality_aware_reid\train.py `
  --data-root $MARS_TRAIN `
  --save-dir quality_aware_reid\models\mars_quality_aware
```

Quality-aware model with degradation ranking supervision:

```powershell
python quality_aware_reid\train.py `
  --data-root $MARS_TRAIN `
  --save-dir quality_aware_reid\models\mars_quality_aware_rank `
  --quality-rank-weight 0.1 `
  --degradation-mode mixed `
  --degradation-severity 0.5
```

## 2. Official MARS Evaluation

Run clean official-protocol evaluation for each trained checkpoint:

```powershell
python quality_aware_reid\eval_mars_official.py `
  --data-root $MARS_ROOT `
  --checkpoint quality_aware_reid\models\mars_quality_aware\best_model.pth `
  --model-type quality_aware `
  --output-json quality_aware_reid\models\mars_quality_aware\metrics_clean.json
```

Required paper metrics:

- Rank-1
- Rank-5
- Rank-10
- Rank-20
- mAP

## 3. Degraded Robustness Evaluation

Run stress tests under the same official query/gallery protocol:

```powershell
python quality_aware_reid\eval_mars_degraded.py `
  --data-root $MARS_ROOT `
  --checkpoint quality_aware_reid\models\mars_quality_aware\best_model.pth `
  --model-type quality_aware `
  --degradation mixed `
  --severity 0.5 `
  --output-json quality_aware_reid\models\mars_quality_aware\metrics_mixed_05.json
```

Recommended degradation settings:

- `blur`, severity `0.5`
- `brightness`, severity `0.5`
- `occlusion`, severity `0.5`
- `mixed`, severity `0.5`

## 4. Quality Fusion Ablations

Evaluate the same checkpoint with quality bias disabled:

```powershell
python quality_aware_reid\eval_mars_official.py `
  --data-root $MARS_ROOT `
  --checkpoint quality_aware_reid\models\mars_quality_aware\best_model.pth `
  --model-type quality_aware `
  --disable-quality-bias `
  --output-json quality_aware_reid\models\mars_quality_aware\metrics_no_bias.json
```

Evaluate multiplicative fusion:

```powershell
python quality_aware_reid\eval_mars_official.py `
  --data-root $MARS_ROOT `
  --checkpoint quality_aware_reid\models\mars_quality_aware\best_model.pth `
  --model-type quality_aware `
  --fusion-mode multiplicative `
  --output-json quality_aware_reid\models\mars_quality_aware\metrics_multiplicative.json
```

## 5. Complexity and Speed

Report parameter count and inference speed:

```powershell
python quality_aware_reid\model_complexity.py `
  --model-type quality_aware `
  --batch-size 8 `
  --seq-len 8 `
  --output-json quality_aware_reid\models\mars_quality_aware\complexity.json
```

## 6. Explainability Export

Export quality scores and attention weights:

```powershell
python quality_aware_reid\visualize_quality.py `
  --bbox-root "$MARS_ROOT\bbox_test" `
  --checkpoint quality_aware_reid\models\mars_quality_aware\best_model.pth `
  --output-json quality_aware_reid\models\mars_quality_aware\quality_attention_samples.json
```

Use the exported JSON to create figure panels showing:

- original sampled frames
- frame-level quality scores
- temporal attention weights
- cases where low-quality frames are down-weighted

## 7. Result Table Summarization

Summarize metrics into CSV and Markdown:

```powershell
python quality_aware_reid\summarize_metrics.py `
  "quality_aware_reid\models\**\metrics*.json" `
  --csv quality_aware_reid\models\summary.csv `
  --markdown quality_aware_reid\models\summary.md
```

## Paper-Ready Minimum

Before writing the main result section, collect:

- Clean official MARS results for all baselines and quality-aware variants.
- Degraded robustness results for semantic attention and quality-aware variants.
- Complexity/speed reports for all compared models.
- At least 4 qualitative examples from quality/attention export.
- Ablation table for fusion mode, quality bias, and degradation ranking.
