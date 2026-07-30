# Version History and Project Progress

This file is the project ledger. Every future version update must add a new
entry here before committing and pushing to GitHub.

## Current Project Status

The ReID work has been separated from the medical-imaging materials into an
independent project:

```text
person_reid_project/
├── quality_aware_reid/                 # active paper-code direction
├── 实验结果_Step3_20260223_102537/      # historical Step3 snapshot
├── README.md
├── TODO_PAPER.md
└── VERSION_HISTORY.md
```

Current technical maturity:

- The project is now an independent Git repository and is connected to GitHub.
- Large model weights, datasets, caches, and local outputs are excluded by `.gitignore`.
- The active paper direction is quality-guided temporal aggregation for video person ReID.
- The codebase already supports clean MARS official-protocol evaluation, degraded robustness testing, metric summarization, and quality/attention export.
- The main missing piece is not code skeleton anymore; it is running rigorous experiments and turning them into paper-grade tables and figures.

## Paper Direction

Working title:

**Quality-Guided Temporal Aggregation for Robust Video Person Re-Identification**

Core idea:

1. Estimate frame quality inside each video tracklet.
2. Inject quality scores into temporal attention so low-quality frames receive lower aggregation weight.
3. Validate robustness under blur, low brightness, occlusion, and mixed degradation.

## Version Log

### 2026-07-30 - `7955c2c` - `feat: add degradation ranking supervision`

Purpose:

- Make the frame quality estimator less like a free-floating MLP and more like a supervised quality module.

Changes:

- Added `quality_aware_reid/degradation.py`.
- Added tensor-level synthetic degradation for normalized video tensors:
  - blur
  - brightness reduction
  - occlusion
  - mixed degradation
- Added `QualityRankingLoss` in `quality_aware_reid/losses.py`.
- Updated `quality_aware_reid/train.py` with optional ranking supervision:
  - `--quality-rank-weight`
  - `--degradation-mode`
  - `--degradation-severity`
- Ranking loss encourages clean videos to have higher average quality scores than degraded videos.
- Fixed a stale `EPOCHS` variable in `train.py`.
- Updated README and paper TODO notes.

Validation:

- All Python files under `quality_aware_reid/` compiled successfully.
- `python quality_aware_reid/train.py --help` showed the new arguments correctly.

Status:

- Local commit created.
- Pushed to GitHub `origin/main`.

### 2026-07-30 - `b0b6d21` - `chore: initialize person reid project`

Purpose:

- Turn the separated ReID folder into a reproducible paper-code project with Git version control.

Changes:

- Initialized `person_reid_project` as a standalone Git repository.
- Added `.gitignore` to exclude:
  - model checkpoints
  - datasets
  - cache files
  - generated experiment outputs
- Preserved historical Step3 code and documents as an experiment snapshot.
- Added or organized the active `quality_aware_reid` framework:
  - `model.py`
  - `dataset.py`
  - `losses.py`
  - `train.py`
  - `baseline_model.py`
  - `train_baseline.py`
  - `eval_mars_official.py`
  - `eval_mars_degraded.py`
  - `visualize_quality.py`
  - `summarize_metrics.py`
  - `run_ablation_eval.ps1`
- Implemented MARS official query/gallery evaluation support.
- Added support for mean pooling and semantic attention baselines.
- Added degraded stress-test evaluation.
- Added JSON metric export and CSV/Markdown summarization.
- Added quality score and attention weight export for later visualization.

Validation:

- Python syntax checks passed for the active project files.
- Baseline/evaluation/help entry points were checked.

Status:

- Local commit created.
- Pushed to GitHub `origin/main`.

## Current Experiment Readiness

Ready to run:

- Train quality-aware model:
  - `quality_aware_reid/train.py`
- Train baselines:
  - `quality_aware_reid/train_baseline.py`
- Evaluate clean MARS official protocol:
  - `quality_aware_reid/eval_mars_official.py`
- Evaluate degraded robustness:
  - `quality_aware_reid/eval_mars_degraded.py`
- Export quality/attention records:
  - `quality_aware_reid/visualize_quality.py`
- Summarize results:
  - `quality_aware_reid/summarize_metrics.py`
- Batch clean/degraded ablation evaluation:
  - `quality_aware_reid/run_ablation_eval.ps1`

## Remaining Work Before Paper Submission

High priority:

- Run official MARS evaluation for existing Step3/semantic-attention checkpoint.
- Train and evaluate:
  - mean pooling baseline
  - semantic attention baseline
  - quality-aware baseline
  - quality-aware + degradation ranking supervision
- Run degraded robustness experiments under the same protocol.
- Generate quality-score and attention-weight visualizations.
- Build final ablation tables from JSON metrics.

Medium priority:

- Add more ablation switches:
  - no quality regularization
  - quality attention without ranking loss
  - additive log-bias vs multiplicative fusion
  - `seq_len=4/8/16`
- Add parameter count/FLOPs/inference-time reporting.
- Add a second dataset or cross-dataset validation if resources allow.

## Versioning Rule

For every future code change:

1. Update this `VERSION_HISTORY.md`.
2. Run a suitable validation command.
3. Commit with a clear message.
4. Push to GitHub.
