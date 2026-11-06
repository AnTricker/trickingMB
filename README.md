# RTMPose + MotionBERT: 2D Pose, 3D Skeleton and Mesh Recovery

## 1. Project Overview

本專案由單視角影片產生 RTMPose 2D Keypoints，並透過 MotionBERT 建立 H36M-17 3D Skeleton 與 SMPL Mesh Visualization。**主要執行流程採用 `batch_infer.py`**；`infer_wild.py`、`infer_wild_mesh.py` 保留作為底層推論程式與單影片除錯用途。

```text
Videos (.mp4) → RTMPose run.py → AlphaPose-compatible JSON
                                    │
             MotionBERT input/raw/  │  input/alphapose/
                       └────────────┴─────────────┐
                                                  ▼
                                          batch_infer.py
                                          ├─ infer_wild.py
                                          │  ├─ X3D/*.npy (T,17,3)
                                          │  └─ X3D/*.mp4
                                          └─ infer_wild_mesh.py
                                             └─ mesh/*.mp4
```

使用 Pretrained MotionBERT Pose3D 與 Mesh Checkpoints，目前未對 Tricking 資料 Fine-tune。**Batch 意指依序處理多支影片**；各影片各自啟動推論子程序、重新載入 Checkpoint，並未將多影片合併為同一個 GPU Tensor Batch。

---

## 2. Environment
| Item | Configuration |
|---|---|
| OS | Windows 11 |
| Terminal | Anaconda Prompt / CMD |
| GPU | NVIDIA CUDA-compatible GPU |
| System CUDA Toolkit | 12.6 |
| PyTorch | 2.1.2 + CUDA 12.1 |
| Python | 3.10 |
| NumPy | 1.26.4 |
| RTMPose | RTMPose-m Body8-Halpe26 |
| MMPose | 1.3.2 |
| MotionBERT | Walter0807/MotionBERT |

CUDA Toolkit 12.6 與 PyTorch CUDA 12.1 runtime 可共存。若需要從原始碼編譯 MMCV CUDA operators，則需額外確認編譯器與 CUDA Toolkit 的相容性。

兩個階段可使用獨立的 Conda environments：

- `rtmpose`：2D Pose Estimation

- `motionbert`：3D Pose / 3D Mesh

如果既有環境已經安裝 PyTorch，請先驗證版本，不必重複下載。

## 3. Project Structure (Batch Workflow)

以下為主要操作需要的目錄；`input/raw/` 與 `input/alphapose/` 是 `batch_infer.py` 實際要求的子目錄。

```text
motionBert/
├── rtmpose/
│   ├── run.py
│   ├── convert.py
│   ├── input/                     # 原始影片（RTMPose）
│   └── output/
│       ├── predictions/           # RTMPose 預測 JSON
│       ├── alphapose/             # 格式轉換後的 JSON（依本地 run.py 設定）
│       └── visualizations/        # 2D 可視化（名稱以實際輸出為準）
└── MotionBERT/
    ├── batch_infer.py             # 批次入口
    ├── infer_wild.py              # 3D Skeleton 子程序
    ├── infer_wild_mesh.py         # SMPL Mesh 子程序
    ├── configs/
    ├── checkpoint/
    │   ├── pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin
    │   └── mesh/FT_MB_release_MB_ft_pw3d/best_epoch.bin
    ├── data/mesh/                # SMPL 與 regressors
    ├── input/
    │   ├── raw/                  # aerial.mp4, backTuck.mp4, ...
    │   └── alphapose/            # aerial_l.json, backTuck_l.json, ...
    └── output/
        ├── X3D/
        │   ├── aerial_lX3D.npy
        │   └── aerial_lX3D.mp4
        └── mesh/
            └── aerial_lMesh.mp4
```

**命名配對規則：** `aerial_l.json`、`aerial_m.json`、`aerial_x.json` 均對應 `input/raw/aerial.mp4`。副檔名支援 `.mp4`、`.mov`、`.avi`、`.mkv`。`batch_infer.py` 只掃描 `input/alphapose/*.json` 的**第一層**，不遞迴子資料夾。

---

# Part A — RTMPose Deployment
## 4. RTMPose Installation
### 4.1 Create Conda Environment
```cmd
conda create -n rtmpose python=3.10 -y
conda activate rtmpose

```

### 4.2 Install PyTorch
```cmd
python -m pip install "numpy==1.26.4"
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

```

Verify GPU:

```cmd
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

```

Expected:

```text
2.1.2+cu121 12.1 True

```

### 4.3 Install OpenMMLab
在 `rtmpose` 環境執行，以下每條命令必須分開執行。

```cmd
python -m pip install "numpy==1.26.4" "setuptools==80.9.0" wheel openmim

```

```cmd
mim install "mmengine>=0.9.0,<1.0.0"

```

```cmd
mim install "mmcv>=2.0.1,<2.2.0"

```

```cmd
mim install "mmdet>=3.1.0,<3.4.0"

```

MMCV 具有 compiled extensions，Windows 上可能需要匹配 PyTorch/CUDA 的 wheel。若 pip 嘗試從原始碼編譯，應先檢查相容版本。

### 4.4 Install MMPose and Dependencies
```cmd
python -m pip install chumpy==0.70 --no-build-isolation

```

```cmd
python -m pip install mmpose==1.3.2 --no-build-isolation

```

如果上述步驟已完成，無須重新執行。

最後固定 NumPy 版本並驗證：

```cmd
python -m pip install "numpy==1.26.4"

```

```cmd
python -c "import numpy, torch, mmcv, mmengine, mmdet, mmpose; print('NumPy:', numpy.__version__); print('CUDA:', torch.cuda.is_available()); print('MMPose:', mmpose.__version__)"

```

```cmd
python -m pip check

```

`pip check` 如果仍顯示版本衝突，需要檢查實際套件版本；單純 import 成功不保證所有依賴相容。

## 5. RTMPose Model
使用：

**RTMPose-m — Body8-Halpe26 — 256×192**

- Keypoints：26

- Output：2D coordinates + confidence

- MMPose model alias：`body26`

- Framework：MMPose

- Person detector：RTMDet（Inferencer 自動初始化）

官方 Model Zoo：

https://github.com/open-mmlab/mmpose/blob/main/configs/body_2d_keypoint/rtmpose/body8/rtmpose_body8-halpe26.md

模型 Config：

https://github.com/open-mmlab/mmpose/blob/main/configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb512-700e_body8-halpe26-256x192.py

官方 RTMPose-m checkpoint：

https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.pth

使用 `MMPoseInferencer(pose2d='body26')` 時，首次執行會自動嘗試下載模型及人物偵測權重。

一般使用模型別名時，不需要手動放置 `.pth` 檔案。

## 6. RTMPose Inference and Handoff

在 `rtmpose` 環境、`rtmpose/` 目錄執行：

```cmd
conda activate rtmpose
python run.py --input input/ --model l
```

其他可用指令：

```cmd
python run.py --input input/aerial.mp4
python run.py --input input/ --model m
python run.py --input input/ --model x
python run.py --input input/ --model l --output output
```

依現有 `run.py` / `convert.py` 的輸出設定，取得每支影片的 **AlphaPose-compatible JSON**（Halpe-26，每幀 78 個數值）。把 JSON 放入 `MotionBERT/input/alphapose/`；把對應原始影片放入 `MotionBERT/input/raw/`。例如：

```text
MotionBERT/input/raw/aerial.mp4
MotionBERT/input/alphapose/aerial_l.json
```

原始 JSON 如果是 MMPose Prediction 格式，需要先轉成 AlphaPose-compatible 格式；**不能直接拿一般 `predictions/*.json` 代替**。轉換範例（依實際檔名）：

```cmd
python convert.py --input output/predictions/aerial.json --output output/alphapose/aerial_l.json
```

RTMPose 的人物選擇目前採逐幀最大 Bounding Box，未進行跨幀追蹤。若原始影片含原速與慢放片段，應先剪成單一播放速度再做時間導數分析。

---

# Part B — MotionBERT Deployment
## 7. MotionBERT Installation
Official repository：

https://github.com/Walter0807/MotionBERT

### 7.1 Clone Repository
```cmd
cd ~\motionBert
git clone https://github.com/Walter0807/MotionBERT.git

```

已有專案時不需重複 Clone。

### 7.2 Prepare Environment
conda env

```cmd
conda create -n motionbert python=3.10 -y
conda activate motionbert

```

安裝 PyTorch：

```cmd
python -m pip install "numpy==1.26.4"
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

```

### 7.3 Install Requirements
在 MotionBERT 根目錄：

```cmd
cd ~\motionBert\MotionBERT

```

先安裝編譯工具：

```cmd
python -m pip install "setuptools==80.9.0" wheel

```

先安裝 `chumpy`：

```cmd
python -m pip install chumpy==0.70 --no-build-isolation

```

再安裝其他依賴：

```cmd
python -m pip install -r requirements.txt

```

最後檢查：

```cmd
python -m pip install "numpy==1.26.4"
python -m pip check

```

### 7.4 Fix Chumpy / NumPy Compatibility
若使用 SMPLify 提供的舊格式 SMPL `.pkl`，可能遇到：

```text
ImportError: cannot import name 'bool' from 'numpy'

```

開啟：

```cmd
notepad %CONDA_PREFIX%\Lib\site-packages\chumpy\__init__.py

```

找到：

```python
from numpy import bool, int, float, complex, object, unicode, str, nan, inf

```

改成：

```python
from numpy import nan, inf

```

這些舊名稱在 `chumpy` 模組中如果仍被引用，需保留對應的 Python built-in aliases。也可以在原檔 `import numpy` 之前加入相容性別名補丁：

```python
import numpy as np
for name, value in {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "unicode": str,
    "str": str,
}.items():
    if name not in np.__dict__:
        setattr(np, name, value)

```

這是 legacy compatibility workaround；修改 site-packages 只作用於目前 Conda environment，且重新安裝套件可能覆蓋修正。

驗證：

```cmd
python -c "import chumpy; print('Chumpy OK')"

```

如果還有其他舊版 Python API 相容性問題，需依 traceback 另外處理。

\---

# Part C — MotionBERT 3D Pose Estimation
## 8. Download 3D Pose Resources
### 8.1 Checkpoint
官方文件：

https://github.com/Walter0807/MotionBERT/blob/main/docs/inference.md

官方 3D Pose Checkpoint：

https://1drv.ms/f/s!AvAdh0LSjEOlgT67igq_cIoYvO2y?e=bfEc73

使用 MotionBERT-Lite checkpoint，放置：

```text
MotionBERT/
└── checkpoint/
    └── pose3d/
        └── FT_MB_lite_MB_ft_h36m_global_lite/
            └── best_epoch.bin

```

`best_epoch.bin` 是模型參數檔。

### 8.2 Windows DataLoader Modification
修改 `infer_wild.py` 中的：

```python
testloader_params = {
    'batch_size': 1,
    'shuffle': False,
    'num_workers': 0,
    'pin_memory': True,
    'drop_last': False
}

```

當 `num_workers=0` 時，移除：

```python
'prefetch_factor': 4,
'persistent_workers': True,

```

此設定避免 Windows `multiprocessing.spawn` 在沒有 main guard 的腳本中反覆初始化。

### 8.3 Single-video Debugging (Optional)

一般批次處理請直接使用 **Section 9.5 / Section 10**。以下僅供單影片測試 `infer_wild.py`：

```cmd
conda activate motionbert
cd /d C:\coding\motionBert\MotionBERT
python infer_wild.py --vid_path input/raw/aerial.mp4 --json_path input/alphapose/aerial_l.json --out_path output/debug_pose
```

此底層程式輸出 `output/debug_pose/X3D.npy` 與 `X3D.mp4`。批次模式會將其重新命名並複製到 `output/X3D/`。3D Skeleton 通常為 `(T, 17, 3)`，其絕對物理尺度尚須另外確認。

---

# Part D — MotionBERT 3D Mesh Recovery
## 9. Download 3D Mesh Resources
3D Mesh Recovery 使用 MotionBERT Mesh checkpoint 與 SMPL 人體模型。

### 9.1 Mesh Checkpoint
官方下載：

https://1drv.ms/f/s!AvAdh0LSjEOlgTmgYNslCDWMNQi9?e=WjcB1F

放置位置：

```text
MotionBERT/
└── checkpoint/
    └── mesh/
        └── FT_MB_release_MB_ft_pw3d/
            └── best_epoch.bin

```

### 9.2 SMPL Body Model
下載網站：

https://smplify.is.tue.mpg.de/

選擇：

**Code and model**

找到：

```text
basicModel_neutral_lbs_10_207_0_v1.0.0.pkl

```

重新命名：

```text
SMPL_NEUTRAL.pkl

```

放置：

```text
MotionBERT/data/mesh/SMPL_NEUTRAL.pkl

```

SMPL 模型需要依照官方授權條款下載及使用，請勿直接將模型檔案上傳到公開 Git Repository。

### 9.3 Additional SMPL Data
MotionBERT 官方 Mesh 文件：

https://github.com/Walter0807/MotionBERT/blob/main/docs/mesh.md

官方資料下載：

https://1drv.ms/f/c/a5438cd242871df0/IgDwHYdC0oxDIIClfwAAAAAAAf0qj7JhZX0sXEaX59EXOC0

至少確認以下檔案：

```text
MotionBERT/data/mesh/
├── SMPL_NEUTRAL.pkl
├── J_regressor_extra.npy
├── J_regressor_h36m_correct.npy
└── smpl_mean_params.npz

```

注意：`J_regressor_h36m_correct.npy` 是 MotionBERT `utils_smpl.py` 使用的實際檔名，與部分其他 SMPL repository 的 `J_regressor_h36m.npy` 不同。

### 9.4 Windows DataLoader Modification
修改 `infer_wild_mesh.py`：

```python
testloader_params = {
    'batch_size': 1,
    'shuffle': False,
    'num_workers': 0,
    'pin_memory': True,
    'drop_last': False
}

```

同樣移除 `prefetch_factor` 與 `persistent_workers`。

### 9.5 MotionBERT Batch Inference (Primary Entry Point)

在 `motionbert` 環境與 `MotionBERT/` 目錄執行：

```cmd
conda activate motionbert
cd /d C:\coding\motionBert\MotionBERT
python batch_infer.py
python batch_infer.py --skip-mesh
python batch_infer.py --force
python batch_infer.py --input input --output output
```

**輸入必須包含：**

```text
input/raw/<video>.mp4
input/alphapose/<video>_l.json   # 也支援 _m、_x 或無模型後綴
```

**實際輸出：**

```text
output/X3D/<json_stem>X3D.npy
output/X3D/<json_stem>X3D.mp4
output/mesh/<json_stem>Mesh.mp4
```

例如 `aerial_l.json` 輸出 `aerial_lX3D.npy`、`aerial_lX3D.mp4`、`aerial_lMesh.mp4`，而不是 `output/X3D.npy`、`output/mesh/mesh.mp4`。

**執行邏輯：**

1. 掃描 `input/alphapose/` 下所有 `.json`，依檔名配對 `input/raw/` 的影片。
2. 若 3D `.npy` 或 `.mp4` 任一不存在，執行 `infer_wild.py`；完整存在則跳過。
3. 未指定 `--skip-mesh` 時，如 Mesh 影片缺失，執行 `infer_wild_mesh.py`，並把同一支影片的 `X3D.npy` 當作 `--ref_3d_motion_path`。
4. 每支影片的底層輸出先放在暫存目錄，檢查檔案存在後複製至正式輸出目錄。
5. 個別影片失敗時記錄錯誤、繼續下一支；最後摘要若仍有失敗項目，程序以非零 Exit Code 結束。

**注意：** `--force` 同時重跑 X3D 與 Mesh（若 Mesh 未被略過）。Batch 目前使用 Serial Subprocesses，沒有 Cross-video Model Reuse，也沒有內建跨影格追蹤或缺幀修復。

---

# Part E — Complete Execution Workflow

## 10. Recommended Batch Commands

已完成 Section 4–9 的環境、Checkpoint、SMPL 資料安裝後，建議依以下順序執行。

**Step 1 — RTMPose：**

```cmd
conda activate rtmpose
cd /d C:\coding\motionBert\rtmpose
python run.py --input input/ --model l
```

**Step 2 — 準備 JSON 和影片：** 將 AlphaPose-compatible JSON 放在 `MotionBERT/input/alphapose/`，原始影片放在 `MotionBERT/input/raw/`。確認 `aerial_l.json` 對應 `aerial.mp4`，每幀具有 26 組 `(x, y, confidence)`。

**Step 3 — MotionBERT Batch：**

```cmd
conda activate motionbert
cd /d C:\coding\motionBert\MotionBERT
python batch_infer.py --skip-mesh
```

確認各影片的 `output/X3D/*X3D.npy` 與 `*X3D.mp4` 已產生後，再執行：

```cmd
python batch_infer.py
```

第二次呼叫會跳過已完整產生的 X3D，補做缺少的 Mesh。要全部重跑時才使用 `--force`。

**Step 4 — Check outputs：**

```cmd
python -c "from pathlib import Path; p=Path('output/X3D'); print([(x.name, __import__('numpy').load(x).shape) for x in sorted(p.glob('*X3D.npy'))])"
```

預期每支影片的 NumPy Shape 為 `(T, 17, 3)`，且 T 與所用 JSON 的影格序列相符。若後續進行 Jerk，還需依影片原始 FPS、座標慣例與 Chunk Boundary 檢查結果。

**Single-video / Legacy Mode：** `infer_wild.py`、`infer_wild_mesh.py` 可供特定影片除錯（見 Section 8.3）。原文件中將影片與 JSON 手動複製成固定 `1.mp4` / `alphapose-results.json` 再逐一推論的流程已移除，以免與現有 Batch 入口混淆。

---
# Part F - analysis
## Evaluation Pipeline — GT-free 3D Reconstruction Analysis

This pipeline evaluates the temporal and structural consistency of MotionBERT-based 3D human motion reconstruction on Tricking videos, without requiring 3D Ground Truth.

The workflow consists of two stages:

1. **Data Preparation:** Validate and normalize RTMPose 2D keypoints and MotionBERT 3D skeletons.
2. **Quantitative Evaluation:** Calculate three GT-free reconstruction consistency metrics.

### 1. Dependencies

```
conda activate trickingMB
pip install numpy opencv-python matplotlib
```

### 2. Stage 1 — Data Preparation

**Script:** `prepare_tricking.py`

This stage reads original RGB videos, RTMPose predictions, and MotionBERT 3D reconstruction results.

**Input:**

| Input          | Format  | Description                                   |
| -------------- | ------- | --------------------------------------------- |
| Original Video | `.mp4`  | Video metadata (FPS, resolution, frame count) |
| RTMPose        | `.json` | Halpe-26 2D keypoints and confidence scores   |
| MotionBERT     | `.npy`  | H36M-17 3D joint coordinates `(T, 17, 3)`     |

**Execution:**

```
python prepare_tricking.py --video-dir input --json-dir rtmpose\output\output-x\alphapose --pose3d-dir MotionBERT\output\output-x\X3D --output-dir output/analysis-x
```

**Processing:**

- Validate frame alignment, missing frames, duplicates, NaN/Inf, and confidence scores.
- Preserve raw 2D and 3D joint coordinates.
- Apply Pelvis-centered Coordinate Transformation.
- Apply clip-level Body-scale Normalization.
- Extract FPS and calculate timestamps.
- Batch-process videos using filename matching.

**Outputs:**

```
output/analysis-x/
├── aerial/
│   ├── aerial_prepared.npz
│   ├── aerial_metadata.json
│   ├── aerial_validation_report.json
│   └── aerial_confidence.csv
├── backTuck/
│   └── ...
├── butterflyKick/
│   └── ...
└── batch_summary.json
```

| Output                     | Description                                                     |
| -------------------------- | --------------------------------------------------------------- |
| `*_prepared.npz`           | Original and normalized 2D/3D joint arrays                      |
| `*_metadata.json`          | FPS, resolution, normalization settings, coordinate information |
| `*_validation_report.json` | Data integrity checks and warnings                              |
| `*_confidence.csv`         | Per-frame RTMPose confidence statistics                         |
| `batch_summary.json`       | Batch processing results                                        |

### 3. Stage 2 — Quantitative Evaluation

**Script:** `evaluate.py`

This stage reads the prepared NPZ data and metadata from each movement subfolder automatically.

**Execution:**

```
python evaluate.py --input-dir output/analysis-x --output-dir output/analysis-x
```

The script recursively processes the movement folders under the input directory. If an individual movement fails validation or evaluation, the error is reported and the remaining movements continue processing.

Three GT-free evaluation metrics are calculated:

| Metric                       | Description                                                                       | Statistics                                  |
| ---------------------------- | --------------------------------------------------------------------------------- | ------------------------------------------- |
| **Normalized Temporal Jerk** | Measures abrupt changes in 3D joint motion using third-order temporal differences | Mean, P95, frame-level and joint-level jerk |
| **Orientation Jump**         | Measures changes in estimated body orientation between consecutive frames         | Mean, P95 (degrees/frame)                   |
| **Bone Length CV**           | Measures temporal variation in reconstructed skeletal segment lengths             | Per-bone CV (%)                             |

### 4. Evaluation Output Structure

Evaluation results are grouped by **metric**, allowing direct comparison of different Tricking movements.

```
output/analysis-x/
│
├── aerial/                         # Stage 1 prepared data
├── backTuck/
├── butterflyKick/
├── butterflyTwist/
├── gainerFlash/
├── gainerSwitch/
├── pop540/
├── rodeoCorkscrew/
├── teradaGrab/
└── touchdownRaiz/
│
├── TemporalJerk/
│   ├── aerial_jerk.png
│   ├── aerial_jerk.csv
│   ├── aerial_joint_jerk.csv
│   ├── backTuck_jerk.png
│   └── ...
│
├── OrientationJump/
│   ├── aerial_orientation.png
│   ├── aerial_orientation.csv
│   ├── backTuck_orientation.png
│   └── ...
│
├── BoneLengthCV/
│   ├── aerial_bone_cv.png
│   ├── aerial_bone_cv.csv
│   ├── backTuck_bone_cv.png
│   └── ...
│
├── batch_summary.json
├── metrics_summary.csv
├── failure_frames.csv
├── evaluation_details.json
└── evaluation_errors.json
```

**Key outputs:**

| Output                    | Purpose                                                  |
| ------------------------- | -------------------------------------------------------- |
| `TemporalJerk/*.png`      | Visualize temporal jerk and identify abnormal peaks      |
| `OrientationJump/*.png`   | Visualize frame-to-frame orientation changes             |
| `BoneLengthCV/*.png`      | Compare skeletal length consistency across joints        |
| `metrics_summary.csv`     | Summarize evaluation metrics across all movements        |
| `failure_frames.csv`      | List candidate failure frames based on metric thresholds |
| `evaluation_details.json` | Detailed evaluation statistics                           |
| `evaluation_errors.json`  | Errors encountered during batch evaluation               |

### 5. Batch Processing Status

Both stages display per-video execution results.

Example:

```
[OK] aerial: 190 frames -> output\analysis-x (3 metrics)
[OK] backTuck: 185 frames -> output\analysis-x (3 metrics)
[ERROR] butterflyKick: ValueError: ... (skipped)
...
Batch complete: 10/11 OK
```

A failure in one video does not interrupt the entire batch.

### 6. Evaluation Notes

- Use **original-speed footage only** for quantitative evaluation. Remove slow-motion replays and editing transitions before reconstruction.
- Temporal Jerk uses the source FPS and clip-level body-scale normalization.
- Pelvis-centered coordinates exclude global root translation.
- High Jerk or Orientation Jump values indicate potential reconstruction instability and require verification against the original video.
- Bone Length CV measures structural consistency; lower values indicate more stable reconstructed bone lengths.
- These three metrics are **GT-free proxy metrics for reconstruction consistency**. They do not measure absolute 3D pose accuracy, such as MPJPE.

For reliable cross-model comparisons, maintain consistent joint definitions, coordinate conventions, sampling rates, and normalization procedures.


# Part G — Intermediate Outputs
## 11. File Descriptions
| File | Producer | Description |
|---|---|---|
| `input/raw/aerial.mp4` | User | 原始 RGB 影片，供時間與 FPS 對照 |
| RTMPose `predictions/*.json` | RTMPose | MMPose 原生 2D 預測 |
| `input/alphapose/aerial_l.json` | RTMPose / `convert.py` | MotionBERT-compatible Halpe-26 關節序列 |
| `output/X3D/aerial_lX3D.npy` | MotionBERT Pose3D | `(T, 17, 3)` 的 H36M-17 數值結果 |
| `output/X3D/aerial_lX3D.mp4` | MotionBERT Pose3D | 3D Skeleton 視覺化 |
| `output/mesh/aerial_lMesh.mp4` | MotionBERT Mesh | SMPL Mesh 視覺化；預設不輸出 Mesh Vertex `.npy` |

產物名稱使用 **JSON Stem**，可以同時保留 `_m`、`_l`、`_x` 等不同 RTMPose 模型結果，避免檔案互相覆蓋。

---

## 12. Keypoint Formats
### RTMPose Halpe-26
```text
(T, 26, 3)

```

每個關節：

```text
(x, y, confidence)

```

座標為原始影片的 2D Pixel Coordinates。

### MotionBERT H36M-17
```text
(T, 17, 3)

```

每個關節：

```text
(X, Y, Z)

```

為模型預測的 3D Pose Coordinates。

### SMPL Mesh
MotionBERT 預測 SMPL 相關姿態與形狀參數，再透過 SMPL Body Model 計算頂點。

標準 SMPL Mesh 包含 6,890 個 vertices。原始推論腳本僅輸出渲染影片。

\---

## 13. Validation

驗證 PyTorch 與 CUDA：

```cmd
python -c "import torch, numpy; print(torch.__version__, numpy.__version__, torch.cuda.is_available())"
```

驗證 MotionBERT 批次結果：

```cmd
python -c "import numpy as np; x=np.load('output/X3D/aerial_lX3D.npy'); print(x.shape, np.isfinite(x).all())"
```

預期 `shape=(T,17,3)`、`isfinite=True`。驗證 AlphaPose JSON（更換為實際檔名）：

```cmd
python -c "import json; d=json.load(open('input/alphapose/aerial_l.json')); print('Frames:', len(d), 'Keypoints:', len(d[0]['keypoints']))"
```

預期單幀 Flattened Keypoint 數量是 `26×3=78`。若原始影片有缺幀、慢放混接或 FPS 不一致，不可未檢查就進行 Jerk / Temporal Consistency 分析。

---

## 14. Limitations
- **Single-person assumption:** 目前 `convert.py` 選擇最大 Bounding Box，未實作完整追蹤。

- **Occlusion:** 人體遮擋、倒置、快速旋轉可能降低 Keypoint Accuracy。

- **Temporal alignment:** JSON 每幀須與原始影片一一對應，漏偵測時應先修復序列。

- **Domain gap:** MotionBERT 預訓練資料與 Tricking 特技運動存在姿態分布差異。

- **Scale ambiguity:** 單視角 3D 重建存在深度及絕對尺度不確定性。

- **SMPL topology:** Mesh 表面受 SMPL 人體模型的拓樸及形狀參數化限制。

- **No default geometry export:** 目前只輸出 Mesh Visualization Video，未直接輸出可編輯的幾何檔案。

## 15. References
- [MMPose GitHub](https://github.com/open-mmlab/mmpose)

- [RTMPose Body8-Halpe26 Model Zoo](https://github.com/open-mmlab/mmpose/blob/main/configs/body_2d_keypoint/rtmpose/body8/rtmpose_body8-halpe26.md)

- [MMPose Installation](https://mmpose.readthedocs.io/en/latest/installation.html)

- [MotionBERT GitHub](https://github.com/Walter0807/MotionBERT)

- [MotionBERT In-the-wild Inference](https://github.com/Walter0807/MotionBERT/blob/main/docs/inference.md)

- [MotionBERT Human Mesh Recovery](https://github.com/Walter0807/MotionBERT/blob/main/docs/mesh.md)

- [MotionBERT Halpe-26 Mapping](https://github.com/Walter0807/MotionBERT/blob/main/lib/data/dataset_wild.py)

- [SMPLify / SMPL Model](https://smplify.is.tue.mpg.de/)

\---

This project integrates MotionBERT, originally developed by its respective authors. The MotionBERT source code and its license are retained in the MotionBERT/ directory. Modifications in this repository are documented separately.
