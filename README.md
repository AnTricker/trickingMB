# RTMPose + MotionBERT: 2D Pose, 3D Skeleton and Mesh Recovery

## 1. Project Overview

本專案使用 RTMPose 與 MotionBERT，由單視角影片取得人體 2D Keypoints，進一步生成 3D Skeleton 與 SMPL 3D Mesh。

主要流程：

```text
Input Video (.mp4)
        |
        v
RTMPose-m (Body8-Halpe26)
        |
        v
MMPose Predictions JSON
        |
        v
convert.py
        |
        v
AlphaPose-compatible JSON
        |
        v
MotionBERT / WildDetDataset
        |
        +-- Halpe-26 → H36M-17
        |
        +-- 3D Pose Estimation
        |       |
        |       +-- X3D.npy
        |       +-- X3D.mp4
        |
        +-- 3D Mesh Recovery
                |
                +-- mesh.mp4
```

目前使用 pretrained models 進行 inference，未進行 Tricking 專用資料的 fine-tuning。

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

## 3. Project Structure

```text
motionBert\
│
├── README.md
│
├── rtmpose\
│   ├── requirements.txt
│   ├── run.py
│   ├── convert.py
│   ├── input\
│   │   └── 1.mp4
│   └── output\
│       ├── predictions\
│       │   └── 1.json
│       ├── visualizations\
│       │   └── 1.mp4
│       └── alphapose-results.json
│
└── MotionBERT\
    ├── requirements.txt
    ├── infer_wild.py
    ├── infer_wild_mesh.py
    │
    ├── configs\
    │   ├── pose3d\
    │   └── mesh\
    │
    ├── checkpoint\
    │   ├── pose3d\
    │   │   └── FT_MB_lite_MB_ft_h36m_global_lite\
    │   │       └── best_epoch.bin
    │   └── mesh\
    │       └── FT_MB_release_MB_ft_pw3d\
    │           └── best_epoch.bin
    │
    ├── data\
    │   └── mesh\
    │       ├── SMPL_NEUTRAL.pkl
    │       ├── J_regressor_extra.npy
    │       ├── J_regressor_h36m_correct.npy
    │       └── smpl_mean_params.npz
    │
    ├── input\
    │   ├── 1.mp4
    │   └── alphapose-results.json
    │
    └── output\
        ├── X3D.npy
        ├── X3D.mp4
        └── mesh\
            └── mesh.mp4
```

MMPose 視覺化資料夾可能因版本與 API 設定而使用 `visualization` 或 `visualizations`，請以實際輸出為準。

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

## 6. RTMPose Inference

執行指令
在指定 Conda 環境、目錄下：
使用 RTMPose-m（預設）
python run.py --input input/rodeoCorkscrew.mp4


使用 RTMPose-l
python run.py --input input/rodeoCorkscrew.mp4 --model l


使用 RTMPose-x
python run.py --input input/rodeoCorkscrew.mp4 --model x


也可以指定其他影片及輸出資料夾：
python run.py --input input/butterflyTwist.mp4 --model x --output output


輸出後如何接 MotionBERT？
例如執行 --model l 後：
output/alphapose/rodeoCorkscrew_l.json


這個檔案可以直接複製到 MotionBERT/input/，再執行原本的 infer_wild.py。
注意兩件事： RTMPose-l／x 首次使用可能需要下載對應權重；而目前的 max(bbox_area) 仍是逐幀選擇最大人物，沒有跨影格身份追蹤。此外，若某幀完全沒偵測到人，程式會明確報錯，不會自行補幀，避免把不完整的 2D 序列送進 MotionBERT。
---

# Part C — MotionBERT Deployment

## 8. MotionBERT Installation

Official repository：

https://github.com/Walter0807/MotionBERT

### 8.1 Clone Repository

```cmd
cd ~\motionBert
git clone https://github.com/Walter0807/MotionBERT.git
```

已有專案時不需重複 Clone。

### 8.2 Prepare Environment

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

### 8.3 Install Requirements

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

### 8.4 Fix Chumpy / NumPy Compatibility

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

---

# Part D — MotionBERT 3D Pose Estimation

## 9. Download 3D Pose Resources

### 9.1 Checkpoint

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

### 9.2 Windows DataLoader Modification

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

### 9.3 Run 3D Pose Estimation

```cmd
conda activate motionbert
cd ~\motionBert\MotionBERT
```

```cmd
python infer_wild.py --vid_path input/1.mp4 --json_path input/alphapose-results.json --out_path output
```

產物：

```text
MotionBERT/output/
├── X3D.mp4
└── X3D.npy
```

- `X3D.mp4`：3D Skeleton visualization。
- `X3D.npy`：每個影格的 H36M-17 3D Keypoints，shape `(T, 17, 3)`。

MotionBERT-Lite 會預測 17 個關節的 3D 座標。此結果不包含完整人體表面 Mesh，也不保證單視角推論得到的 3D 座標具有真實世界尺度。

---

# Part E — MotionBERT 3D Mesh Recovery

## 10. Download 3D Mesh Resources

3D Mesh Recovery 使用 MotionBERT Mesh checkpoint 與 SMPL 人體模型。

### 10.1 Mesh Checkpoint

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

### 10.2 SMPL Body Model

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

### 10.3 Additional SMPL Data

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

### 10.4 Windows DataLoader Modification

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

### 10.5 Run 3D Pose Estimation (Generate X3D.npy)

在執行 3D Mesh Recovery 前，可先透過 MotionBERT 的 `infer_wild.py` 產生 3D Skeleton，取得後續 Mesh Root Motion 調整所需的 `X3D.npy`。

**Step 1 — 執行 3D Pose Estimation**

```cmd
python infer_wild.py --vid_path input/1.mp4 --json_path input/alphapose-results.json --out_path output
```

**Step 2 — 確認輸出**

```text
MotionBERT/
└── output/
    ├── X3D.npy
    └── X3D.mp4
```

| File | Description |
|---|---|
| `X3D.npy` | 3D Skeleton 座標，shape 為 `(T, 17, 3)`，包含每幀的 17 個 H36M 關節 |
| `X3D.mp4` | 3D Skeleton 視覺化影片 |

### 10.6 Run 3D Mesh Recovery

MotionBERT 提供 `infer_wild_mesh.py`，將 2D Keypoints 重建為 SMPL 3D Human Mesh。

**Option A — 不使用 3D Skeleton Reference**

```cmd
python infer_wild_mesh.py --vid_path input/1.mp4 --json_path input/alphapose-results.json --out_path output/mesh
```

**Option B — 使用 X3D.npy 作為 3D Motion Reference（建議）**

```cmd
python infer_wild_mesh.py --vid_path input/1.mp4 --json_path input/alphapose-results.json --out_path output/mesh --ref_3d_motion_path output/X3D.npy
```

`--ref_3d_motion_path` 會利用先前估計的 3D Skeleton 作為參考，進行 Mesh 的尺度匹配與 Root Trajectory 調整。由於參考資料本身仍是單視角估計結果，因此無法保證真實世界尺度與位置精度。

**Output**

```text
MotionBERT/
└── output/
    ├── X3D.npy
    ├── X3D.mp4
    └── mesh/
        └── mesh.mp4
```

- `X3D.npy`：3D Skeleton 數值資料。
- `X3D.mp4`：3D Skeleton 視覺化。
- `mesh.mp4`：SMPL 3D Mesh 視覺化。

**注意：** Mesh 腳本需要 CUDA-enabled PyTorch。Windows 環境應將 `infer_wild.py` 與 `infer_wild_mesh.py` 的 DataLoader `num_workers` 設定為 `0`，避免 multiprocessing 錯誤。
### 10.6 Mesh Output

```text
MotionBERT/output/mesh/
└── mesh.mp4
```

`mesh.mp4` 為 3D SMPL Mesh 視覺化影片。

官方 `infer_wild_mesh.py` 預設在記憶體中計算 `verts_all`，但不會直接儲存 `.obj`、`.ply`、`.npy` 頂點資料。若要用於 Blender 或後續 3D 分析，需要另外擴充匯出程式。

---

# Part F — Complete Execution Workflow

## 11. Full Commands

以下假設 RTMPose、MotionBERT、模型權重及環境均已完成部署。

### Step 1 — RTMPose 2D Inference

```cmd
conda activate rtmpose
cd ~\motionBert\rtmpose
python run.py
```

產生：

```text
rtmpose/output/predictions/1.json
```

### Step 2 — Convert Keypoints

```cmd
python convert.py --input output/predictions/1.json --output output/alphapose-results.json
```

產生：

```text
rtmpose/output/alphapose-results.json
```

### Step 3 — Transfer Files

```cmd
if not exist ..\MotionBERT\input mkdir ..\MotionBERT\input
copy output\alphapose-results.json ..\MotionBERT\input\alphapose-results.json
copy input\1.mp4 ..\MotionBERT\input\1.mp4
```

### Step 4 — MotionBERT 3D Skeleton

```cmd
conda activate motionbert
cd ~\motionBert\MotionBERT
python infer_wild.py --vid_path input/1.mp4 --json_path input/alphapose-results.json --out_path output
```

產生：

```text
output/X3D.mp4
output/X3D.npy
```

### Step 5 — MotionBERT 3D Mesh

```cmd
python infer_wild_mesh.py --vid_path input/1.mp4 --json_path input/alphapose-results.json --out_path output/mesh --ref_3d_motion_path output/X3D.npy
```

產生：

```text
output/mesh/mesh.mp4
```

---

# Part G — Intermediate Outputs

## 12. File Descriptions

| File | Producer | Data | Purpose |
|---|---|---|---|
| `1.mp4` | User | RGB Video | Original video |
| `predictions/1.json` | RTMPose | Halpe-26, 2D | Raw pose detections |
| `alphapose-results.json` | `convert.py` | Halpe-26, 2D | MotionBERT-compatible input |
| `X3D.npy` | MotionBERT Pose3D | H36M-17, 3D | Numerical 3D skeleton |
| `X3D.mp4` | MotionBERT Pose3D | Video | Skeleton visualization |
| `mesh.mp4` | MotionBERT Mesh | Video | SMPL mesh visualization |

## 13. Keypoint Formats

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

---

## 14. Validation

驗證 PyTorch：

```cmd
python -c "import torch, numpy; print(torch.__version__, numpy.__version__, torch.cuda.is_available())"
```

驗證 MotionBERT Pose3D 輸出：

```cmd
python -c "import numpy as np; x=np.load('output/X3D.npy'); print(x.shape)"
```

預期：

```text
(T, 17, 3)
```

驗證 JSON：

```cmd
python -c "import json; d=json.load(open('input/alphapose-results.json')); print('Frames:', len(d), 'Keypoints:', len(d[0]['keypoints']))"
```

預期 Keypoints 數量：

```text
78
```

## 16. Limitations

- **Single-person assumption:** 目前 `convert.py` 選擇最大 Bounding Box，未實作完整追蹤。
- **Occlusion:** 人體遮擋、倒置、快速旋轉可能降低 Keypoint Accuracy。
- **Temporal alignment:** JSON 每幀須與原始影片一一對應，漏偵測時應先修復序列。
- **Domain gap:** MotionBERT 預訓練資料與 Tricking 特技運動存在姿態分布差異。
- **Scale ambiguity:** 單視角 3D 重建存在深度及絕對尺度不確定性。
- **SMPL topology:** Mesh 表面受 SMPL 人體模型的拓樸及形狀參數化限制。
- **No default geometry export:** 目前只輸出 Mesh Visualization Video，未直接輸出可編輯的幾何檔案。

## 17. References

- [MMPose GitHub](https://github.com/open-mmlab/mmpose)
- [RTMPose Body8-Halpe26 Model Zoo](https://github.com/open-mmlab/mmpose/blob/main/configs/body_2d_keypoint/rtmpose/body8/rtmpose_body8-halpe26.md)
- [MMPose Installation](https://mmpose.readthedocs.io/en/latest/installation.html)
- [MotionBERT GitHub](https://github.com/Walter0807/MotionBERT)
- [MotionBERT In-the-wild Inference](https://github.com/Walter0807/MotionBERT/blob/main/docs/inference.md)
- [MotionBERT Human Mesh Recovery](https://github.com/Walter0807/MotionBERT/blob/main/docs/mesh.md)
- [MotionBERT Halpe-26 Mapping](https://github.com/Walter0807/MotionBERT/blob/main/lib/data/dataset_wild.py)
- [SMPLify / SMPL Model](https://smplify.is.tue.mpg.de/)

---
This project integrates MotionBERT, originally developed by its respective authors. The MotionBERT source code and its license are retained in the MotionBERT/ directory. Modifications in this repository are documented separately.