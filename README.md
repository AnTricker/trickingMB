# 單視角人體動作重建：MotionBERT、MotionAGFormer 與 GVHMR

本專案以 Tricking 影片為輸入，整理三種人體動作重建流程：**RTMPose + MotionBERT**、**MotionAGFormer** 與 **GVHMR**。目前使用預訓練模型進行推論，並提供 MotionBERT 結果的無 Ground Truth（GT-free）時序與骨架一致性分析。

前半部列出方法概述、日常執行指令與輸出；首次使用請先完成後方的環境、Checkpoint 與人體模型部署。

- [1. 三種方法概述](#methods)
- [2. 專案結構與共用輸入](#structure)
- [3. RTMPose + MotionBERT 指令](#motionbert)
- [4. MotionAGFormer 指令](#motionagformer)
- [5. GVHMR 指令](#gvhmr)
- [6. MotionBERT 結果分析](#analysis)
- [補充說明：環境與部署](#deployment)
- [補充說明：資料格式、驗證與限制](#details)

<a id="methods"></a>

## 1. 三種方法概述

| 方法 | 現有流程 | 主要輸出 | 執行入口 |
|---|---|---|---|
| RTMPose + MotionBERT | RTMPose Halpe-26 2D → MotionBERT 3D Pose / SMPL Mesh | H36M-17 `(T,17,3)` NumPy、骨架影片、Mesh 影片 | `rtmpose/run.py`、`MotionBERT/batch_infer.py` |
| MotionAGFormer | YOLOv3 + HRNet 2D → Transformer-GCNFormer 3D Pose | 2D Keypoints NPZ、2D / 3D / 合成影格、合成影片 | `MotionAGFormer/batch_infer.py`、`demo/vis.py` |
| GVHMR | 人物追蹤、ViTPose / HMR2 特徵、相機運動估計 → 人體參數與全域動作 | `hmr4d_results.pt`、相機視角 / 全域 Mesh 影片 | `GVHMR/batch_infer.py`、`tools/demo/demo.py` |

三條流程各自處理 2D 或影像特徵；MotionAGFormer 與 GVHMR 的現有入口不讀取 RTMPose AlphaPose JSON。MotionBERT 與 MotionAGFormer 的骨架結果，以及 GVHMR 的人體參數，需先統一關節、座標、尺度與時間定義，才能進行跨方法數值比較。

<a id="structure"></a>

## 2. 專案結構與共用輸入

```text
motionBert/
├── README.md
├── input/                        # 共用原始影片，例如 aerial.mp4
├── output/                       # GT-free 分析結果
├── prepare.py                    # MotionBERT 評估資料準備
├── evaluate.py                   # GT-free 指標計算
├── rtmpose/
│   ├── run.py                    # 2D 推論並自動轉換 AlphaPose JSON
│   └── output/output-l/          # 指令範例使用的 RTMPose-l 輸出
├── MotionBERT/
│   ├── batch_infer.py
│   ├── infer_wild.py
│   ├── infer_wild_mesh.py
│   ├── input/raw/                # 配對影片
│   ├── input/alphapose/          # 配對 JSON
│   └── output/output-l/          # 指令範例使用的 3D / Mesh 輸出
├── MotionAGFormer/
│   ├── batch_infer.py
│   ├── demo/vis.py
│   └── demo/output/              # 目前 demo 固定的輸出位置
└── GVHMR/
    ├── batch_infer.py             # 遞迴推論、續跑與合併影片整理
    ├── tools/demo/demo.py
    ├── tools/demo/demo_folder.py
    ├── inputs/checkpoints/
    └── outputs/                   # 批次入口依 input 分類存放
```

以下執行指令以 **Windows Anaconda Prompt / CMD**、專案位於 `C:\coding\motionBert` 為例；請依實際路徑調整。各流程使用自己的 Conda environment。`output-l` 是範例中透過參數指定的資料夾名稱，不是自動產生的模型設定。

將原速影片放入根目錄 `input/`。MotionBERT / MotionAGFormer 的批次入口只掃描指定資料夾第一層，支援 `.mp4`、`.mov`、`.avi`、`.mkv`。GVHMR 的 `batch_infer.py` 遞迴掃描所有子資料夾的 `.mp4`（副檔名不分大小寫）；上游 `demo_folder.py` 只掃描第一層。以下複製指令以 `.mp4` 為例。

### Git 管理方式

`MotionBERT/`、`MotionAGFormer/`、`GVHMR/` 都是主專案直接追蹤的一般子目錄，不是獨立儲存庫，也不是主專案的 Git submodule。三種方法的程式碼、本地修改、設定、README 與 LICENSE 統一在 `C:\coding\motionBert` 執行 `git add`、`git commit`、`git push`。

根目錄 `.gitignore` 會排除下載的 Checkpoint、SMPL / SMPL-X 人體模型、訓練資料、原始影片、推論與分析產物、環境及快取。上游程式附帶的範例影片、文件圖片，以及 GVHMR 必要的小型 regressor、映射與設定資源仍保留追蹤；`.pt` / `.npy` 等副檔名不一定代表可丟棄的推論產物。

`.gitignore` 不會自動取消已加入追蹤或暫存的檔案。日常操作在根目錄執行：

```cmd
git add -- .gitignore README.md MotionBERT MotionAGFormer GVHMR
git diff --cached --stat
git commit -m "Update reconstruction pipelines"
```

`GVHMR/.gitmodules` 是上游保留的選用 DPVO 宣告；目前一般推論使用 SimpleVO，不需要將 GVHMR 本身設為 submodule。

<a id="motionbert"></a>

## 3. RTMPose + MotionBERT 指令

### 3.1 RTMPose：影片 → AlphaPose JSON

```cmd
conda activate rtmpose
cd /d C:\coding\motionBert\rtmpose
python run.py --input ../input --model l --output output/output-l
```

`--model` 支援 `m`、`l`、`x`（預設 `m`）。改用其他大小時，建議同時更改輸出資料夾，以區分結果。單影片範例：

```cmd
python run.py --input ../input/aerial.mp4 --model l --output output/output-l
```

`run.py` 會自動選擇逐幀最大 Bounding Box 的人物，並轉換為 AlphaPose-compatible Halpe-26 JSON，無須另外執行轉換腳本。

```text
rtmpose/output/output-l/
├── predictions/aerial_l.json
├── alphapose/aerial_l.json       # MotionBERT 使用此格式
└── visualizations/aerial_l.mp4
```

### 3.2 準備 MotionBERT 輸入

```cmd
cd /d C:\coding\motionBert\MotionBERT
if not exist input\raw mkdir input\raw
if not exist input\alphapose mkdir input\alphapose
copy /Y ..\input\*.mp4 input\raw\
copy /Y ..\rtmpose\output\output-l\alphapose\*.json input\alphapose\
```

配對規則：`aerial_l.json`、`aerial_m.json`、`aerial_x.json` 均對應 `input/raw/aerial.mp4`，也支援無模型後綴的 JSON。JSON 每幀需包含 26 組 `(x, y, confidence)`，即 78 個數值。請使用 `alphapose/`，而非 MMPose 原生 `predictions/` JSON。

### 3.3 MotionBERT：3D 骨架與 Mesh

先產生骨架，再補做 Mesh：

```cmd
conda activate motionbert
cd /d C:\coding\motionBert\MotionBERT
python batch_infer.py --skip-mesh --output output/output-l
python batch_infer.py --output output/output-l
```

第二條指令會跳過已完整存在的骨架輸出，並產生缺少的 Mesh。其他操作：

```cmd
python batch_infer.py --force --output output/output-l
python batch_infer.py --input input --output output
python infer_wild.py --vid_path input/raw/aerial.mp4 --json_path input/alphapose/aerial_l.json --out_path output/debug_pose
```

- `--force`：重跑骨架與 Mesh；搭配 `--skip-mesh` 時只重跑骨架。
- 批次輸出：`output/output-l/X3D/aerial_lX3D.npy`、`aerial_lX3D.mp4`，以及 `output/output-l/mesh/aerial_lMesh.mp4`。
- 單影片底層入口輸出固定名稱 `X3D.npy`、`X3D.mp4`；批次入口再依 JSON stem 重新命名。
- Batch 為逐影片子程序，會重新載入模型。個別影片失敗後繼續處理，最後若有失敗則回傳非零 Exit Code。
- 骨架 `.npy` 或 `.mp4` 任一缺失就重跑骨架；Mesh 使用對應骨架作為 `--ref_3d_motion_path`。

<a id="motionagformer"></a>

## 4. MotionAGFormer 指令

此流程自行執行 YOLOv3 + HRNet 2D Pose，再重建 H36M-17 骨架。**目前本地 `demo/vis.py` 固定使用 243-frame Large 架構與 `checkpoint/motionagformer-l-h36m.pth.tr`**，沒有模型大小或輸出目錄的 CLI 選項。

### 4.1 共用影片資料夾批次推論

```cmd
conda activate motionagformer
cd /d C:\coding\motionBert\MotionAGFormer
python batch_infer.py ../input --gpu 0
```

批次入口依序啟動 `demo/vis.py`，第一支失敗的影片會中止批次。它不提供 `--force` 或既有結果跳過選項。

### 4.2 單影片推論

將影片放入 `MotionAGFormer/demo/video/` 後執行：

```cmd
python demo/vis.py --video aerial.mp4 --gpu 0
```

兩種入口都輸出到：

```text
MotionAGFormer/demo/output/aerial/
├── input_2D/keypoints.npz        # reconstruction 欄位：2D Keypoints + confidence
├── pose2D/                      # 2D 影格
├── pose3D/                      # 3D 影格
├── pose/                        # 合成影格
└── aerial.mp4                   # 合成影片
```

目前 demo 不儲存 3D 骨架 NumPy，無法直接接到本專案的 `prepare.py`。現有 `output-b/`、`output-l/` 是其他批次結果資料夾，不能透過目前 CLI 指定。合成影片使用原始 FPS 取整數後加 5，時間分析應以原始影片的 FPS 為準。

<a id="gvhmr"></a>

## 5. GVHMR 指令

GVHMR 透過人物追蹤、ViTPose 與 HMR2 影像特徵估計人體參數與全域動作，並渲染相機視角及全域 Mesh。它有獨立的 Checkpoint 與 SMPL / SMPL-X 部署需求，見後方補充說明。
### 5.0 需開WSL才能用這邊
```
(base) root@An:/mnt/c/coding/motionBert# conda activate gvhmr
(gvhmr) root@An:/mnt/c/coding/motionBert# 
```

### 5.1 共用影片資料夾批次推論

一般相機運動估計使用預設的 SimpleVO：

```cmd
cd /d C:\coding\motionBert\GVHMR
python tools/demo/demo_folder.py -f ../input -d outputs/demo
```

若影片由固定相機拍攝，可加 `-s` 略過相機運動估計：

```cmd
python tools/demo/demo_folder.py -f ../input -d outputs/demo -s
```

上述為上游第一層資料夾入口。處理目前多層 `input/`，請使用新增的遞迴入口：

```cmd
python batch_infer.py
```

預設路徑依腳本位置解析，不受執行目錄影響；整理影片預設存放於專案根目錄的 `output/Output_InCam_Global_Horizon/`。自訂相對路徑則相對於執行時的工作目錄。自訂路徑或指定固定相機：

```cmd
python batch_infer.py --input ../input --output-root outputs --summary-root ../output/Output_InCam_Global_Horizon
python batch_infer.py -s
```

輸出會自動建立並保留輸入分類，例如：

```text
input/1twist/cork/corkscrew.mp4
GVHMR/outputs/1twist/cork/corkscrew/hmr4d_results.pt
GVHMR/outputs/1twist/cork/corkscrew/corkscrew_3_incam_global_horiz.mp4
output/Output_InCam_Global_Horizon/1twist/cork/corkscrew_3_incam_global_horiz.mp4
```

- 原始合併影片已存在：完全跳過 GVHMR；整理區缺檔時才直接複製，保留結果影片原檔名與內容，不重新編碼。
- 只有 `hmr4d_results.pt`：呼叫單影片入口，沿用快取補渲染／合併；沒有推論結果時才執行完整流程。
- 整理區已有對應影片：跳過複製，不覆寫。判斷僅依檔案存在，不檢查影片內容；更換輸入、模型或相機設定時請使用新的輸出與整理根目錄。
- 開始前檢查同分類的 stem 衝突（含大小寫差異）；發現衝突就報錯。輸出及整理根目錄必須位於輸入資料夾之外，避免產物被再次掃描。
- 個別影片失敗後繼續處理；子程序失敗或缺少合併影片時不複製該支影片，最後有失敗即回傳非零 exit code。
- 最後統計 `inference`（選擇完整推論的次數）、`reused`（沿用影片或推論資料的次數）、`copied`、`skipped`（已有整理影片）與 `failed`。前兩者包含失敗的嘗試，計數可重疊。無輸入影片時顯示零筆並正常結束。

### 5.2 單影片推論

```cmd
python tools/demo/demo.py --video ../input/aerial.mp4 --output_root outputs/demo
python tools/demo/demo.py --video ../input/aerial.mp4 --output_root outputs/static -s
```

- `-s` / `--static_cam` 僅適用固定相機。
- `--use_dpvo` 是單影片入口的選用參數，需另外安裝 DPVO；未指定時使用 SimpleVO。
- `--f_mm` 可在單影片入口指定全片幅等效焦距；`--verbose` 可輸出額外預處理視覺化。
- 上游 `demo_folder.py` 逐支啟動子程序，遇到失敗就中止；新增的 `batch_infer.py` 則繼續處理。已有的預處理與重建檔案會被重用；更換影片、設定或模型時，請指定新的輸出根目錄。

```text
GVHMR/outputs/demo/aerial/
├── 0_input_video.mp4
├── preprocess/                  # bbx.pt、vitpose.pt、vit_features.pt 等
├── hmr4d_results.pt              # smpl_params_incam、smpl_params_global 等
├── 1_incam.mp4
├── 2_global.mp4
└── aerial_3_incam_global_horiz.mp4
```

目前程式以 30 FPS 寫入輸入副本與渲染影片；以其他 FPS 的影片做時間分析前，需確認時間對齊。`hmr4d_results.pt` 是參數字典，不能直接作為 MotionBERT 的 `(T,17,3)` NumPy 使用。

<a id="analysis"></a>

## 6. MotionBERT 結果分析

現有 `prepare.py` / `evaluate.py` 使用 **AlphaPose Halpe-26 JSON + H36M-17 3D NumPy**。以下接續第 3 節的 RTMPose-l / MotionBERT 輸出，不會自動轉換其他方法的結果。

### 6.1 準備與評估指令

在已安裝 NumPy、OpenCV、Matplotlib 的 `motionbert` 環境執行：

```cmd
conda activate motionbert
cd /d C:\coding\motionBert
python prepare.py --video-dir input --json-dir rtmpose/output/output-l/alphapose --pose3d-dir MotionBERT/output/output-l/X3D --output-dir output/analysis-l
python evaluate.py --input-dir output/analysis-l --output-dir output/analysis-l
```

改評估 RTMPose-x 時，將上游輸出資料夾及以上路徑一致改成 `output-x` / `analysis-x`。每個 JSON / NPY 資料夾應只包含此次要比較的版本，否則同一影片配對到多個結果時會報錯。

`prepare.py` 遞迴配對影片與結果，檢查 2D 影格及 3D Shape / 有限值，保留原始座標，對 3D 做 Pelvis-centered transformation 和整段影片固定的 Body-scale normalization，再依原始 FPS 建立時間戳。兩階段均會記錄個別影片錯誤並繼續處理其餘影片。

### 6.2 指標與產物

| 指標 | 用途 |
|---|---|
| Normalized Temporal Jerk | 以三階時間差分觀察正規化 3D 動作突變，輸出 Mean、P95 與逐幀 / 逐關節數據 |
| Orientation Jump | 以肩膀與髖部定義的身體方向 proxy 觀察相鄰影格角度跳變，單位 degrees/frame |
| Bone Length CV | 觀察重建骨段長度的時序變異，輸出各骨段 CV (%) |

```text
output/analysis-l/
├── aerial/
│   ├── aerial_prepared.npz
│   └── aerial_metadata.json
├── backTuck/
│   └── ...
├── batch_summary.json           # prepare.py 的批次結果
├── TemporalJerk/                # *_jerk.png、*_jerk.csv、*_joint_jerk.csv
├── OrientationJump/             # *_orientation.png、*_orientation.csv
├── BoneLengthCV/                # *_bone_cv.png、*_bone_cv.csv
├── metrics_summary.csv
├── failure_frames.csv
├── evaluation_details.json
└── evaluation_errors.json
```

這些指標衡量 **GT-free 重建一致性**，不代表 MPJPE 等絕對 3D 準確度。只使用原速影片，先移除慢放重播與剪輯轉場；高 Jerk / Orientation Jump 需回看原片判讀。Pelvis-centered 座標不包含全域根節點平移。

---

<a id="deployment"></a>

## 補充說明：環境與部署

以下保留 RTMPose / MotionBERT 的詳細部署步驟，並補上 MotionAGFormer 與 GVHMR 的需求。日常推論使用前半部指令；各方法請分開建立環境。

### A. RTMPose / MotionBERT 環境基準

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

### B. RTMPose 部署

#### B.1 Create Conda Environment

```cmd
conda create -n rtmpose python=3.10 -y
conda activate rtmpose

```

#### B.2 Install PyTorch

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

#### B.3 Install OpenMMLab

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

#### B.4 Install MMPose and Dependencies

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

#### B.5 RTMPose Model

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

實際 `run.py` 依 `--model` 選擇 m / l / x 的完整 Config 名稱；上列 `body26` 是 MMPose 的模型別名範例。


### C. MotionBERT 環境安裝

Official repository：

https://github.com/Walter0807/MotionBERT

現有 `MotionBERT/` 已包含本專案的批次入口與本地修改，無須重複 Clone。

#### C.1 Prepare Environment

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

#### C.2 Install Requirements

在 MotionBERT 根目錄：

```cmd
cd /d C:\coding\motionBert\MotionBERT

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

#### C.3 Fix Chumpy / NumPy Compatibility

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

### D. MotionBERT 3D Pose 資源

#### D.1 Checkpoint

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

#### D.2 Windows DataLoader 設定

本地 `infer_wild.py` 已使用以下設定；更新上游程式時請確認：

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

### E. MotionBERT Mesh 資源

3D Mesh Recovery 使用 MotionBERT Mesh checkpoint 與 SMPL 人體模型。

#### E.1 Mesh Checkpoint

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

#### E.2 SMPL Body Model

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

#### E.3 Additional SMPL Data

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

#### E.4 Windows DataLoader Modification

本地 `infer_wild_mesh.py` 已使用以下設定；更新上游程式時請確認：

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

### F. MotionAGFormer 部署

上游環境記錄為 Python 3.8.10、PyTorch 2.0.0、CUDA 12.2；以下為獨立環境的建立方式，PyTorch CUDA 版本需依本機配置選擇。

```cmd
conda create -n motionagformer python=3.8 -y
conda activate motionagformer
cd /d C:\coding\motionBert\MotionAGFormer
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`requirements.txt` 未固定 PyTorch 版本；若需重現上游環境，請先依本機配置安裝相容的 PyTorch / torchvision，再安裝其餘依賴。現有 demo 直接呼叫 `.cuda()`，需要可用的 NVIDIA GPU。

依 [MotionAGFormer 上游文件](MotionAGFormer/README.md) 的 Demo 與 Evaluation 下載連結準備：

```text
MotionAGFormer/
├── checkpoint/
│   └── motionagformer-l-h36m.pth.tr
└── demo/lib/checkpoint/
    ├── pose_hrnet_w48_384x288.pth
    └── yolov3.weights
```

上游 Demo 原本示範 Base 模型，但目前本地程式固定為 Large；只下載 Base checkpoint 不符合目前入口。完整訓練與 Dataset preprocessing 見子專案 README。

### G. GVHMR 部署

依據本地 [GVHMR 安裝說明](GVHMR/docs/INSTALL.md) 與 `requirements.txt`：Python 3.10、PyTorch 2.3.0 + CUDA 12.1、torchvision 0.18.0、NumPy 1.23.5。請獨立建立 `gvhmr` 環境，避免與 MotionBERT 的 NumPy / PyTorch 版本混用。

**原始 requirements 指定 Linux x86_64 的 PyTorch3D wheel，不能直接當成 Windows 安裝指令。** 以下安裝範例使用 Linux / WSL；原生 Windows 需先準備相容的 PyTorch3D，再調整該依賴。前方的 CMD 指令適用已完成相應環境部署的 Windows 工作目錄；在 WSL 中請使用 Linux 路徑，例如 `/mnt/c/coding/motionBert/GVHMR`。

```bash
conda create -n gvhmr python=3.10 -y
conda activate gvhmr
cd /mnt/c/coding/motionBert/GVHMR
python -m pip install -r requirements.txt
python -m pip install -e .
python -c "import torch, pytorch3d; print(torch.__version__, torch.cuda.is_available())"
```

依子專案安裝說明的下載連結與各模型授權，放置以下資料：

```text
GVHMR/inputs/checkpoints/
├── body_models/
│   ├── smpl/SMPL_NEUTRAL.pkl
│   └── smplx/SMPLX_NEUTRAL.npz
├── gvhmr/gvhmr_siga24_release.ckpt
├── hmr2/epoch=10-step=25000.ckpt
├── vitpose/vitpose-h-multi-coco.pth
└── yolo/yolov8x.pt
```

安裝 DPVO 為選用項目，額外使用 `inputs/checkpoints/dpvo/dpvo.pth`，請依 `GVHMR/docs/INSTALL.md` 處理其編譯依賴。一般推論使用 SimpleVO，固定相機則使用 `-s`。渲染另使用子專案 `hmr4d/utils/body_model/` 的 regressors 與 SMPL-X → SMPL 轉換資源，請保留完整來源檔案。

---

<a id="details"></a>

## 補充說明：資料格式、驗證與限制

### H. MotionBERT 中間產物

| File | Producer | Description |
|---|---|---|
| `input/raw/aerial.mp4` | User | 原始 RGB 影片，供時間與 FPS 對照 |
| RTMPose `predictions/*.json` | RTMPose | MMPose 原生 2D 預測 |
| `input/alphapose/aerial_l.json` | RTMPose `run.py` | MotionBERT-compatible Halpe-26 關節序列 |
| `output/output-l/X3D/aerial_lX3D.npy` | MotionBERT Pose3D | `(T, 17, 3)` 的 H36M-17 數值結果 |
| `output/output-l/X3D/aerial_lX3D.mp4` | MotionBERT Pose3D | 3D Skeleton 視覺化 |
| `output/output-l/mesh/aerial_lMesh.mp4` | MotionBERT Mesh | SMPL Mesh 視覺化；預設不輸出 Mesh Vertex `.npy` |

產物名稱使用 **JSON Stem**，可以同時保留 `_m`、`_l`、`_x` 等不同 RTMPose 模型結果，避免檔案互相覆蓋。

---

### I. Keypoint Formats

#### RTMPose Halpe-26

```text
(T, 26, 3)

```

每個關節：

```text
(x, y, confidence)

```

座標為原始影片的 2D Pixel Coordinates。

#### MotionBERT H36M-17

```text
(T, 17, 3)

```

每個關節：

```text
(X, Y, Z)

```

為模型預測的 3D Pose Coordinates。

#### SMPL Mesh

MotionBERT 預測 SMPL 相關姿態與形狀參數，再透過 SMPL Body Model 計算頂點。

標準 SMPL Mesh 包含 6,890 個 vertices。原始推論腳本僅輸出渲染影片。

---

### J. MotionBERT Validation

以下在 `motionbert` 環境與 `MotionBERT/` 工作目錄執行，路徑接續第 3 節。

驗證 PyTorch 與 CUDA：

```cmd
python -c "import torch, numpy; print(torch.__version__, numpy.__version__, torch.cuda.is_available())"
```

驗證 MotionBERT 批次結果：

```cmd
python -c "import numpy as np; x=np.load('output/output-l/X3D/aerial_lX3D.npy'); print(x.shape, np.isfinite(x).all())"
```

預期 `shape=(T,17,3)`、`isfinite=True`。驗證 AlphaPose JSON（更換為實際檔名）：

```cmd
python -c "import json; d=json.load(open('input/alphapose/aerial_l.json')); print('Frames:', len(d), 'Keypoints:', len(d[0]['keypoints']))"
```

預期單幀 Flattened Keypoint 數量是 `26×3=78`。若原始影片有缺幀、慢放混接或 FPS 不一致，不可未檢查就進行 Jerk / Temporal Consistency 分析。

---

### K. MotionBERT / 單視角重建限制

- **Single-person assumption:** 目前 RTMPose `run.py` 選擇最大 Bounding Box，未實作完整追蹤。

- **Occlusion:** 人體遮擋、倒置、快速旋轉可能降低 Keypoint Accuracy。

- **Temporal alignment:** JSON 每幀須與原始影片一一對應，漏偵測時應先修復序列。

- **Domain gap:** MotionBERT 預訓練資料與 Tricking 特技運動存在姿態分布差異。

- **Scale ambiguity:** 單視角 3D 重建存在深度及絕對尺度不確定性。

- **SMPL topology:** Mesh 表面受 SMPL 人體模型的拓樸及形狀參數化限制。

- **No default geometry export:** 目前只輸出 Mesh Visualization Video，未直接輸出可編輯的幾何檔案。

### L. References

- [MMPose GitHub](https://github.com/open-mmlab/mmpose)

- [RTMPose Body8-Halpe26 Model Zoo](https://github.com/open-mmlab/mmpose/blob/main/configs/body_2d_keypoint/rtmpose/body8/rtmpose_body8-halpe26.md)

- [MMPose Installation](https://mmpose.readthedocs.io/en/latest/installation.html)

- [MotionBERT GitHub](https://github.com/Walter0807/MotionBERT)

- [MotionBERT In-the-wild Inference](https://github.com/Walter0807/MotionBERT/blob/main/docs/inference.md)

- [MotionBERT Human Mesh Recovery](https://github.com/Walter0807/MotionBERT/blob/main/docs/mesh.md)

- [MotionBERT Halpe-26 Mapping](https://github.com/Walter0807/MotionBERT/blob/main/lib/data/dataset_wild.py)

- [SMPLify / SMPL Model](https://smplify.is.tue.mpg.de/)

- [MotionAGFormer 本地文件](MotionAGFormer/README.md)
- [GVHMR 本地文件](GVHMR/README.md)
- [GVHMR 部署說明](GVHMR/docs/INSTALL.md)

本專案整合 MotionBERT、MotionAGFormer 與 GVHMR，各子專案的原始作者、引用與授權請見其 README 和 LICENSE。模型權重、SMPL / SMPL-X、原始影片與產出資料依各自授權管理，不隨程式碼上傳。
