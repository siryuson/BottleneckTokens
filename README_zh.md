# Bottleneck Tokens for Unified Multimodal Retrieval

本仓库用于开源复现论文
**Bottleneck Tokens for Unified Multimodal Retrieval**（`BToks`），对应
[arXiv:2604.11095](https://arxiv.org/abs/2604.11095)。

Python import namespace 保留为 `vlm2emb`，用于兼容当前可运行工程；公开方法名、
配置名和文档主线统一使用 `BToks`。

英文主文档见 [README.md](README.md)。

## 范围

包含：

- BToks token 注入、池化和 bottleneck-cache generation 工具。
- Generative Information Condensation 训练损失和 Condensation Mask 路径。
- VLM2Vec、GME-style 和 BToks 训练预设。
- Lance 格式训练数据和 MMEB-V2 评测数据加载。
- 训练、评测、数据转换和公开性检查脚本。
- 配置、BToks 模块、训练辅助逻辑和评测入口的重点测试。

不包含：

- checkpoint。checkpoint 发布与本次代码开源分开处理。
- 私有日志、本地缓存、内部集群 launcher 或内部数据路径。
- 本地规划产物或仅供 workspace 使用的 agent 指令文件。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
python -m pip install -r requirements.txt
```

分布式 GPU 训练时，请按本机 CUDA/PyTorch 环境安装匹配的 FlashAttention 和
DeepSpeed，例如：

```bash
python -m pip install -e ".[gpu]"
```

## 数据

运行时使用 Lance 表。公开配置默认读取本地路径：

- `VLM2EMB_MMEB_TRAIN_ROOT`，默认 `./data/MMEB-train`
- `VLM2EMB_MMEB_V2_ROOT`，默认 `./data/MMEB-V2`
- `VLM2EMB_DATA_ROOT`，默认 `./data`，用于 base 和 expanded preset 里的其它训练根目录。

默认 README 训练和评测命令需要先下载。下面的仓库 ID 是当前已准备好的
Lance 数据公开托管地址，不代表论文作者或组织的唯一身份。

```bash
export BTOKS_MMEB_TRAIN_HF_REPO=siyrus/BToks-MMEB-train
export BTOKS_MMEB_V2_HF_REPO=siyrus/BToks-MMEB-V2
export BTOKS_VIDORE_HF_REPO=siyrus/BToks-vidore_colpali_train_set
export BTOKS_VISRAG_HF_REPO=siyrus/BToks-openbmb_VisRAG-Ret-Train-In-domain-data
export BTOKS_LLAVAHOUND_HF_REPO=siyrus/BToks-ShareGPTVideo_train_video_and_instruction

python scripts/download_lance_data.py "$BTOKS_MMEB_TRAIN_HF_REPO" --local-dir data/MMEB-train
python scripts/download_lance_data.py "$BTOKS_MMEB_V2_HF_REPO" --local-dir data/MMEB-V2
python scripts/download_lance_data.py "$BTOKS_VIDORE_HF_REPO" --local-dir data/vidore_colpali_train_set
python scripts/download_lance_data.py "$BTOKS_VISRAG_HF_REPO" --local-dir data/openbmb_VisRAG-Ret-Train-In-domain-data
python scripts/download_lance_data.py "$BTOKS_LLAVAHOUND_HF_REPO" --local-dir data/ShareGPTVideo_train_video_and_instruction
```

也可以显式指定本地路径：

```bash
export VLM2EMB_MMEB_TRAIN_ROOT=/path/to/MMEB-train
export VLM2EMB_MMEB_V2_ROOT=/path/to/MMEB-V2
export VLM2EMB_DATA_ROOT=/path/to/additional-training-roots
```

expanded training preset 会使用 `VLM2EMB_DATA_ROOT` 下更多公开数据根目录；
每个本地目录名应和对应 Hugging Face dataset repo 中 `BToks-` 后面的后缀一致。

## 训练

BToks Qwen2-VL-2B：

```bash
PYTHONPATH=src python scripts/train.py configs/presets/btoks_qwen2vl_2b_v1.yaml \
  train.args.output_dir=./outputs/btoks_qwen2vl_2b_v1 \
  train.args.report_to=none
```

VLM2Vec baseline：

```bash
PYTHONPATH=src python scripts/train.py configs/presets/vlm2vec_qwen2vl_2b.yaml \
  train.args.output_dir=./outputs/vlm2vec_qwen2vl_2b \
  train.args.report_to=none
```

GME-style baseline：

```bash
PYTHONPATH=src python scripts/train.py configs/presets/gme_qwen2vl_2b.yaml \
  train.args.output_dir=./outputs/gme_qwen2vl_2b \
  train.args.report_to=none
```

## 评测

本次首个代码发布不包含 checkpoint。准备好你自己的本地 checkpoint 后，可以运行：

```bash
PYTHONPATH=src python scripts/eval.py configs/presets/btoks_qwen2vl_2b_v1.yaml \
  --checkpoint /path/to/local/checkpoint \
  --output-dir ./eval/btoks_qwen2vl_2b_v1
```

## 验证

```bash
PYTHONPATH=src python -c "from vlm2emb import BToksConfig; print(BToksConfig.model_type)"
PYTHONPATH=src python -c "from vlm2emb.config import load_config; load_config('configs/presets/btoks_qwen2vl_2b_v1.yaml')"
PYTHONPATH=src python scripts/train.py --help
PYTHONPATH=src python scripts/eval.py --help
PYTHONPATH=src python scripts/check_public_release.py
PYTHONPATH=src pytest tests/test_btoks_ablation.py tests/test_btoks_trainer.py tests/evaluate/test_eval_script.py
```

创建公开首个完整代码 commit 前，还需要检查 `git status --short`，确保
本地规划、缓存和 agent-control 文件不会进入公开提交。
