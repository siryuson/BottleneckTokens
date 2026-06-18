# 快速入门

本指南帮助你安装 BToks 并运行你的第一个实验。

## 环境要求

- Python 3.11 或更高版本

## 安装

### 标准安装

```bash
# 安装包
pip install -e .
```

### 开发安装

用于贡献者和开发者：

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 验证安装
python -c "from vlm2emb import Config, Registry; print('BToks 安装成功！')"
```

### 完整安装

安装所有可选依赖：

```bash
pip install -e ".[full]"
```

## 核心依赖

| 包 | 版本 | 用途 |
|---|------|------|
| torch | ≥2.0.0 | 深度学习框架 |
| transformers | ≥4.52.0 | Hugging Face transformers |
| accelerate | ≥0.20.0 | 分布式训练 |
| peft | ≥0.10.0 | 参数高效微调 |

## 快速开始

### 1. 验证安装

```python
from vlm2emb import AutoModule

# 检查可用组件
print("可用模块:", AutoModule.list_modules())
```

### 2. 加载配置

```python
from vlm2emb import load_config

# 加载公开训练 preset
config = load_config("configs/presets/btoks_qwen2vl_2b_v1.yaml")

# 访问配置值
print(f"模型类型: {config.model.type}")
```

### 3. 从配置创建组件

```python
from vlm2emb import AutoModule

# 从配置字典构建模块（必须包含 'type' 键）
backbone = AutoModule.from_config({
    "type": "Qwen2VLBackbone",
    "model_name_or_path": "Qwen/Qwen2-VL-7B-Instruct",
})

# 或者使用 build 方法直接指定类型
pooling = AutoModule.build("LastTokenPooling")

print(f"隐藏层大小: {backbone.hidden_size}")
```

## 项目结构

```
BToks/
├── configs/                 # YAML 配置文件
│   ├── models/             # 模型配置
│   ├── peft/               # LoRA 配置
│   ├── training/           # 训练策略
│   ├── datasets/           # 数据集配置
│   ├── eval/               # 评测片段
│   └── presets/            # 完整实验预设
├── src/vlm2emb/            # 源代码
│   ├── auto.py             # Auto* 注册表
│   ├── config.py           # 配置加载
│   ├── registry.py         # 注册表实现
│   ├── modules/            # Backbone、Pooling、Normalize
│   ├── data/               # 数据集、整理器、预处理
│   └── training/           # 训练器、回调
├── docs/                   # 文档
│   ├── en/                 # 英文文档
│   └── zh/                 # 中文文档
└── tests/                  # 测试套件
```

## 后续步骤

- [架构概览](../architecture/overview.md) - 了解系统设计
- [配置指南](../architecture/config-system.md) - 掌握 YAML 配置
- [开发指南](../developer-guide/contributing.md) - 参与 BToks 开发

## 故障排除

### 导入错误

如果遇到导入错误，请确保使用 Python 3.11+：

```bash
python --version  # 应该是 3.11.x 或更高版本
```

### CUDA 问题

验证 CUDA 是否可用：

```python
import torch
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"CUDA 版本: {torch.version.cuda}")
```

### 内存问题

对于大模型，考虑：
- 使用 LoRA 进行参数高效微调 
- 减小批次大小
- 启用梯度检查点

---

_需要帮助？在公开仓库中提交 issue。_
