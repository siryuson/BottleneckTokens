# MMLongBench-Doc 训练数据集

> 状态：已接入。当前实现提供 raw-preserving Lance 转换、PDF 单页图像表和 `mmlongbench_doc_train` runtime。

长文档理解数据集。archive 训练 parser 只使用 `evidence_pages` 中恰好包含一个页码的样本，并把对应 PDF 页渲染成正例文档图像。

| 属性 | 值 |
|------|-----|
| 模态 | visdoc |
| 任务类型 | `document_image_retrieval` |
| 用途 | 训练 / OOD 评测风险审计 |
| 注册名 | `mmlongbench_doc_train` |
| 转换脚本 | `scripts/convert/train/convert_mmlongbench_doc_to_lance.py` |
| runtime | `src/vlm2emb/data/datasets/mmlongbench_doc_train.py` |

## 原始数据

| 属性 | 值 |
|------|-----|
| 原始名称 | MMLongBench-Doc |
| HuggingFace | [yubo2333/MMLongBench-Doc](https://huggingface.co/datasets/yubo2333/MMLongBench-Doc) |
| 本地目录 | `./data/converted` |
| parquet | `data/train-00000-of-00001.parquet` |
| PDF 目录 | `documents/` |

原始 parquet 保留字段如下，不在主表中增加派生列：

| 字段 | 类型 | 说明 |
|------|------|------|
| `doc_id` | string | PDF 文件名 |
| `doc_type` | string | 文档类型 |
| `question` | string | 问题文本 |
| `answer` | string | 答案文本 |
| `evidence_pages` | string | 字符串形式的页码列表 |
| `evidence_sources` | string | 证据来源 |
| `answer_format` | string | 答案格式 |

## 转换结果

转换输出根目录建议为 `./data/converted`。

转换命令必须显式传入 MMEB-V2 `MMLongBench-doc` eval root。该参数用于生成默认的 leak-free 训练 split；如果缺失或路径下没有 `queries.lance`，转换会直接失败，避免静默生成未排除 eval 重叠的训练表。

| 路径 | 行数 | 说明 |
|------|------|------|
| `data/official_train_raw.lance` | 1091 | 原始 parquet 全量行 |
| `data/official_train.lance` | 473 | archive parser 可用样本：单一证据页且 PDF 页可渲染 |
| `data/official_train_without_mmeb_v2_eval.lance` | 4 | 默认训练 split，排除 MMEB-V2 `MMLongBench-doc` eval query 重叠 |
| `data/pages.lance` | 418 | PDF 单页图像表，按唯一 `{doc_id}#page={page_index}` 去重 |
| `data/exclusions/unusable_rows.lance` | 618 | 非单页、缺 PDF 或渲染失败样本 |
| `data/exclusions/mmeb_v2_eval.lance` | 469 | 与 MMEB-V2 eval query 重叠的可用样本 |

`data/pages.lance` schema：

| 字段 | 类型 | 说明 |
|------|------|------|
| `path` | string | join key，格式为 `{doc_id}#page={page_index}` |
| `doc_id` | string | PDF 文件名 |
| `page_index` | int32 | archive parser 使用的页码索引 |
| `image` | binary | 渲染后的 JPEG 字节 |

## Runtime 输出

默认配置：

```yaml
type: mmlongbench_doc_train
path: ./data/converted
split: official_train_without_mmeb_v2_eval
transform:
  query:
    instruction: "Find a document image that matches the given query:"
    instruction_body_separator: space
    trailing_newline: ensure_single
  positive:
    instruction: "Understand the content of the provided document image."
    visual_token_placement: own_line
    trailing_newline: ensure_single
  negative:
    trailing_newline: ensure_single
    empty: empty_multimodal_input
```

真实默认 split 样本之一：

- 原始问题：`List all the different icons about networks that can be found in Status Bar`
- query text：`Find a document image that matches the given query: List all the different icons about networks that can be found in Status Bar\n`
- positive text：`<|image_pad|>\nUnderstand the content of the provided document image.\n`
- positive media：`mi_phone.pdf#page=10`

## 审计结论

本数据集不能按 archive 全量默认启用。`official_train` 中 473 条 archive 可用样本有 469 条与 MMEB-V2 `MMLongBench-doc` eval query 文本重叠。默认训练配置必须使用 `official_train_without_mmeb_v2_eval`；只有在复现 archive 训练配方时才使用 `official_train`。

转换需要 PyMuPDF 来渲染 PDF 页面，也需要可读取的 MMEB-V2 eval root 来执行重叠排除。缺少任一依赖时，转换脚本会直接报错；runtime 不依赖 PyMuPDF，只读取已经写入 Lance 的 JPEG 字节。
