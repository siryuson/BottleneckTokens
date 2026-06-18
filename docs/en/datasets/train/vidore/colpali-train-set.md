# ColPali Train Set

> Document retrieval training subset in the ViDoRe system, directly converted to BToks Lance single-table format.
> Within the source / conversion / runtime split, `image/query/answer` are stored facts while `<|image_pad|>` injection is runtime transform policy.

| Property | Value |
|------|-----|
| Modality | visdoc |
| Task Type | `document_retrieval` |
| Usage | Training |
| Provenance Layers | 2 layers |
| Training Weight | 10 |

## Original Dataset

### Source Info
| Property | Value |
|------|-----|
| Original Name | `vidore/colpali_train_set` |
| Paper | ColPali: Efficient Document Retrieval with Vision Language Models ([arXiv:2407.01449](https://arxiv.org/abs/2407.01449)) |
| Primary Authors | Raphael Fritsch, Titouan Yvon, Pablo Pirolles, Thibault Formal, Benjamin Piwowarski, Stéphane Clinchant |
| HuggingFace | [vidore/colpali_train_set](https://huggingface.co/datasets/vidore/colpali_train_set) |
| License | See original dataset |

### Schema
The original data centers on document page images and query-answer pairs, with BToks training using the corresponding fields:

| Field | Type | Description |
|------|------|------|
| image | struct{bytes, path} | Document page image |
| query | string | Query text |
| answer | string | Positive sample text answer |

### Sample
> Full examples are provided in the BToks Lance section below.

---

## BToks Lance Format

### Source Info
| Property | Value |
|------|-----|
| Data Loader | `vidore_train` |
| Implementation | `src/vlm2emb/data/datasets/vidore.py` |
| Registry Name | `vidore_train` |
| Lance Path | `./data/converted` |

### Lance Layout

**train.lance (Single Table)**:

| Column | Type | Description |
|------|------|------|
| image | struct{bytes: binary, path: string} | Document page image |
| image_filename | string | Original image filename |
| query | string | Query text |
| answer | string | Positive text answer |
| source | string | Source subset |
| options | string | Auxiliary upstream metadata |
| page | string | Page identifier when available |
| model | string | Upstream model metadata |
| prompt | string | Upstream prompt metadata |
| answer_type | string | Answer-type metadata |

### Training-side Mapping
- query: runtime transform turns it into `"<|image_pad|>\n{query}\n"` plus the single page image
- positive: `"{answer}\n"` as a text-only positive sample
- negative: empty text placeholder
- `source / page / answer_type` and related columns stay in storage metadata and are not part of the minimal training view

### Sample

The examples below are sampled directly from `./data/converted`.

- `image.bytes` is summarized as `bytes / sha1 / format / size`.
- Other fields keep their Lance values.

#### Example 1

```json
{
  "data/train.lance": {
    "image": {
      "path": "1810.07757_2.jpg",
      "bytes": {
        "bytes": 167632,
        "sha1": "229b31fafc52",
        "format": "JPEG",
        "size": [
          1000,
          1600
        ]
      }
    },
    "image_filename": "images/1810.07757_2.jpg",
    "query": "Comparing panels a, b, c, and d, which statement best describes the data variance?",
    "answer": "D",
    "source": "arxiv_qa",
    "options": "['A. The variance of the data decreases from panel a to panel d.', 'B. The variance of the data increases from panel a to panel d.', 'C. The data presents no variance in any of the panels.', 'D. The variance of the data is inconsistent across the panels.', '-']",
    "page": "",
    "model": "gpt4V",
    "prompt": "",
    "answer_type": null
  }
}
```

#### Example 2

```json
{
  "data/train.lance": {
    "image": {
      "path": "page_9.jpg",
      "bytes": {
        "bytes": 319580,
        "sha1": "855059eda212",
        "format": "JPEG",
        "size": [
          1700,
          2200
        ]
      }
    },
    "image_filename": "data/scrapped_pdfs_split/pages_extracted/energy_train/1d09a977-063b-463f-a897-2eda99c1a4f6.pdf/page_9.jpg",
    "query": "What is the duration of the course mentioned in the image?",
    "answer": "['five to ten hours, not including field trips']",
    "source": "pdf",
    "options": null,
    "page": "9",
    "model": "sonnet",
    "prompt": "\n        You are an assistant specialized in Multimodal RAG tasks.\n\n        The task is the following: given an image from a pdf page, you will have to \n        generate questions that can be asked by a user to retrieve information from \n        a large documentary corpus. \n        The question should be relevant to the page, and should not be too specific \n        or too general. The question should be about the subject of the page, and \n        the answer need to be found in the page. \n\n        Remember that the question is asked by a user to get some information from a\n        large documentary corpus that contains multimodal data. Generate a question \n        that could be asked by a user without knowing the existence and the content \n        of the corpus. \n\n        Generate as well the answer to the question, which should be found in the\n        page. And the format of the answer should be a list of words answering the\n        question. \n\n\n        Generate at most THREE pairs of questions and answers per page in a \n        dictionary with the following format, answer ONLY this dictionary\n        NOTHING ELSE: \n\n\n        {\n            \"questions\": [\n                {\n                    \"question\": \"XXXXXX\",\n                    \"answer\": [\"YYYYYY\"]\n                },\n                {\n                    \"question\": \"XXXXXX\",\n                    \"answer\": [\"YYYYYY\"]\n                },\n                {\n                    \"question\": \"XXXXXX\",\n                    \"answer\": [\"YYYYYY\"]\n                },\n            ]\n        }\n        where XXXXXX is the question and ['YYYYYY'] is the corresponding list of answers\n        that could be as long as needed. \n\n\n        Note: If there are no questions to ask about the page, return an empty list.\n        Focus on making relevant questions concerning the page. \n\n        Here is the page: \n\n",
    "answer_type": null
  }
}
```

#### Example 3

```json
{
  "data/train.lance": {
    "image": {
      "path": null,
      "bytes": {
        "bytes": 63908,
        "sha1": "3ca701571c57",
        "format": "PNG",
        "size": [
          1709,
          2293
        ]
      }
    },
    "image_filename": "0fd47b51ae9248ef36669b8619b1223f268edae3e7a44ac1e6cebbbfaaf69f96",
    "query": "What is the date?\nYour answer should be very brief.",
    "answer": "OCTOBER 17, 1995.",
    "source": "docvqa",
    "options": null,
    "page": null,
    "model": null,
    "prompt": null,
    "answer_type": null
  }
}
```
