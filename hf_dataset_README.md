---
license: cc-by-nc-4.0
language:
- vi
task_categories:
- visual-question-answering
pretty_name: Vietnamese Medicinal Herb VQA
size_categories:
- 10K<n<100K
---

# Vietnamese Medicinal Herb VQA

This dataset contains Vietnamese visual question answering samples for medicinal herbs and traditional medicinal materials.

Each JSONL row has the following fields:

```json
{
  "id": "P000002676",
  "name": "Cây Hẹ",
  "image": "images/P000002266.jpg",
  "question": "Hoa trong ảnh có màu gì?",
  "answer": "Màu trắng"
}
```

## Splits

| Split | Images | QA pairs |
|---|---:|---:|
| Train | 3032 | 23211 |
| Validation | 379 | 2998 |
| Test | 380 | 2952 |

The test split includes a manually reviewed test JSONL file used for evaluation.

## Intended Use

The dataset is intended for Vietnamese multimodal Visual Question Answering experiments, especially zero-shot and fine-tuned evaluation of vision-language models such as Qwen2.5-VL, LLaVA, BLIP/BLIP-2, PaliGemma, and related models.

## Notes

- Questions and answers are in Vietnamese.
- Answers are short, generally up to 10 words after preprocessing.
- The domain is specialized: Vietnamese medicinal herbs and medicinal materials.
- Images are referenced by relative path in the JSONL files.

