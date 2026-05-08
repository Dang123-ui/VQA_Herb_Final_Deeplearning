---
license: other
base_model: Qwen/Qwen2.5-VL-3B-Instruct
library_name: peft
language:
- vi
tags:
- qwen2.5-vl
- visual-question-answering
- lora
- qlora
- vietnamese
- medicinal-herbs
---

# Qwen2.5-VL Herb QLoRA Adapter

This repository contains the fine-tuned QLoRA adapter for Vietnamese Medicinal Herb Visual Question Answering.

Base model:

```text
Qwen/Qwen2.5-VL-3B-Instruct
```

Adapter file:

```text
final_finetuned_adapter.zip
```

The adapter was trained on a Vietnamese medicinal herb VQA dataset with image-question-answer triples.

Dataset:

```text
gdllelf/vietnamese-medicinal-herb-vqa
```

## Intended Use

Load the base model and apply this PEFT/LoRA adapter for Vietnamese visual question answering on medicinal herb images.

