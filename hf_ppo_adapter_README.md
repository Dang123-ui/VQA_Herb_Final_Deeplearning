---
license: other
base_model: Qwen/Qwen2.5-VL-3B-Instruct
library_name: peft
language:
- vi
tags:
- qwen2.5-vl
- visual-question-answering
- ppo
- reinforcement-learning
- lora
- vietnamese
- medicinal-herbs
---

# Qwen2.5-VL Herb PPO Adapter

This repository contains the PPO/RL adapter trained after supervised fine-tuning for Vietnamese Medicinal Herb Visual Question Answering.

Base model:

```text
Qwen/Qwen2.5-VL-3B-Instruct
```

Source SFT adapter:

```text
gdllelf/qwen25-vl-herb-qlora-adapter
```

Adapter file:

```text
final_ppo_adapter.zip
```

The PPO reward used in the project combines VQA soft accuracy and normalized BERTScore-F1.

Dataset:

```text
gdllelf/vietnamese-medicinal-herb-vqa
```

