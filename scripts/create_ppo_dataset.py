import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def normalize_image_path(image):
    return str(image or "").replace("\\", "/")


def create_ppo_dataset(input_path, output_path, limit):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as fout:
        for line in fin:
            if count >= limit:
                break
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            question = str(row.get("question", "")).strip()
            answer = str(row.get("answer", "")).strip()
            image = normalize_image_path(row.get("image", ""))

            if not question or not answer or not image:
                continue

            count += 1
            ppo_row = {
                "id": f"ppo_{count:06d}",
                "image": image,
                "question": question,
                "reference_answer": answer,
            }
            fout.write(json.dumps(ppo_row, ensure_ascii=False) + "\n")

    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="train/train.jsonl")
    parser.add_argument("--output", default="ppo/ppo_train_5000.jsonl")
    parser.add_argument("--limit", type=int, default=5000)
    args = parser.parse_args()

    input_path = ROOT / args.input
    output_path = ROOT / args.output

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    count = create_ppo_dataset(input_path, output_path, args.limit)
    print(f"Created {count} PPO rows: {output_path}")


if __name__ == "__main__":
    main()
