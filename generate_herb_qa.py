import argparse
import json
import mimetypes
import os
import re
import time
import unicodedata
import random
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup
from google import genai
from google.genai import types


BASE_URL = "https://tracuuduoclieu.vn"
DEFAULT_MODEL = "gemma-4-26b-a4b-it"
QA_PER_IMAGE = 10
QA_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "qa": {
            "type": "array",
            "minItems": QA_PER_IMAGE,
            "maxItems": QA_PER_IMAGE,
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                },
                "required": ["question", "answer"],
            },
        },
    },
    "required": ["qa"],
}
BATCH_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "qa": QA_RESPONSE_SCHEMA["properties"]["qa"],
                },
                "required": ["id", "qa"],
            },
        },
    },
    "required": ["items"],
}
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi,vi-VN;q=0.9,en;q=0.8",
}


def find_image_path(images_dir: Path, image_id: str) -> Path | None:
    matches = sorted(images_dir.glob(f"{image_id}.*"))
    return matches[0] if matches else None


def clean_text(text: str, max_chars: int = 6000) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def fetch_url(url: str) -> str:
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.text


def find_herb_detail_url(name: str) -> str | None:
    search_url = f"{BASE_URL}/search/{quote_plus(name)}"
    soup = BeautifulSoup(fetch_url(search_url), "html.parser")
    candidates = soup.select("article h2 a, .entry-title a, h2.entry-title a")
    if not candidates:
        return None

    normalized_name = name.casefold()
    for link in candidates:
        title = link.get_text(" ", strip=True).casefold()
        href = link.get("href")
        if href and normalized_name in title:
            return href

    return candidates[0].get("href")


def fetch_herb_web_context(name: str, max_chars: int = 6000) -> dict[str, str]:
    try:
        detail_url = find_herb_detail_url(name)
        if not detail_url:
            return {"url": "", "text": ""}

        soup = BeautifulSoup(fetch_url(detail_url), "html.parser")
        detail = soup.select_one("#detail-dl") or soup.select_one("main.content") or soup
        return {"url": detail_url, "text": clean_text(detail.get_text(" ", strip=True), max_chars)}
    except requests.RequestException as exc:
        return {"url": "", "text": f"Không lấy được nội dung web: {exc}"}


def build_prompt(item: dict[str, Any], web_context: dict[str, str]) -> str:
    return f"""
Bạn là chuyên gia tạo dữ liệu VQA tiếng Việt về dược liệu, tập trung vào những gì nhìn thấy trong ảnh.

Nhiệm vụ:
- Dựa vào nội dung nhìn thấy trong ảnh.
- Tạo đúng 10 câu hỏi và câu trả lời.
- Trong 10 câu hỏi phải có 1 câu hỏi nhận dạng tên dược liệu trong ảnh, ví dụ "Tên dược liệu trong ảnh là gì?". Câu trả lời là tên dược liệu đã cung cấp.
- Trong 10 câu hỏi phải có 1 câu hỏi về chức năng/công dụng của dược liệu có kèm tên dược liệu trong câu hỏi, ví dụ "Dược liệu {item["name"]} có công dụng gì theo thông tin được cung cấp?". Câu trả lời chỉ dựa trên nội dung web được cung cấp.
- 8 câu hỏi còn lại phải tập trung vào đặc điểm trực quan của dược liệu trong ảnh.
- Bao phủ nhiều nhóm đặc điểm: hình dạng lá/thân/rễ/hoa/quả, màu sắc, bề mặt nhẵn/sần/có lông/bóng, dạng mọc chùm/cụm/đơn lẻ/dây leo/bụi, vị trí trái/phải/trên/dưới/giữa, kích thước tương đối, bộ phận nổi bật nhất.
- Bộ câu hỏi phải đa dạng: yes/no dựa trên ảnh, đếm số lượng nếu nhìn rõ, nhận dạng bộ phận, thuộc tính trực quan, quan hệ không gian/vị trí, so sánh kích thước tương đối.
- Với câu hỏi đếm số lượng, chỉ hỏi khi có thể đếm được rõ từ ảnh; nếu không rõ thì hỏi thuộc tính hoặc vị trí.
- Không hỏi về tên khoa học, họ thực vật, hoạt chất, cách dùng, liều dùng hoặc thông tin không nhìn thấy trong ảnh. 
- Không bịa chi tiết không quan sát được trong ảnh.
- Câu trả lời ngắn gọn, trực tiếp, dựa trên ảnh.
- Trả về JSON thuần, không markdown.

Thông tin mẫu:
id: {item["id"]}
tên dược liệu: {item["name"]}
nội dung web để trả lời câu công dụng:
{web_context.get("text", "")}

Schema JSON bắt buộc:
{{
  "qa": [
    {{
      "question": "...",
      "answer": "..."
    }}
  ]
}}
""".strip()


def build_batch_prompt(samples: list[dict[str, Any]]) -> str:
    blocks = []
    for sample in samples:
        item = sample["item"]
        web_context = sample["web_context"]
        blocks.append(
            f"""
id: {item["id"]}
tên dược liệu: {item["name"]}
nội dung web để trả lời câu công dụng:
{web_context.get("text", "")}
""".strip()
        )

    joined_blocks = "\n\n---\n\n".join(blocks)
    return f"""
Bạn là chuyên gia tạo dữ liệu VQA tiếng Việt về dược liệu, tập trung vào những gì nhìn thấy trong ảnh.

Bạn sẽ nhận nhiều ảnh trong cùng một request. Trước mỗi ảnh có dòng đánh dấu dạng "Ảnh của mẫu ID ...".

Nhiệm vụ cho từng ảnh:
- Tạo đúng 10 câu hỏi và câu trả lời cho đúng ID của ảnh đó.
- Trong 10 câu hỏi phải có 1 câu hỏi nhận dạng tên dược liệu trong ảnh, ví dụ "Tên dược liệu trong ảnh là gì?". Câu trả lời là tên dược liệu đã cung cấp.
- Trong 10 câu hỏi phải có 1 câu hỏi về chức năng/công dụng của dược liệu có kèm tên dược liệu trong câu hỏi, ví dụ "Dược liệu này có công dụng gì theo thông tin được cung cấp?". Câu trả lời chỉ dựa trên nội dung web được cung cấp.
- 8 câu hỏi còn lại phải tập trung vào đặc điểm trực quan của dược liệu trong ảnh.
- Bao phủ nhiều nhóm đặc điểm: hình dạng lá/thân/rễ/hoa/quả, màu sắc, bề mặt nhẵn/sần/có lông/bóng, dạng mọc chùm/cụm/đơn lẻ/dây leo/bụi, vị trí trái/phải/trên/dưới/giữa, kích thước tương đối, bộ phận nổi bật nhất.
- Bộ câu hỏi phải đa dạng: yes/no dựa trên ảnh, đếm số lượng nếu nhìn rõ, nhận dạng bộ phận, thuộc tính trực quan, quan hệ không gian/vị trí, so sánh kích thước tương đối.
- Với câu hỏi đếm số lượng, chỉ hỏi khi có thể đếm được tương đối rõ từ ảnh; nếu ảnh không phù hợp thì thay bằng câu hỏi thuộc tính hoặc không gian.
- Không hỏi về tên khoa học, họ thực vật, hoạt chất, cách dùng, liều dùng hoặc thông tin không nhìn thấy trong ảnh. Chỉ được hỏi tên dược liệu thông thường đã cung cấp và đúng 1 câu về chức năng/công dụng dựa trên nội dung web.
- Không trộn thông tin giữa các ID.
- Không bịa chi tiết không quan sát được trong ảnh.
- Câu trả lời ngắn gọn, trực tiếp, dựa trên ảnh.
- Trả về JSON thuần, không markdown.

Thông tin từng mẫu:
{joined_blocks}

Schema JSON bắt buộc:
{{
  "items": [
    {{
      "id": "P000000001",
      "qa": [
        {{
          "question": "...",
          "answer": "..."
        }}
      ]
    }}
  ]
}}
""".strip()


def build_topup_prompt(sample: dict[str, Any], existing_qa: list[dict[str, str]], needed: int) -> str:
    item = sample["item"]
    existing_text = json.dumps(existing_qa, ensure_ascii=False, indent=2)
    return f"""
Bạn là chuyên gia tạo dữ liệu VQA tiếng Việt về dược liệu, tập trung vào những gì nhìn thấy trong ảnh.

Ảnh và thông tin thuộc mẫu:
id: {item["id"]}
tên dược liệu: {item["name"]}
nội dung web để trả lời câu công dụng:
{sample["web_context"].get("text", "")}

Đã có các Q&A sau, không được lặp lại ý hỏi:
{existing_text}

Hãy tạo thêm đúng {needed} cặp câu hỏi/câu trả lời còn thiếu. Chỉ trả về {needed} cặp, không trả nhiều hơn.
- Chỉ hỏi về đặc điểm trực quan trong ảnh: lá/thân/rễ/hoa/quả, màu sắc, bề mặt, dạng mọc, vị trí trái/phải/trên/dưới/giữa, số lượng nếu rõ, yes/no dựa trên ảnh, bộ phận nổi bật nhất.
- Nếu trong các Q&A đã có chưa có câu hỏi nhận dạng tên dược liệu, hãy thêm 1 câu hỏi kiểu "Tên dược liệu trong ảnh là gì?" và trả lời bằng tên dược liệu đã cung cấp.
- Nếu trong các Q&A đã có chưa có câu hỏi về chức năng/công dụng, hãy thêm 1 câu hỏi có kèm tên dược liệu kiểu "Dược liệu {item["name"]} có công dụng gì theo thông tin được cung cấp?" và chỉ trả lời dựa trên nội dung web.
- Câu hỏi phải đa dạng: yes/no, đếm số lượng nếu ảnh phù hợp, nhận dạng bộ phận, thuộc tính trực quan, không gian/vị trí, so sánh kích thước tương đối.
- Không hỏi về tên khoa học, họ thực vật, hoạt chất, cách dùng, liều dùng hoặc thông tin không nhìn thấy trong ảnh. Chỉ được hỏi tên dược liệu thông thường đã cung cấp và đúng 1 câu về chức năng/công dụng dựa trên nội dung web.
- Viết câu trả lời ngắn gọn, khoảng 1 câu.
- Không bịa chi tiết không quan sát được trong ảnh.
- Trả về JSON thuần, không markdown.

Schema JSON bắt buộc:
{{
  "qa": [
    {{
      "question": "...",
      "answer": "..."
    }}
  ]
}}
""".strip()


def parse_json_response(text: str) -> Any:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def normalize_key(key: str) -> str:
    key = unicodedata.normalize("NFD", key)
    key = "".join(ch for ch in key if unicodedata.category(ch) != "Mn")
    key = key.lower()
    key = re.sub(r"[^a-z0-9]+", "_", key)
    return key.strip("_")


def get_qa_value(row: dict[str, Any], wanted: str) -> Any:
    aliases = {
        "question": {"question", "q", "cau_hoi", "hoi", "prompt"},
        "answer": {"answer", "a", "cau_tra_loi", "tra_loi", "response"},
    }

    for key, value in row.items():
        normalized = normalize_key(str(key))
        if normalized in aliases[wanted]:
            return value
    return None


def normalize_model_response(
    parsed: Any,
    item: dict[str, Any],
    image_path: Path,
    web_context: dict[str, str],
) -> dict[str, Any]:
    if isinstance(parsed, list):
        result = {"qa": parsed}
    elif isinstance(parsed, dict):
        result = parsed
    else:
        raise ValueError(f"Model trả về JSON không hợp lệ: {type(parsed).__name__}")

    if "qa" not in result:
        for key in ("qas", "questions", "items", "data"):
            if isinstance(result.get(key), list):
                result["qa"] = result[key]
                break

    if not isinstance(result.get("qa"), list):
        raise ValueError("Model không trả về danh sách qa")

    normalized_qa = []
    for row in result["qa"][:QA_PER_IMAGE]:
        if not isinstance(row, dict):
            continue
        question = get_qa_value(row, "question")
        answer = get_qa_value(row, "answer")
        if question and answer:
            normalized_qa.append({"question": str(question), "answer": str(answer)})

    return {
        "id": item["id"],
        "name": item["name"],
        "source_url": web_context.get("url", ""),
        "image_path": str(image_path),
        "qa": normalized_qa,
    }


def ensure_qa_count(row: dict[str, Any], expected: int = QA_PER_IMAGE) -> None:
    count = len(row.get("qa", []))
    if count != expected:
        raise ValueError(f"Model trả về {count} Q&A hợp lệ, cần đúng {expected}")


def normalize_batch_response(parsed: Any, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sample_by_id = {sample["item"]["id"]: sample for sample in samples}

    if isinstance(parsed, list):
        items = parsed
    elif isinstance(parsed, dict) and isinstance(parsed.get("items"), list):
        items = parsed["items"]
    elif isinstance(parsed, dict) and len(samples) == 1:
        return [normalize_model_response(parsed, samples[0]["item"], samples[0]["image_path"], samples[0]["web_context"])]
    else:
        raise ValueError("Model không trả về danh sách items")

    rows = []
    seen_ids = set()
    for raw_item in items:
        if not isinstance(raw_item, dict):
            continue

        image_id = str(raw_item.get("id", ""))
        sample = sample_by_id.get(image_id)
        if not sample:
            continue

        row = normalize_model_response(
            raw_item,
            sample["item"],
            sample["image_path"],
            sample["web_context"],
        )
        rows.append(row)
        seen_ids.add(image_id)

    missing_ids = sorted(set(sample_by_id) - seen_ids)
    if missing_ids:
        raise ValueError(f"Model thiếu kết quả cho ID: {', '.join(missing_ids)}")

    return rows


def is_retryable_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "deadline_exceeded",
            "504",
            "503",
            "500",
            "429",
            "resource_exhausted",
            "unavailable",
            "high demand",
            "temporarily unavailable",
            "timeout",
            "getaddrinfo failed",
            "errno 11001",
            "name resolution",
            "không trả về",
            "model trả về",
            "json",
        )
    )


def retry_delay_seconds(exc: Exception, attempt: int, base_sleep: float) -> float:
    text = str(exc).lower()
    if "high demand" in text or "503" in text or "unavailable" in text:
        return max(30.0, base_sleep) * (attempt + 1) + random.uniform(0, 5.0)
    if "getaddrinfo" in text or "errno 11001" in text:
        return max(15.0, base_sleep) * (attempt + 1) + random.uniform(0, 3.0)
    return base_sleep * (2**attempt) + random.uniform(0, 2.0)


def call_with_retries(func, retries: int, base_sleep: float):
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return func()
        except Exception as exc:
            last_exc = exc
            if attempt >= retries or not is_retryable_error(exc):
                raise
            delay = retry_delay_seconds(exc, attempt, base_sleep)
            print(f"Lỗi tạm thời: {exc}. Thử lại sau {delay:.1f}s...")
            time.sleep(delay)
    raise last_exc


def generate_one(
    client: genai.Client,
    model: str,
    item: dict[str, Any],
    image_path: Path,
    web_context: dict[str, str],
) -> dict[str, Any]:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    image_part = types.Part.from_bytes(data=image_path.read_bytes(), mime_type=mime_type)
    prompt = build_prompt(item, web_context)

    response = client.models.generate_content(
        model=model,
        contents=[image_part, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=QA_RESPONSE_SCHEMA,
            temperature=0.2,
        ),
    )

    parsed = parse_json_response(response.text or "{}")
    row = normalize_model_response(parsed, item, image_path, web_context)
    ensure_qa_count(row)
    return row


def generate_topup(
    client: genai.Client,
    model: str,
    sample: dict[str, Any],
    existing_qa: list[dict[str, str]],
    needed: int,
) -> list[dict[str, str]]:
    image_path = sample["image_path"]
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    image_part = types.Part.from_bytes(data=image_path.read_bytes(), mime_type=mime_type)
    prompt = build_topup_prompt(sample, existing_qa, needed)

    response = client.models.generate_content(
        model=model,
        contents=[image_part, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
        ),
    )

    row = normalize_model_response(
        parse_json_response(response.text or "{}"),
        sample["item"],
        sample["image_path"],
        sample["web_context"],
    )
    if not row["qa"]:
        raise ValueError("Top-up không trả về Q&A hợp lệ")
    return row["qa"][:needed]


def generate_sample_chunked(
    client: genai.Client,
    model: str,
    sample: dict[str, Any],
    chunk_size: int,
    min_chunk_size: int,
    retries: int,
    retry_sleep: float,
) -> dict[str, Any]:
    qa: list[dict[str, str]] = []
    current_chunk_size = max(min_chunk_size, chunk_size)
    while len(qa) < QA_PER_IMAGE:
        remaining = QA_PER_IMAGE - len(qa)
        needed = min(current_chunk_size, remaining)
        print(f"Sinh {needed} Q&A cho {sample['item']['id']} ({len(qa)}/{QA_PER_IMAGE} đã có)")
        try:
            extra_qa = call_with_retries(
                lambda: generate_topup(client, model, sample, qa, needed),
                retries,
                retry_sleep,
            )
        except Exception:
            if needed > min_chunk_size:
                current_chunk_size = min_chunk_size
                print(f"Hạ tải {sample['item']['id']}: chuyển sang {min_chunk_size} Q&A/request")
                continue
            raise
        qa.extend(extra_qa)

    return {
        "id": sample["item"]["id"],
        "name": sample["item"]["name"],
        "source_url": sample["web_context"].get("url", ""),
        "image_path": str(sample["image_path"]),
        "qa": qa[:QA_PER_IMAGE],
    }


def generate_batch(
    client: genai.Client,
    model: str,
    samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    contents: list[Any] = []
    for sample in samples:
        image_path = sample["image_path"]
        mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
        contents.append(f"Ảnh của mẫu ID {sample['item']['id']} - {sample['item']['name']}:")
        contents.append(types.Part.from_bytes(data=image_path.read_bytes(), mime_type=mime_type))
    contents.append(build_batch_prompt(samples))

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=BATCH_RESPONSE_SCHEMA,
            temperature=0.2,
        ),
    )

    parsed = parse_json_response(response.text or "{}")
    return normalize_batch_response(parsed, samples)


def load_existing(output_path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    if not output_path.exists():
        return [], set()

    with output_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data, {str(row.get("id")) for row in data}


def save_output(output_path: Path, rows: list[dict[str, Any]]) -> None:
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    tmp_path.replace(output_path)


def chunks(rows: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [rows[index : index + size] for index in range(0, len(rows), size)]


def item_number(image_id: str) -> int:
    match = re.search(r"\d+", image_id)
    if not match:
        raise ValueError(f"ID không hợp lệ: {image_id}")
    return int(match.group(0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="herb_images.json")
    parser.add_argument("--images-dir", default="herb_images")
    parser.add_argument("--output", default="herb_qa.json")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-id", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context-chars", type=int, default=1200)
    parser.add_argument("--api-timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--retry-sleep", type=float, default=10.0)
    parser.add_argument("--qa-chunk-size", type=int, default=5)
    parser.add_argument("--min-qa-per-request", type=int, default=5)
    parser.add_argument("--no-topup", action="store_true")
    parser.add_argument("--sleep", type=float, default=3.0)
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Thiếu GEMINI_API_KEY. Hãy set biến môi trường trước khi chạy.")

    input_path = Path(args.input)
    images_dir = Path(args.images_dir)
    output_path = Path(args.output)

    with input_path.open("r", encoding="utf-8") as f:
        items = json.load(f)
    if args.start_id:
        start_number = item_number(args.start_id)
        items = [item for item in items if item_number(item["id"]) >= start_number]
    if args.limit:
        items = items[: args.limit]
    if args.min_qa_per_request < 1:
        raise SystemExit("--min-qa-per-request phải >= 1")
    if args.qa_chunk_size < args.min_qa_per_request:
        raise SystemExit("--qa-chunk-size phải >= --min-qa-per-request")

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=args.api_timeout * 1000),
    )
    output_rows, done_ids = load_existing(output_path)

    pending_items = []
    for item in items:
        if item["id"] in done_ids:
            print(f"Skip {item['id']}: đã có trong {output_path}")
        else:
            pending_items.append(item)

    if args.batch_size != 1:
        print("Ghi chú: script đang ưu tiên an toàn API, mỗi request chỉ chạy 1 ảnh. Đã ép --batch-size về 1.")
    for batch_items in chunks(pending_items, 1):
        samples = []
        for item in batch_items:
            image_id = item["id"]
            image_path = find_image_path(images_dir, image_id)
            if not image_path:
                print(f"Skip {image_id}: không tìm thấy ảnh")
                continue

            print(f"Chuẩn bị {image_id} - {item['name']}")
            web_context = fetch_herb_web_context(item["name"], args.context_chars)
            samples.append(
                {
                    "item": item,
                    "image_path": image_path,
                    "web_context": web_context,
                }
            )

        if not samples:
            continue

        sample_ids = ", ".join(sample["item"]["id"] for sample in samples)
        if len(samples) == 1:
            print(f"Generate chunked ({sample_ids})")
            try:
                results = [
                    generate_sample_chunked(
                        client,
                        args.model,
                        samples[0],
                        max(1, min(args.qa_chunk_size, QA_PER_IMAGE)),
                        max(1, min(args.min_qa_per_request, QA_PER_IMAGE)),
                        args.retries,
                        args.retry_sleep,
                    )
                ]
            except Exception as exc:
                print(f"Lỗi {sample_ids}: {exc}")
                continue
        else:
            print(f"Generate batch ({len(samples)} ảnh): {sample_ids}")
            try:
                results = call_with_retries(
                    lambda: generate_batch(client, args.model, samples),
                    args.retries,
                    args.retry_sleep,
                )
            except Exception as exc:
                print(f"Lỗi batch {sample_ids}: {exc}")
                print("Fallback: chuyển sang sinh từng ảnh theo từng cụm nhỏ.")
                results = []
                for sample in samples:
                    try:
                        results.append(
                            generate_sample_chunked(
                                client,
                                args.model,
                                sample,
                                max(1, min(args.qa_chunk_size, QA_PER_IMAGE)),
                                max(1, min(args.min_qa_per_request, QA_PER_IMAGE)),
                                args.retries,
                                args.retry_sleep,
                            )
                        )
                    except Exception as sample_exc:
                        print(f"Lỗi {sample['item']['id']}: {sample_exc}")

        for result in results:
            if len(result["qa"]) < QA_PER_IMAGE and not args.no_topup:
                needed = QA_PER_IMAGE - len(result["qa"])
                sample = next(sample for sample in samples if sample["item"]["id"] == result["id"])
                print(f"Bổ sung {needed} Q&A cho {result['id']}")
                try:
                    extra_qa = call_with_retries(
                        lambda: generate_topup(client, args.model, sample, result["qa"], needed),
                        args.retries,
                        args.retry_sleep,
                    )
                    result["qa"].extend(extra_qa)
                except Exception as exc:
                    print(f"Lỗi bổ sung {result['id']}: {exc}")

            if len(result["qa"]) != QA_PER_IMAGE:
                print(f"Skip {result['id']}: chỉ có {len(result['qa'])}/{QA_PER_IMAGE} Q&A")
                continue

            output_rows.append(result)
            done_ids.add(result["id"])
        save_output(output_path, output_rows)
        time.sleep(args.sleep)

    print(f"Đã lưu {len(output_rows)} mẫu vào {output_path}")


if __name__ == "__main__":
    main()
