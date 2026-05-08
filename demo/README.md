# Vietnamese Medicinal Herb VQA Demo

Demo gồm hai phần:

- Colab chạy model thật bằng Gradio API.
- FastAPI local hiển thị giao diện, upload ảnh/câu hỏi và gọi Gradio API.

## 1. Chạy model trên Colab

Mở notebook:

```text
demo/colab/qwen25_vl_finetuned_gradio_server.ipynb
```

Notebook sẽ load:

```text
Qwen/Qwen2.5-VL-3B-Instruct
final_finetuned_adapter trong Google Drive
```

Sau khi chạy cell cuối, Colab in ra URL dạng:

```text
https://xxxx.gradio.live
```

Copy URL đó.

## 2. Cấu hình FastAPI local

Trong thư mục `demo`, tạo file `.env` từ `.env.example`:

```powershell
Copy-Item .env.example .env
```

Sửa:

```env
GRADIO_API_URL=https://xxxx.gradio.live
GRADIO_API_NAME=/predict
```

## 3. Chạy giao diện

```powershell
cd D:\DeepLearning\CK\demo
.\run.ps1
```

Mở:

```text
http://127.0.0.1:8000
```

## Ghi chú

Demo đã bỏ attention heatmap. Giao diện chỉ giữ chức năng:

- Upload ảnh.
- Nhập câu hỏi tiếng Việt.
- Gọi model fine-tuned trên Colab.
- Hiển thị câu trả lời.
