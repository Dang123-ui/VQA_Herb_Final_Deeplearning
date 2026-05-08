const imageInput = document.getElementById("image-input");
const imagePreview = document.getElementById("image-preview");
const questionInput = document.getElementById("question-input");
const predictForm = document.getElementById("predict-form");
const answerOutput = document.getElementById("answer-output");
const inferenceOutput = document.getElementById("inference-output");
const statusOutput = document.getElementById("status-output");
const submitButton = document.getElementById("submit-button");

const sampleQuestionChips = document.querySelectorAll(".sample-question-chip");

sampleQuestionChips.forEach((chip) => {
  chip.addEventListener("click", () => {
    questionInput.value = chip.dataset.question || "";
    questionInput.focus();
  });
});

imageInput.addEventListener("change", () => {
  const file = imageInput.files?.[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (event) => {
    imagePreview.src = event.target.result;
  };
  reader.readAsDataURL(file);
});

predictForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const file = imageInput.files?.[0];
  const question = questionInput.value.trim();

  if (!file) {
    statusOutput.textContent = "Vui lòng chọn ảnh.";
    return;
  }

  if (!question) {
    statusOutput.textContent = "Vui lòng nhập câu hỏi.";
    return;
  }

  const formData = new FormData();
  formData.append("image", file);
  formData.append("question", question);

  try {
    submitButton.disabled = true;
    submitButton.classList.add("opacity-70", "cursor-not-allowed");
    statusOutput.textContent = "Đang gọi mô hình trên Colab...";
    answerOutput.textContent = "Đang suy luận...";
    inferenceOutput.textContent = "...";

    const response = await fetch("/api/v1/predict", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const payload = await response.json().catch(() => null);
      throw new Error(payload?.detail || "API request failed");
    }

    const data = await response.json();
    answerOutput.textContent = data.answer || "Không có câu trả lời.";
    inferenceOutput.textContent = `${data.inference_ms ?? "--"} ms`;
    statusOutput.textContent = "Suy luận thành công.";
  } catch (error) {
    console.error(error);
    answerOutput.textContent = "Lỗi";
    inferenceOutput.textContent = "-- ms";
    statusOutput.textContent = error.message || "Không thể gọi mô hình. Kiểm tra Gradio URL trong .env.";
  } finally {
    submitButton.disabled = false;
    submitButton.classList.remove("opacity-70", "cursor-not-allowed");
  }
});
