const API_BASE = "";
const API_KEY = "super-secret-key";

const logEl = document.getElementById("log");
const resultEl = document.getElementById("result");

const checkStatusBtn = document.getElementById("checkStatusBtn");
const clearBtn = document.getElementById("clearBtn");
const uploadBtn = document.getElementById("uploadBtn");
const getTaskBtn = document.getElementById("getTaskBtn");

const filesInput = document.getElementById("filesInput");
const taskIdInput = document.getElementById("taskIdInput");

let annotatedObjectUrls = [];

function apiUrl(path) {
  return `${API_BASE}${path}`;
}

function apiHeaders(extra = {}) {
  return {
    "X-API-Key": API_KEY,
    ...extra,
  };
}

function revokeAnnotatedObjectUrls() {
  for (const objectUrl of annotatedObjectUrls) {
    URL.revokeObjectURL(objectUrl);
  }

  annotatedObjectUrls = [];
}

function log(value) {
  const text = typeof value === "string"
    ? value
    : JSON.stringify(value, null, 2);

  logEl.textContent = `${new Date().toLocaleTimeString()} ${text}\n\n${logEl.textContent}`;
}

function setLoading(button, loading, loadingText = "Загрузка...") {
  button.disabled = loading;
  button.dataset.oldText = button.dataset.oldText || button.textContent;
  button.textContent = loading ? loadingText : button.dataset.oldText;
}

async function requestJSON(url, options = {}) {
  const response = await fetch(url, options);
  const text = await response.text();

  let data = null;

  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = { raw: text };
  }

  if (!response.ok) {
    const message = data && data.error ? data.error : `HTTP ${response.status}`;
    throw new Error(message);
  }

  return data;
}

async function checkStatus() {
  setLoading(checkStatusBtn, true);

  try {
    const data = await requestJSON(apiUrl("/api/status"));
    log(data);
    resultEl.innerHTML = `<div class="success">Backend status: ${escapeHtml(data.status)}</div>`;
  } catch (error) {
    log(`ERROR: ${error.message}`);
    resultEl.innerHTML = `<div class="error">${escapeHtml(error.message)}</div>`;
  } finally {
    setLoading(checkStatusBtn, false);
  }
}

async function uploadFiles() {
  const files = Array.from(filesInput.files);

  if (files.length === 0) {
    alert("Выбери один или несколько файлов");
    return;
  }

  const formData = new FormData();

  for (const file of files) {
    formData.append("files", file);
  }

  setLoading(uploadBtn, true, "Отправка...");

  try {
    const data = await requestJSON(apiUrl("/api/tasks/upload"), {
      method: "POST",
      headers: apiHeaders(),
      body: formData,
    });

    taskIdInput.value = data.task_id;
    log(data);

    await pollTask(data.task_id);
  } catch (error) {
    log(`ERROR: ${error.message}`);
    resultEl.innerHTML = `<div class="error">${escapeHtml(error.message)}</div>`;
  } finally {
    setLoading(uploadBtn, false);
  }
}

async function getTask(taskId) {
  const id = (taskId || taskIdInput.value).trim();

  if (!id) {
    alert("Укажи task_id");
    return null;
  }

  const data = await requestJSON(apiUrl(`/api/tasks/${encodeURIComponent(id)}`), {
    method: "GET",
    headers: apiHeaders(),
  });

  log(data);
  renderTask(data);
  await loadAnnotatedImages();

  return data;
}

async function pollTask(taskId) {
  const id = (taskId || taskIdInput.value).trim();

  if (!id) {
    alert("Укажи task_id");
    return;
  }

  setLoading(uploadBtn, true, "Распознавание...");

  try {
    for (let i = 0; i < 60; i++) {
      const data = await getTask(id);

      if (!data) {
        return;
      }

      if (
        data.status === "completed" ||
        data.status === "completed_with_errors" ||
        data.status === "failed"
      ) {
        return;
      }

      await new Promise((resolve) => setTimeout(resolve, 1000));
    }

    log("Опрос остановлен по таймауту");
  } catch (error) {
    log(`ERROR: ${error.message}`);
    resultEl.innerHTML = `<div class="error">${escapeHtml(error.message)}</div>`;
  }
}

function renderTask(task) {
  revokeAnnotatedObjectUrls();

  const images = task.images || [];

  const imagesHtml = images.map((image) => {
    const numbers = image.numbers || [];

    const numbersHtml = numbers.length > 0
      ? numbers.map((number) => `
          <div class="number">${escapeHtml(number.number)}</div>
          <div class="muted small">
            avg_digit_confidence: ${formatNumber(number.avg_digit_confidence)} |
            detect: ${formatNumber(number.detect_confidence)} |
            segment: ${formatNumber(number.segment_confidence)} |
            score: ${formatNumber(number.candidate_score)}
          </div>
        `).join("")
      : `<div class="muted">Номера не найдены</div>`;

    const annotatedHtml = image.annotated_url
      ? `
        <div class="annotated-image-wrap">
          <div class="annotated-loading" data-annotated-loading="${escapeHtml(image.id)}">
            Загрузка обработанного изображения...
          </div>
          <img
            class="annotated-image"
            data-annotated-image="${escapeHtml(image.id)}"
            data-annotated-url="${escapeHtml(image.annotated_url)}"
            alt="Annotated result"
          />
        </div>
      `
      : "";

    return `
      <div class="image-card">
        <div><strong>${escapeHtml(image.external_id || image.id)}</strong></div>
        <div style="margin-top: 6px;">
          <span class="status ${escapeHtml(image.status)}">${escapeHtml(image.status)}</span>
        </div>
        ${image.error ? `<div class="error" style="margin-top: 8px;">${escapeHtml(image.error)}</div>` : ""}
        <div style="margin-top: 10px;">${numbersHtml}</div>
        ${annotatedHtml}
      </div>
    `;
  }).join("");

  resultEl.innerHTML = `
    <div class="task-card">
      <div><strong>Task:</strong> ${escapeHtml(task.id)}</div>
      <div style="margin-top: 8px;">
        <span class="status ${escapeHtml(task.status)}">${escapeHtml(task.status)}</span>
      </div>
      <div class="muted" style="margin-top: 8px;">
        processed: ${task.processed_images}/${task.total_images}
      </div>
      ${task.error ? `<div class="error" style="margin-top: 8px;">${escapeHtml(task.error)}</div>` : ""}
      <div style="margin-top: 12px;">${imagesHtml}</div>
    </div>
  `;
}

async function loadAnnotatedImages() {
  const images = Array.from(document.querySelectorAll("[data-annotated-image]"));

  for (const imageEl of images) {
    const imageId = imageEl.dataset.annotatedImage;
    const annotatedUrl = imageEl.dataset.annotatedUrl;
    const loadingEl = document.querySelector(`[data-annotated-loading="${cssEscape(imageId)}"]`);

    if (!annotatedUrl) {
      continue;
    }

    try {
      const response = await fetch(apiUrl(annotatedUrl), {
        method: "GET",
        headers: apiHeaders(),
      });

      if (!response.ok) {
        let message = `HTTP ${response.status}`;

        try {
          const data = await response.json();
          if (data && data.error) {
            message = data.error;
          }
        } catch {
          // ignore non-json response
        }

        throw new Error(message);
      }

      const blob = await response.blob();
      const objectUrl = URL.createObjectURL(blob);

      annotatedObjectUrls.push(objectUrl);

      imageEl.src = objectUrl;
      imageEl.style.display = "block";

      if (loadingEl) {
        loadingEl.textContent = "";
        loadingEl.style.display = "none";
      }
    } catch (error) {
      if (loadingEl) {
        loadingEl.innerHTML = `<span class="error">Не удалось загрузить обработанное изображение: ${escapeHtml(error.message)}</span>`;
      }

      log(`ANNOTATED IMAGE ERROR: ${error.message}`);
    }
  }
}

function cssEscape(value) {
  if (window.CSS && typeof window.CSS.escape === "function") {
    return window.CSS.escape(value);
  }

  return String(value).replaceAll('"', '\\"');
}

function formatNumber(value) {
  if (typeof value !== "number") {
    return "-";
  }

  return value.toFixed(3);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

checkStatusBtn.addEventListener("click", checkStatus);

clearBtn.addEventListener("click", () => {
  logEl.textContent = "";
  resultEl.innerHTML = "";
  revokeAnnotatedObjectUrls();
});

uploadBtn.addEventListener("click", uploadFiles);

getTaskBtn.addEventListener("click", async () => {
  try {
    await getTask();
  } catch (error) {
    log(`ERROR: ${error.message}`);
    resultEl.innerHTML = `<div class="error">${escapeHtml(error.message)}</div>`;
  }
});

checkStatus();
