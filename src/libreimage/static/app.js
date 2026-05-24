const defaults = {
  restorePrompt: "Restore the image naturally. Repair damage, remove artifacts, preserve identity, texture, lighting, and composition.",
  negative: "text, watermark, logo, plastic skin, oversharpening, distorted geometry, extra objects",
  inpaintPrompt: "Natural invisible repair matching the surrounding image.",
};

const state = {
  sessionId: null,
  images: [],
  selectedId: null,
  stage: "upload",
  finalIndex: 0,
  mask: {
    source: null,
    image: null,
    overlay: document.createElement("canvas"),
    painting: false,
    mode: "paint",
    last: null,
    scale: 1,
    offsetX: 0,
    offsetY: 0,
  },
};

const $ = (id) => document.getElementById(id);
const statusEl = $("status");

function imageUrl(image) {
  return `/api/session/${state.sessionId}/images/${image.id}`;
}

function setStatus(text, error = false) {
  statusEl.textContent = normalizeStatus(text);
  statusEl.style.color = error ? "var(--danger)" : "var(--muted)";
}

function normalizeStatus(text) {
  const trimmed = String(text || "").trim().replace(/[.。]+$/u, "");
  if (!trimmed) return "";
  return trimmed.charAt(0).toLocaleUpperCase() + trimmed.slice(1);
}

function selectedImage() {
  return state.images.find((image) => image.id === state.selectedId) || state.images.at(-1) || null;
}

function setStage(stage) {
  state.stage = stage;
  if (stage === "final") {
    const selectedIndex = state.images.findIndex((image) => image.id === state.selectedId);
    if (selectedIndex >= 0) state.finalIndex = selectedIndex;
  }
  document.querySelectorAll(".view").forEach((view) => view.classList.toggle("active", view.id === stage));
  document.querySelectorAll(".step").forEach((button) => button.classList.toggle("active", button.dataset.stage === stage));
  render();
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || "Request failed.");
  return payload;
}

async function startSession() {
  const payload = await api("/api/session", { method: "POST" });
  state.sessionId = payload.session_id;
  state.images = payload.images;
  state.selectedId = state.images.at(-1)?.id || null;
  setStatus("Session ready.");
  render();
}

async function uploadFile(file) {
  if (!file) return;
  const form = new FormData();
  form.append("image", file, file.name);
  setStatus("Uploading image...");
  const payload = await api(`/api/session/${state.sessionId}/upload`, { method: "POST", body: form });
  state.images = payload.images;
  state.selectedId = payload.image.id;
  setStatus("Image added.");
  setStage("restore");
}

function render() {
  renderGrids();
  renderStageImages();
  drawMask();
}

function renderGrids() {
  renderGrid($("uploadGrid"), state.images);
  renderFilmstrip($("restoreGallery"), stageCandidates(["upload", "restore"]));
  renderFilmstrip($("inpaintGallery"), stageCandidates(["upload", "restore", "inpaint"]));
  renderFilmstrip($("sharpenGallery"), stageCandidates(["upload", "restore", "inpaint", "sharpen"]));
  renderFilmstrip($("finalGallery"), state.images);
}

function stageCandidates(stages) {
  return state.images.filter((image) => stages.includes(image.stage));
}

function renderGrid(container, images) {
  container.innerHTML = "";
  images.forEach((image) => container.appendChild(thumb(image)));
}

function renderFilmstrip(container, images) {
  container.innerHTML = "";
  images.forEach((image) => container.appendChild(thumb(image)));
}

function thumb(image) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = `thumb${image.id === state.selectedId ? " selected" : ""}`;
  button.innerHTML = `
    <img src="${imageUrl(image)}" alt="">
    <span class="thumb-title"><span>${image.stage}</span><span>${image.width}x${image.height}</span></span>
  `;
  button.addEventListener("click", () => {
    selectImage(image);
  });
  return button;
}

function selectImage(image) {
  state.selectedId = image.id;
  if (state.stage === "final") {
    state.finalIndex = Math.max(0, state.images.findIndex((item) => item.id === image.id));
  }
  if (state.stage === "inpaint") prepareMask(image);
  render();
}

function setFinalIndex(index) {
  state.finalIndex = Math.max(0, Math.min(state.images.length - 1, index));
  const image = state.images[state.finalIndex];
  if (image) state.selectedId = image.id;
  render();
}

function renderStageImages() {
  const image = selectedImage();
  const url = image ? imageUrl(image) : "";
  $("restoreImage").src = url;
  setImageAspect($("restoreShell"), image);
  $("inpaintResult").src = url;
  $("sharpenSource").src = url;
  $("sharpenResult").src = latestByStage("sharpen") ? imageUrl(latestByStage("sharpen")) : url;
  renderFinal();
  if (state.stage === "inpaint" && image && state.mask.source !== image.id) prepareMask(image);
}

function setImageAspect(element, image) {
  if (!element || !image || !image.width || !image.height) {
    element?.style.removeProperty("--image-aspect");
    return;
  }
  element.style.setProperty("--image-aspect", String(image.width / image.height));
}

function latestByStage(stage) {
  return [...state.images].reverse().find((image) => image.stage === stage);
}

function renderFinal() {
  if (!state.images.length) {
    $("finalImage").removeAttribute("src");
    $("finalTitle").textContent = "No image selected";
    return;
  }
  state.finalIndex = Math.max(0, Math.min(state.finalIndex, state.images.length - 1));
  const image = state.images[state.finalIndex];
  $("finalImage").src = imageUrl(image);
  $("finalTitle").textContent = `${image.stage} - ${image.width}x${image.height}`;
  state.selectedId = state.stage === "final" ? image.id : state.selectedId;
}

async function runRestore() {
  const image = selectedImage();
  if (!image) return setStatus("Upload an image first.", true);
  const form = new FormData();
  appendCommon(form, "restore", image.id);
  await runGeneration("restore", `/api/session/${state.sessionId}/restore`, form);
}

async function runInpaint() {
  const image = selectedImage();
  if (!image) return setStatus("Select an image first.", true);
  const maskBlob = await new Promise((resolve) => state.mask.overlay.toBlob(resolve, "image/png"));
  const form = new FormData();
  appendCommon(form, "inpaint", image.id);
  form.append("mask", maskBlob, "mask.png");
  await runGeneration("inpaint", `/api/session/${state.sessionId}/inpaint`, form);
}

async function runSharpen() {
  const image = selectedImage();
  if (!image) return setStatus("Select an image first.", true);
  const form = new FormData();
  form.append("image_id", image.id);
  form.append("radius", $("sharpRadius").value);
  form.append("amount", $("sharpAmount").value);
  form.append("threshold", $("sharpThreshold").value);
  form.append("contrast", $("sharpContrast").value);
  form.append("color", $("sharpColor").value);
  await runGeneration("sharpen", `/api/session/${state.sessionId}/sharpen`, form);
}

function appendCommon(form, prefix, imageId) {
  form.append("image_id", imageId);
  form.append("prompt", $(`${prefix}Prompt`).value);
  form.append("negative_prompt", $(`${prefix}Negative`).value);
  form.append("steps", $(`${prefix}Steps`).value);
  form.append("guidance_scale", $(`${prefix}Guidance`).value);
  form.append("strength", $(`${prefix}Strength`).value);
  if ($(`${prefix}Lora`)) form.append("lora_scale", $(`${prefix}Lora`).value);
  form.append("seed", $(`${prefix}Seed`).value);
}

async function runGeneration(stage, url, form) {
  const button = stage === "restore" ? $("runRestore") : stage === "inpaint" ? $("runInpaint") : $("runSharpen");
  button.disabled = true;
  setStatus(`Running ${stage}. Model loading can take a while on the first pass.`);
  try {
    const payload = await api(url, { method: "POST", body: form });
    state.images = payload.images;
    state.selectedId = payload.image.id;
    state.finalIndex = state.images.findIndex((image) => image.id === payload.image.id);
    setStatus(`${stage} result saved to the session library.`);
    render();
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    button.disabled = false;
  }
}

async function deleteSelected() {
  const image = selectedImage();
  if (!image) return;
  await api(`/api/session/${state.sessionId}/images/${image.id}`, { method: "DELETE" });
  state.images = state.images.filter((item) => item.id !== image.id);
  state.selectedId = state.images.at(-1)?.id || null;
  state.finalIndex = Math.min(state.finalIndex, Math.max(0, state.images.length - 1));
  setStatus("Image deleted from the session library.");
  render();
}

function prepareMask(image) {
  state.mask.source = image.id;
  const img = new Image();
  img.onload = () => {
    state.mask.image = img;
    state.mask.overlay.width = img.naturalWidth;
    state.mask.overlay.height = img.naturalHeight;
    const ctx = state.mask.overlay.getContext("2d");
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, state.mask.overlay.width, state.mask.overlay.height);
    drawMask();
  };
  img.src = imageUrl(image);
}

function drawMask() {
  const canvas = $("maskCanvas");
  if (!canvas || !state.mask.image) return;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width));
  canvas.height = Math.max(1, Math.floor(rect.height));
  const ctx = canvas.getContext("2d");
  const image = state.mask.image;
  const scale = Math.min(canvas.width / image.naturalWidth, canvas.height / image.naturalHeight);
  const drawW = Math.round(image.naturalWidth * scale);
  const drawH = Math.round(image.naturalHeight * scale);
  state.mask.scale = scale;
  state.mask.offsetX = Math.floor((canvas.width - drawW) / 2);
  state.mask.offsetY = Math.floor((canvas.height - drawH) / 2);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(image, state.mask.offsetX, state.mask.offsetY, drawW, drawH);
  ctx.save();
  ctx.globalAlpha = 0.44;
  ctx.drawImage(state.mask.overlay, state.mask.offsetX, state.mask.offsetY, drawW, drawH);
  ctx.globalCompositeOperation = "source-atop";
  ctx.fillStyle = "#d8ff75";
  ctx.fillRect(state.mask.offsetX, state.mask.offsetY, drawW, drawH);
  ctx.restore();
}

function pointerToImage(event) {
  const canvas = $("maskCanvas");
  const rect = canvas.getBoundingClientRect();
  const x = (event.clientX - rect.left - state.mask.offsetX) / state.mask.scale;
  const y = (event.clientY - rect.top - state.mask.offsetY) / state.mask.scale;
  if (!state.mask.image || x < 0 || y < 0 || x >= state.mask.image.naturalWidth || y >= state.mask.image.naturalHeight) return null;
  return { x, y };
}

function stroke(a, b) {
  if (!a || !b) return;
  const ctx = state.mask.overlay.getContext("2d");
  const width = Number($("brushSize").value);
  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.lineWidth = width;
  ctx.strokeStyle = state.mask.mode === "paint" ? "#fff" : "#000";
  ctx.fillStyle = ctx.strokeStyle;
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(b.x, b.y, width / 2, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
  drawMask();
}

async function saveImages(images) {
  if (!images.length) return;
  if ("showDirectoryPicker" in window) {
    const dir = await window.showDirectoryPicker();
    for (const image of images) {
      const handle = await dir.getFileHandle(`${image.stage}-${image.id}.png`, { create: true });
      const writable = await handle.createWritable();
      const blob = await fetch(imageUrl(image)).then((response) => response.blob());
      await writable.write(blob);
      await writable.close();
    }
    setStatus("Saved selected image files.");
    return;
  }
  images.forEach((image) => {
    const anchor = document.createElement("a");
    anchor.href = imageUrl(image);
    anchor.download = `${image.stage}-${image.id}.png`;
    anchor.click();
  });
  setStatus("Browser folder picker unavailable; downloaded images instead.");
}

function bindEvents() {
  $("restorePrompt").value = defaults.restorePrompt;
  $("restoreNegative").value = defaults.negative;
  $("inpaintPrompt").value = defaults.inpaintPrompt;
  $("inpaintNegative").value = defaults.negative;

  document.querySelectorAll(".step").forEach((button) => button.addEventListener("click", () => setStage(button.dataset.stage)));
  $("uploadInput").addEventListener("change", (event) => uploadFile(event.target.files[0]));
  document.querySelector(".dropzone").addEventListener("dragover", (event) => event.preventDefault());
  document.querySelector(".dropzone").addEventListener("drop", (event) => {
    event.preventDefault();
    uploadFile(event.dataTransfer.files[0]);
  });
  $("runRestore").addEventListener("click", runRestore);
  $("restoreUseLatest").addEventListener("click", () => setStage("inpaint"));
  $("runInpaint").addEventListener("click", runInpaint);
  $("runSharpen").addEventListener("click", runSharpen);
  document.querySelectorAll("[data-delete-current]").forEach((button) => button.addEventListener("click", deleteSelected));
  $("prevImage").addEventListener("click", () => setFinalIndex(state.finalIndex - 1));
  $("nextImage").addEventListener("click", () => setFinalIndex(state.finalIndex + 1));
  $("saveCurrent").addEventListener("click", () => saveImages(selectedImage() ? [selectedImage()] : []));
  $("saveAll").addEventListener("click", () => saveImages(state.images));
  $("restartFromFinal").addEventListener("click", () => setStage("restore"));

  $("brushBtn").addEventListener("click", () => setMaskMode("paint"));
  $("eraseBtn").addEventListener("click", () => setMaskMode("erase"));
  $("clearMask").addEventListener("click", () => {
    const ctx = state.mask.overlay.getContext("2d");
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, state.mask.overlay.width, state.mask.overlay.height);
    drawMask();
  });
  bindCanvas();
  window.addEventListener("resize", drawMask);
  window.addEventListener("keydown", (event) => {
    if (state.stage !== "final") return;
    if (event.key === "ArrowLeft") setFinalIndex(state.finalIndex - 1);
    if (event.key === "ArrowRight") setFinalIndex(state.finalIndex + 1);
  });
}

function setMaskMode(mode) {
  state.mask.mode = mode;
  $("brushBtn").classList.toggle("active", mode === "paint");
  $("eraseBtn").classList.toggle("active", mode === "erase");
}

function bindCanvas() {
  const canvas = $("maskCanvas");
  canvas.addEventListener("pointerdown", (event) => {
    const point = pointerToImage(event);
    if (!point) return;
    state.mask.painting = true;
    state.mask.last = point;
    canvas.setPointerCapture(event.pointerId);
    stroke(point, point);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!state.mask.painting) return;
    const point = pointerToImage(event);
    if (!point) return;
    stroke(state.mask.last, point);
    state.mask.last = point;
  });
  canvas.addEventListener("pointerup", () => { state.mask.painting = false; state.mask.last = null; });
  canvas.addEventListener("pointercancel", () => { state.mask.painting = false; state.mask.last = null; });
}

bindEvents();
startSession().catch((error) => setStatus(error.message, true));
