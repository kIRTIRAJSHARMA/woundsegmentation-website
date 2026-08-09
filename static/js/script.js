const fileInput      = document.getElementById("fileInput");
const dropZone       = document.getElementById("dropZone");
const dropContent    = document.getElementById("dropContent");
const preview        = document.getElementById("preview");
const analyzeBtn     = document.getElementById("analyzeBtn");
const spinner        = document.getElementById("spinner");
const statusText     = document.getElementById("statusText");
const resultsSection = document.getElementById("resultsSection");

/* ── About drawer ─────────────────────────────────────────── */

const aboutBtn       = document.getElementById("aboutBtn");
const aboutDrawer    = document.getElementById("aboutDrawer");
const drawerBackdrop = document.getElementById("drawerBackdrop");
const drawerClose    = document.getElementById("drawerClose");

function openDrawer() {
  aboutDrawer.classList.add("open");
  drawerBackdrop.classList.add("open");
  aboutDrawer.setAttribute("aria-hidden", "false");
  document.body.style.overflow = "hidden";
  drawerClose.focus();
}

function closeDrawer() {
  aboutDrawer.classList.remove("open");
  drawerBackdrop.classList.remove("open");
  aboutDrawer.setAttribute("aria-hidden", "true");
  document.body.style.overflow = "";
  aboutBtn.focus();
}

aboutBtn.addEventListener("click", openDrawer);
drawerClose.addEventListener("click", closeDrawer);
drawerBackdrop.addEventListener("click", closeDrawer);

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && aboutDrawer.classList.contains("open")) {
    closeDrawer();
  }
});

/* ── File upload ──────────────────────────────────────────── */

let selectedFile = null;

dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("drag-over");
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  if (e.dataTransfer.files.length) {
    handleFile(e.dataTransfer.files[0]);
  }
});

fileInput.addEventListener("change", () => {
  if (fileInput.files.length) {
    handleFile(fileInput.files[0]);
  }
});

function handleFile(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    preview.src = e.target.result;
    preview.style.display = "block";
    dropContent.style.display = "none";
  };
  reader.readAsDataURL(file);
  analyzeBtn.disabled = false;
  statusText.textContent = "";
}

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  analyzeBtn.disabled = true;
  spinner.classList.add("active");
  statusText.textContent = "Running model inference...";
  resultsSection.style.display = "none";

  const formData = new FormData();
  formData.append("image", selectedFile);

  try {
    const res = await fetch("https://woundsegmentation-chwb.onrender.com/predict", {
      method: "POST",
      body: formData,
    });

    const text = await res.text();
    let data;
    try {
      data = JSON.parse(text);
    } catch (_) {
      throw new Error(`Server returned non-JSON response (HTTP ${res.status})`);
    }

    if (!res.ok) {
      throw new Error(data.error || "Prediction failed");
    }

    document.getElementById("imgOriginal").src = data.original_url;
    document.getElementById("imgOverlay").src  = data.overlay_url;
    document.getElementById("imgHeatmap").src  = data.heatmap_url;
    document.getElementById("imgMask").src     = data.mask_url;
    document.getElementById("areaValue").textContent = data.wound_area_pct + "%";

    resultsSection.style.display = "block";
    spinner.classList.remove("active");
    statusText.textContent = "Analysis complete.";
  } catch (err) {
    spinner.classList.remove("active");
    statusText.textContent = "Error: " + err.message;
  } finally {
    analyzeBtn.disabled = false;
  }
});
