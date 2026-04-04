const boot = window.APP_BOOT || { model: "", characters: {} };

const characterTabsEl = document.getElementById("characterTabs");
const siliconeControlsEl = document.getElementById("siliconeControls");
const shiftRangeEl = document.getElementById("shiftRange");
const shiftValueEl = document.getElementById("shiftValue");
const siliconeMetaEl = document.getElementById("siliconeMeta");
const moodBtn = document.getElementById("moodBtn");
const memoryBtn = document.getElementById("memoryBtn");
const chatTitleEl = document.getElementById("chatTitle");
const chatSubtitleEl = document.getElementById("chatSubtitle");
const chatLogEl = document.getElementById("chatLog");
const resetBtn = document.getElementById("resetBtn");
const composerEl = document.getElementById("composer");
const messageInputEl = document.getElementById("messageInput");
const composerStatusEl = document.getElementById("composerStatus");
const imageMediaEl = document.getElementById("imageMedia");
const videoMediaEl = document.getElementById("videoMedia");

let appState = null;

function activeCharacterKey() {
  return appState?.active_character || "silicone_subject";
}

function activeCharacterState() {
  return appState?.characters?.[activeCharacterKey()] || null;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function formatTimestamp(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleString();
}

function describeShift(value) {
  const shift = Number(value || 0);
  const angela = Math.round((1 - shift) * 100);
  const housewife = Math.round(shift * 100);
  return `shift=${shift.toFixed(2)} (Angela ${angela}%, Housewife ${housewife}%)`;
}

function renderCharacterTabs() {
  characterTabsEl.innerHTML = "";
  const characters = appState?.characters || {};

  Object.entries(characters).forEach(([key, character]) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `character-tab${key === activeCharacterKey() ? " active" : ""}`;
    button.innerHTML = `
      <span class="character-name">${escapeHtml(character.label)}</span>
    `;
    button.addEventListener("click", () => setActiveCharacter(key));
    characterTabsEl.appendChild(button);
  });
}

function renderMessages() {
  const character = activeCharacterState();
  if (!character) {
    chatLogEl.innerHTML = "";
    return;
  }

  const messages = Array.isArray(character.messages) ? character.messages : [];
  if (!messages.length) {
    chatLogEl.innerHTML = `
      <div class="bubble assistant">
        ${escapeHtml(character.empty_message || "Start the conversation.")}
      </div>
    `;
    return;
  }

  chatLogEl.innerHTML = messages.map((message) => {
    const role = message.role === "user" ? "user" : "assistant";
    const metaBits = [];
    if (message.created_at) metaBits.push(formatTimestamp(message.created_at));
    if (role === "assistant" && Array.isArray(message.retrieved_context) && message.retrieved_context.length) {
      const preview = message.retrieved_context
        .slice(0, 2)
        .map((chunk) => `${chunk.persona || "archive"} @ ${Number(chunk.score || 0).toFixed(3)}`)
        .join(" | ");
      metaBits.push(preview);
    }
    return `
      <div class="bubble ${role}">
        ${escapeHtml(message.content || "")}
        ${metaBits.length ? `<div class="bubble-meta">${escapeHtml(metaBits.join(" | "))}</div>` : ""}
      </div>
    `;
  }).join("");

  chatLogEl.scrollTop = chatLogEl.scrollHeight;
}

function mediaCard(entry, type) {
  return mediaCardWithClass(entry, type, "");
}

function mediaCardWithClass(entry, type, extraClass) {
  const className = ["media-card", extraClass].filter(Boolean).join(" ");
  if (type === "image") {
    return `
      <figure class="${className}">
        <img src="${entry.url}" alt="${escapeHtml(entry.name)}" loading="lazy">
        <figcaption>${escapeHtml(entry.name)}</figcaption>
      </figure>
    `;
  }

  return `
    <figure class="${className}">
      <video controls preload="metadata">
        <source src="${entry.url}">
      </video>
      <figcaption>${escapeHtml(entry.name)}</figcaption>
    </figure>
  `;
}

function renderMedia() {
  const character = activeCharacterState();
  if (!character) return;

  const media = character.media || {};
  const images = Array.isArray(media.images) ? media.images : [];
  const videos = Array.isArray(media.videos) ? media.videos : [];

  imageMediaEl.classList.toggle("has-gallery", images.length > 1);
  videoMediaEl.classList.toggle("has-gallery", videos.length > 1);

  imageMediaEl.innerHTML = images.length
    ? [
        mediaCardWithClass(images[0], "image", "is-featured"),
        ...images.slice(1).map((entry) => mediaCard(entry, "image")),
      ].join("")
    : `<div class="media-empty">No images yet.</div>`;

  videoMediaEl.innerHTML = videos.length
    ? [
        mediaCardWithClass(videos[0], "video", "is-featured"),
        ...videos.slice(1).map((entry) => mediaCard(entry, "video")),
      ].join("")
    : `<div class="media-empty">No videos yet.</div>`;

}

function renderSidebar() {
  const character = activeCharacterState();
  if (!character) return;

  chatTitleEl.textContent = character.label;
  chatSubtitleEl.textContent = character.subtitle || "";

  const isSilicone = activeCharacterKey() === "silicone_subject";
  siliconeControlsEl.classList.toggle("hidden", !isSilicone);
  if (isSilicone) {
    shiftRangeEl.value = Number(character.shift || 0);
    shiftValueEl.textContent = describeShift(character.shift || 0);
    const mood = character.mood ? `Mood: ${character.mood}` : "Mood: none";
    const memories = Array.isArray(character.active_memories) ? character.active_memories.length : 0;
    siliconeMetaEl.textContent = `${mood} | Active memories: ${memories}`;
  }
}

function renderAll() {
  if (!appState) return;
  renderCharacterTabs();
  renderSidebar();
  renderMedia();
  renderMessages();
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || data.detail || "Request failed.");
  }
  return data;
}

async function loadState() {
  appState = await fetchJson("/api/chat/state");
  renderAll();
}

async function setActiveCharacter(key) {
  composerStatusEl.textContent = "Switching character...";
  appState = await fetchJson("/api/chat/state", {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ active_character: key }),
  });
  renderAll();
  composerStatusEl.textContent = "";
}

async function updateShift() {
  appState = await fetchJson("/api/chat/state", {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      active_character: "silicone_subject",
      shift: Number(shiftRangeEl.value),
    }),
  });
  renderAll();
}

async function randomizeMood() {
  composerStatusEl.textContent = "Randomizing mood...";
  const payload = await fetchJson("/api/chat/randomize-mood", { method: "POST" });
  appState = payload.state;
  renderAll();
  composerStatusEl.textContent = "Mood updated.";
}

async function addMemory() {
  composerStatusEl.textContent = "Adding memory...";
  const payload = await fetchJson("/api/chat/add-memory", { method: "POST" });
  appState = payload.state;
  renderAll();
  composerStatusEl.textContent = `Memory added: ${payload.memory}`;
}

async function resetCurrentTranscript() {
  composerStatusEl.textContent = "Resetting local transcript...";
  appState = await fetchJson("/api/chat/state/reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ character_key: activeCharacterKey() }),
  });
  renderAll();
  composerStatusEl.textContent = "Local transcript reset.";
}

function appendPendingBubble(text) {
  const bubble = document.createElement("div");
  bubble.className = "bubble assistant pending";
  bubble.textContent = text;
  bubble.dataset.pending = "true";
  chatLogEl.appendChild(bubble);
  chatLogEl.scrollTop = chatLogEl.scrollHeight;
}

function clearPendingBubble() {
  const pending = chatLogEl.querySelector("[data-pending='true']");
  if (pending) pending.remove();
}

async function sendMessage(event) {
  event.preventDefault();
  const message = messageInputEl.value.trim();
  if (!message) return;

  composerStatusEl.textContent = "Sending...";
  const character = activeCharacterState();
  if (!character) return;
  const previousState = JSON.parse(JSON.stringify(appState));

  const optimisticMessages = [...(character.messages || []), {
    role: "user",
    content: message,
    created_at: new Date().toISOString(),
  }];
  appState.characters[activeCharacterKey()].messages = optimisticMessages;
  renderMessages();
  appendPendingBubble("Thinking...");
  messageInputEl.value = "";

  try {
    const payload = await fetchJson("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        character_key: activeCharacterKey(),
        message,
      }),
    });
    clearPendingBubble();
    appState = payload.state;
    renderAll();
    composerStatusEl.textContent = "Reply received.";
  } catch (error) {
    clearPendingBubble();
    appState = previousState;
    composerStatusEl.textContent = error.message;
    renderAll();
  }
}

shiftRangeEl.addEventListener("input", () => {
  shiftValueEl.textContent = describeShift(shiftRangeEl.value);
});
shiftRangeEl.addEventListener("change", updateShift);
moodBtn.addEventListener("click", randomizeMood);
memoryBtn.addEventListener("click", addMemory);
resetBtn.addEventListener("click", resetCurrentTranscript);
composerEl.addEventListener("submit", sendMessage);

loadState().catch((error) => {
  composerStatusEl.textContent = error.message;
});
