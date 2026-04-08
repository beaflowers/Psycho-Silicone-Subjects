const boot = window.APP_BOOT || { model: "", characters: {} };

const characterTabsEl = document.getElementById("characterTabs");
const siliconeControlsEl = document.getElementById("siliconeControls");
const shiftRangeEl = document.getElementById("shiftRange");
const shiftValueEl = document.getElementById("shiftValue");
const chatTitleEl = document.getElementById("chatTitle");
const chatSubtitleEl = document.getElementById("chatSubtitle");
const chatLogEl = document.getElementById("chatLog");
const memoryDebugLogEl = document.getElementById("memoryDebugLog");
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

function formatMemoryEntry(entry) {
  const timestamp = entry?.timestamp ? `[${entry.timestamp}] ` : "";
  const kind = entry?.kind ? `${entry.kind}: ` : "";
  const excerpt = entry?.excerpt || "(empty excerpt)";
  return `${timestamp}${kind}${excerpt}`;
}

function excerptText(value, maxLength = 220) {
  const text = String(value || "").trim().replaceAll(/\s+/g, " ");
  if (text.length <= maxLength) return text;
  return `${text.slice(0, Math.max(0, maxLength - 3)).trimEnd()}...`;
}

function renderMemoryDebug() {
  const character = activeCharacterState();
  if (!character || !memoryDebugLogEl) {
    return;
  }

  const messages = Array.isArray(character.messages) ? character.messages : [];
  const assistantMessages = messages
    .map((message, index) => ({ message, index }))
    .filter(({ message }) => message.role === "assistant")
    .map(({ message, index }, turnIndex) => ({
      message,
      index,
      turn: turnIndex + 1,
    }))
    .reverse();

  if (!assistantMessages.length) {
    memoryDebugLogEl.innerHTML = `<p class="debug-empty">No interactions yet.</p>`;
    return;
  }

  memoryDebugLogEl.innerHTML = assistantMessages.map(({ message, turn, index }) => {
    const shock = message.shock_context;
    const stamp = formatTimestamp(message.created_at) || "Unknown time";
    const retrieved = Array.isArray(message.retrieved_context) ? message.retrieved_context : [];
    const userPrompt = [...messages.slice(0, index)].reverse().find((item) => item.role === "user");

    const ragHtml = retrieved.length
      ? `<ul class="debug-list">${retrieved
          .map((chunk) => {
            const persona = chunk?.persona || "archive";
            const score = Number(chunk?.score || 0).toFixed(3);
            const source = chunk?.source_path || "unknown source";
            const snippet = excerptText(chunk?.text || "", 180);
            return `<li><strong>${escapeHtml(persona)}</strong> @ ${escapeHtml(score)}<br><span class="debug-path">${escapeHtml(source)}</span><br>${escapeHtml(snippet)}</li>`;
          })
          .join("")}</ul>`
      : `<p class="debug-line">No RAG chunks were attached to this reply.</p>`;

    if (!shock || typeof shock !== "object") {
      return `
        <article class="debug-turn">
          <p class="debug-turn-title">Turn ${turn} · ${escapeHtml(stamp)}</p>
          <p class="debug-subhead">User Prompt</p>
          <p class="debug-line">${escapeHtml(userPrompt?.content || "n/a")}</p>
          <p class="debug-subhead">RAG Retrieved Context</p>
          ${ragHtml}
          <p class="debug-subhead">Shock Session Context</p>
          <p class="debug-line">Session: n/a</p>
          <p class="debug-line">Persona/Role: n/a</p>
          <p class="debug-subhead">Primary Memory</p>
          <p class="debug-line">No primary memory entries selected.</p>
        </article>
      `;
    }

    const primaryEntries = Array.isArray(shock.primary_entries) ? shock.primary_entries : [];
    const contrastEntries = Array.isArray(shock.contrast_entries) ? shock.contrast_entries : [];

    const primaryHtml = primaryEntries.length
      ? `<ul class="debug-list">${primaryEntries
          .map((entry) => `<li>${escapeHtml(formatMemoryEntry(entry))}</li>`)
          .join("")}</ul>`
      : `<p class="debug-line">No primary memory entries selected.</p>`;

    const contrastSection = contrastEntries.length
      ? `
        <p class="debug-subhead">Other Participant Memory</p>
        <ul class="debug-list">${contrastEntries
          .map((entry) => `<li>${escapeHtml(formatMemoryEntry(entry))}</li>`)
          .join("")}</ul>
      `
      : "";

    return `
      <article class="debug-turn">
        <p class="debug-turn-title">Turn ${turn} · ${escapeHtml(stamp)}</p>
        <p class="debug-subhead">User Prompt</p>
        <p class="debug-line">${escapeHtml(userPrompt?.content || "n/a")}</p>
        <p class="debug-subhead">RAG Retrieved Context</p>
        ${ragHtml}
        <p class="debug-subhead">Shock Session Context</p>
        <p class="debug-line">Session: ${escapeHtml(shock.session_id || "n/a")}</p>
        <p class="debug-line">Persona/Role: ${escapeHtml(shock.character_persona || "n/a")} (${escapeHtml(shock.character_role || "participant")})</p>
        <p class="debug-subhead">Primary Memory</p>
        ${primaryHtml}
        ${contrastSection}
      </article>
    `;
  }).join("");
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
  }
}

function renderAll() {
  if (!appState) return;
  renderCharacterTabs();
  renderSidebar();
  renderMedia();
  renderMessages();
  renderMemoryDebug();
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
resetBtn.addEventListener("click", resetCurrentTranscript);
composerEl.addEventListener("submit", sendMessage);

loadState().catch((error) => {
  composerStatusEl.textContent = error.message;
});
