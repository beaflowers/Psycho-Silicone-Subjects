const bootstrapData = JSON.parse(document.getElementById("bootstrap-data").textContent);

const chatLog = document.getElementById("chat-log");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const sendButton = document.getElementById("send-button");
const moodButton = document.getElementById("mood-button");
const memoryButton = document.getElementById("memory-button");
const newSessionButton = document.getElementById("new-session-button");
const shiftRange = document.getElementById("shift-range");
const shiftOutput = document.getElementById("shift-output");
const moodReadout = document.getElementById("mood-readout");
const memoryReadout = document.getElementById("memory-readout");
const revealItems = document.querySelectorAll(".reveal-on-scroll");

let state = null;

function setupScrollReveal() {
  if (!("IntersectionObserver" in window)) {
    for (const item of revealItems) {
      item.classList.add("is-visible");
    }
    return;
  }

  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      }
    },
    {
      threshold: 0.2,
      rootMargin: "0px 0px -8% 0px",
    },
  );

  for (const item of revealItems) {
    if (!item.classList.contains("is-visible")) {
      observer.observe(item);
    }
  }
}

function setBusy(isBusy) {
  sendButton.disabled = isBusy;
  moodButton.disabled = isBusy;
  memoryButton.disabled = isBusy;
  newSessionButton.disabled = isBusy;
}

function appendMessage(role, content) {
  const article = document.createElement("article");
  article.className = `chat-message ${role}`;

  const badge = document.createElement("p");
  badge.className = "message-role";
  badge.textContent = role === "assistant" ? "Subject" : "You";

  const body = document.createElement("p");
  body.className = "message-body";
  body.textContent = content;

  article.append(badge, body);
  chatLog.appendChild(article);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function renderState(currentState) {
  state = currentState;
  shiftRange.value = Number(state.shift || 0).toFixed(2);
  shiftOutput.value = Number(state.shift || 0).toFixed(2);
  moodReadout.textContent = state.mood || "none";
  memoryReadout.textContent = `${(state.active_memories || []).length} active`;
  chatLog.innerHTML = "";

  for (const message of state.messages || []) {
    appendMessage(message.role, message.content);
  }
}

async function loadState() {
  const response = await fetch("/api/chat/state");
  if (!response.ok) {
    throw new Error("Unable to load chat state.");
  }
  const currentState = await response.json();
  renderState(currentState);
}

async function patchState(payload) {
  const response = await fetch("/api/chat/state", {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error("Unable to update session state.");
  }
  const updated = await response.json();
  renderState(updated);
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = chatInput.value.trim();
  if (!message || !state) {
    return;
  }

  appendMessage("user", message);
  chatInput.value = "";
  setBusy(true);

  try {
    const response = await fetch("/api/chat/message", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.error || "The archive could not answer.");
    }

    renderState(payload.state);
  } catch (error) {
    appendMessage("assistant", error.message);
  } finally {
    setBusy(false);
  }
});

moodButton.addEventListener("click", async () => {
  if (!state) {
    return;
  }
  setBusy(true);
  try {
    const response = await fetch("/api/chat/mood", {
      method: "POST",
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Unable to randomize mood.");
    }
    renderState(payload.state);
  } catch (error) {
    appendMessage("assistant", error.message);
  } finally {
    setBusy(false);
  }
});

memoryButton.addEventListener("click", async () => {
  if (!state) {
    return;
  }
  setBusy(true);
  try {
    const response = await fetch("/api/chat/memory", {
      method: "POST",
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Unable to surface a memory.");
    }
    renderState(payload.state);
    appendMessage("assistant", `Memory surfaced: ${payload.memory}`);
  } catch (error) {
    appendMessage("assistant", error.message);
  } finally {
    setBusy(false);
  }
});

newSessionButton.addEventListener("click", async () => {
  setBusy(true);
  try {
    const response = await fetch("/api/chat/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title: "Subject interview log",
        shift: Number(shiftRange.value),
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Unable to reset the chat state.");
    }
    renderState(payload);
    appendMessage("assistant", "The saved subject file has been reset.");
  } catch (error) {
    appendMessage("assistant", error.message);
  } finally {
    setBusy(false);
  }
});

shiftRange.addEventListener("input", () => {
  shiftOutput.value = Number(shiftRange.value).toFixed(2);
});

shiftRange.addEventListener("change", async () => {
  if (!state) {
    return;
  }
  try {
    await patchState({ shift: Number(shiftRange.value) });
  } catch (error) {
    appendMessage("assistant", error.message);
  }
});

window.addEventListener("load", async () => {
  setupScrollReveal();
  shiftOutput.value = Number(shiftRange.value).toFixed(2);
  try {
    await loadState();
    if (bootstrapData.ragError) {
      appendMessage("assistant", `RAG startup warning: ${bootstrapData.ragError}`);
    } else if (!(state.messages || []).length) {
      appendMessage("assistant", "Archive ready. Begin the interview when you are.");
    }
  } catch (error) {
    appendMessage("assistant", error.message);
  }
});
