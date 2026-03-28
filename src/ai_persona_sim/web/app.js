const state = {
  sessionId: null,
  activeTab: 'chat',
  experimentStarted: false,
  experimentDone: false,
};

const el = {
  setupForm: document.querySelector('#setup-form'),
  startSessionBtn: document.querySelector('#start-session-btn'),
  sessionMeta: document.querySelector('#session-meta'),
  status: document.querySelector('#status'),
  downloadJsonl: document.querySelector('#download-jsonl'),
  downloadCsv: document.querySelector('#download-csv'),
  tabChat: document.querySelector('#tab-chat'),
  tabExperiment: document.querySelector('#tab-experiment'),
  modeChat: document.querySelector('#mode-chat'),
  modeExperiment: document.querySelector('#mode-experiment'),
  personaPath: document.querySelector('#persona-path'),
  memoriesPath: document.querySelector('#memories-path'),
  chatModel: document.querySelector('#chat-model'),
  embedModel: document.querySelector('#embed-model'),
  topK: document.querySelector('#top-k'),
  chatForm: document.querySelector('#chat-form'),
  chatInput: document.querySelector('#chat-input'),
  chatSendBtn: document.querySelector('#chat-send-btn'),
  chatLog: document.querySelector('#chat-log'),
  chatReasoning: document.querySelector('#chat-reasoning'),
  expStartForm: document.querySelector('#experiment-start-form'),
  expStart: document.querySelector('#exp-start'),
  expEnd: document.querySelector('#exp-end'),
  expStep: document.querySelector('#exp-step'),
  expFeeling: document.querySelector('#exp-feeling'),
  startExpBtn: document.querySelector('#start-exp-btn'),
  nextStepBtn: document.querySelector('#next-step-btn'),
  authorityCard: document.querySelector('#authority-card'),
  experimentDecision: document.querySelector('#experiment-decision'),
  experimentLog: document.querySelector('#experiment-log'),
};

function setStatus(text, isError = false) {
  el.status.textContent = text;
  el.status.classList.toggle('error', isError);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

async function postJson(url, payload) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.error || `Request failed with status ${res.status}`);
  }
  return data;
}

function openTab(tabName) {
  state.activeTab = tabName;
  const chatActive = tabName === 'chat';
  el.tabChat.classList.toggle('active', chatActive);
  el.tabExperiment.classList.toggle('active', !chatActive);
  el.modeChat.classList.toggle('active', chatActive);
  el.modeExperiment.classList.toggle('active', !chatActive);
}

function renderSessionMeta(data) {
  el.sessionMeta.innerHTML = [
    `<div><strong>Session:</strong> ${escapeHtml(data.session_id)}</div>`,
    `<div><strong>Persona:</strong> ${escapeHtml(data.persona_name)}</div>`,
    `<div><strong>Trace:</strong> ${escapeHtml(data.trace_path)}</div>`,
  ].join('');
}

function memoriesToHtml(memories) {
  if (!Array.isArray(memories) || memories.length === 0) {
    return '<p>(none)</p>';
  }
  return [
    '<ul>',
    ...memories.map(
      (m) =>
        `<li><strong>${escapeHtml(m.id)}</strong> score=${escapeHtml(Number(m.score).toFixed(3))} tags=${escapeHtml((m.tags || []).join(', '))}</li>`
    ),
    '</ul>',
  ].join('');
}

function appendChatTurn(userMessage, response) {
  const item = document.createElement('article');
  item.className = 'log-item';
  item.innerHTML = [
    `<p><strong>You:</strong> ${escapeHtml(userMessage)}</p>`,
    `<p><strong>Persona:</strong> ${escapeHtml(response.response_text || '')}</p>`,
  ].join('');
  el.chatLog.prepend(item);
}

function renderChatReasoning(response) {
  el.chatReasoning.classList.remove('empty');
  el.chatReasoning.innerHTML = [
    `<p><strong>Reasoning:</strong> ${escapeHtml(response.reasoning_background || '(none)')}</p>`,
    `<p><strong>Memories Used:</strong> ${escapeHtml((response.memories_used || []).join(', ') || '(none)')}</p>`,
    '<div><strong>Retrieved Memories</strong></div>',
    memoriesToHtml(response.retrieved_memories || []),
  ].join('');
}

function renderAuthorityCard(payload) {
  if (!payload || !payload.next_voltage) {
    el.authorityCard.className = 'authority-card empty';
    el.authorityCard.textContent = 'No upcoming authority step.';
    return;
  }
  el.authorityCard.className = 'authority-card';
  el.authorityCard.innerHTML = [
    `<div><strong>Next Level:</strong> ${escapeHtml(payload.next_voltage)}V</div>`,
    `<div><strong>Authority Command:</strong></div>`,
    `<p>${escapeHtml(payload.next_authority_command || '')}</p>`,
  ].join('');
}

function renderExperimentDecision(step) {
  el.experimentDecision.classList.remove('empty');
  el.experimentDecision.innerHTML = [
    `<p><strong>Voltage:</strong> ${escapeHtml(step.voltage)}V</p>`,
    `<p><strong>Authority:</strong> ${escapeHtml(step.authority_command || '')}</p>`,
    `<p><strong>Learner cue:</strong> ${escapeHtml(step.learner_cue || '')}</p>`,
    `<p><strong>Action:</strong> ${escapeHtml(step.action)} | <strong>Confidence:</strong> ${escapeHtml(Number(step.confidence || 0).toFixed(2))}</p>`,
    `<p><strong>Reasoning:</strong> ${escapeHtml(step.reasoning_background || '(none)')}</p>`,
    `<p><strong>Memories Used:</strong> ${escapeHtml((step.memories_used || []).join(', ') || '(none)')}</p>`,
    '<div><strong>Retrieved Memories</strong></div>',
    memoriesToHtml(step.retrieved_memories || []),
  ].join('');
}

function appendExperimentLog(step) {
  const item = document.createElement('article');
  item.className = 'log-item';
  item.innerHTML = [
    `<h3>Turn ${escapeHtml(step.turn_index)} - ${escapeHtml(step.voltage)}V</h3>`,
    `<p><strong>Action:</strong> ${escapeHtml(step.action)} (${escapeHtml(Number(step.confidence || 0).toFixed(2))})</p>`,
    `<p><strong>Reasoning:</strong> ${escapeHtml(step.reasoning_background || '(none)')}</p>`,
    `<p><strong>Memories:</strong> ${escapeHtml((step.memories_used || []).join(', ') || '(none)')}</p>`,
  ].join('');
  el.experimentLog.prepend(item);
}

function enableSessionUi() {
  el.chatSendBtn.disabled = false;
  el.startExpBtn.disabled = false;
  el.downloadJsonl.disabled = false;
  el.downloadCsv.disabled = false;
}

function resetExperimentUi() {
  state.experimentStarted = false;
  state.experimentDone = false;
  el.nextStepBtn.disabled = true;
  el.experimentDecision.classList.add('empty');
  el.experimentDecision.textContent = 'No experiment turns yet.';
  el.experimentLog.innerHTML = '';
  renderAuthorityCard(null);
}

function downloadTrace(format) {
  if (!state.sessionId) {
    setStatus('Start a session first.', true);
    return;
  }
  const url = `/api/export?session_id=${encodeURIComponent(state.sessionId)}&format=${encodeURIComponent(format)}`;
  window.open(url, '_blank', 'noopener,noreferrer');
}

el.tabChat.addEventListener('click', () => openTab('chat'));
el.tabExperiment.addEventListener('click', () => openTab('experiment'));

el.downloadJsonl.addEventListener('click', () => downloadTrace('jsonl'));
el.downloadCsv.addEventListener('click', () => downloadTrace('csv'));

el.setupForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  try {
    el.startSessionBtn.disabled = true;
    setStatus('Starting session...');
    const payload = {
      persona_path: el.personaPath.value.trim(),
      memories_path: el.memoriesPath.value.trim(),
      chat_model: el.chatModel.value.trim(),
      embed_model: el.embedModel.value.trim(),
      top_k: Number(el.topK.value || 3),
    };
    const data = await postJson('/api/start_session', payload);
    state.sessionId = data.session_id;
    renderSessionMeta(data);
    enableSessionUi();
    resetExperimentUi();
    el.chatLog.innerHTML = '';
    el.chatReasoning.className = 'reasoning empty';
    el.chatReasoning.textContent = 'No chat turns yet.';
    setStatus('Session started. You can now chat or run the experiment.');
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    el.startSessionBtn.disabled = false;
  }
});

el.chatForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!state.sessionId) {
    setStatus('Start a session first.', true);
    return;
  }
  const message = el.chatInput.value.trim();
  if (!message) {
    return;
  }
  try {
    el.chatSendBtn.disabled = true;
    setStatus('Generating chat response...');
    const response = await postJson('/api/chat_turn', {
      session_id: state.sessionId,
      message,
    });
    appendChatTurn(message, response);
    renderChatReasoning(response);
    el.chatInput.value = '';
    setStatus('Chat turn complete.');
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    if (state.sessionId) {
      el.chatSendBtn.disabled = false;
    }
  }
});

el.expStartForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!state.sessionId) {
    setStatus('Start a session first.', true);
    return;
  }
  try {
    el.startExpBtn.disabled = true;
    setStatus('Starting experiment...');
    const payload = {
      session_id: state.sessionId,
      start_voltage: Number(el.expStart.value || 15),
      end_voltage: Number(el.expEnd.value || 450),
      step: Number(el.expStep.value || 15),
    };
    const out = await postJson('/api/experiment_start', payload);
    state.experimentStarted = true;
    state.experimentDone = false;
    el.nextStepBtn.disabled = false;
    renderAuthorityCard(out);
    setStatus('Experiment started. Click "Run Next Step".');
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    if (state.sessionId) {
      el.startExpBtn.disabled = false;
    }
  }
});

el.nextStepBtn.addEventListener('click', async () => {
  if (!state.sessionId || !state.experimentStarted) {
    setStatus('Start experiment first.', true);
    return;
  }
  try {
    el.nextStepBtn.disabled = true;
    setStatus('Running experiment step...');
    const step = await postJson('/api/experiment_step', {
      session_id: state.sessionId,
      user_feeling: el.expFeeling.value.trim(),
    });
    renderExperimentDecision(step);
    appendExperimentLog(step);
    renderAuthorityCard({
      next_voltage: step.next_voltage,
      next_authority_command: step.next_authority_command,
    });
    if (step.done) {
      state.experimentDone = true;
      el.nextStepBtn.disabled = true;
      setStatus('Experiment ended.');
      return;
    }
    setStatus(`Step ${step.turn_index} complete at ${step.voltage}V.`);
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    if (state.sessionId && state.experimentStarted && !state.experimentDone) {
      el.nextStepBtn.disabled = false;
    }
  }
});
