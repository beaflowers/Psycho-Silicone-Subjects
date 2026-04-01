const state = {
  sessionId: null,
  experimentStarted: false,
  experimentDone: false,
};

const el = {
  setupForm: document.querySelector('#setup-form'),
  startSessionBtn: document.querySelector('#start-session-btn'),
  sessionMeta: document.querySelector('#session-meta'),
  status: document.querySelector('#status'),

  personaAdminPath: document.querySelector('#persona-admin-path'),
  memoriesAdminPath: document.querySelector('#memories-admin-path'),
  chatSessionsAdminPath: document.querySelector('#chat-sessions-admin-path'),
  shockSessionsAdminPath: document.querySelector('#shock-sessions-admin-path'),
  personaReceiverPath: document.querySelector('#persona-receiver-path'),
  memoriesReceiverPath: document.querySelector('#memories-receiver-path'),
  chatSessionsReceiverPath: document.querySelector('#chat-sessions-receiver-path'),
  shockSessionsReceiverPath: document.querySelector('#shock-sessions-receiver-path'),
  chatModel: document.querySelector('#chat-model'),
  embedModel: document.querySelector('#embed-model'),
  topK: document.querySelector('#top-k'),

  downloadJsonl: document.querySelector('#download-jsonl'),
  downloadCsv: document.querySelector('#download-csv'),

  chatTarget: document.querySelector('#chat-target'),
  chatForm: document.querySelector('#chat-form'),
  chatInput: document.querySelector('#chat-input'),
  chatSendBtn: document.querySelector('#chat-send-btn'),
  finishChatBtn: document.querySelector('#finish-chat-btn'),
  chatLog: document.querySelector('#chat-log'),
  chatReasoning: document.querySelector('#chat-reasoning'),

  expStartForm: document.querySelector('#experiment-start-form'),
  expStart: document.querySelector('#exp-start'),
  expEnd: document.querySelector('#exp-end'),
  expStep: document.querySelector('#exp-step'),
  operatorNote: document.querySelector('#operator-note'),
  startExpBtn: document.querySelector('#start-exp-btn'),
  nextStepBtn: document.querySelector('#next-step-btn'),
  finishExpBtn: document.querySelector('#finish-exp-btn'),
  receiverCard: document.querySelector('#receiver-card'),
  adminCard: document.querySelector('#admin-card'),
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

function resetExperimentUi() {
  state.experimentStarted = false;
  state.experimentDone = false;
  el.nextStepBtn.disabled = true;
  el.finishExpBtn.disabled = true;
  el.experimentLog.innerHTML = '';
  el.receiverCard.className = 'reasoning empty';
  el.receiverCard.textContent = 'No receiver reaction yet.';
  el.adminCard.className = 'reasoning empty';
  el.adminCard.textContent = 'No admin decision yet.';
}

function enableSessionUi() {
  el.chatSendBtn.disabled = false;
  el.finishChatBtn.disabled = true;
  el.startExpBtn.disabled = false;
  el.downloadJsonl.disabled = false;
  el.downloadCsv.disabled = false;
}

function renderSessionMeta(out) {
  el.sessionMeta.innerHTML = [
    `<div><strong>Session:</strong> ${escapeHtml(out.session_id || '')}</div>`,
    `<div><strong>Admin Subject:</strong> ${escapeHtml(out.admin_persona_name || '')}</div>`,
    `<div><strong>Receiver Subject:</strong> ${escapeHtml(out.receiver_persona_name || '')}</div>`,
    `<div><strong>Trace:</strong> ${escapeHtml(out.trace_path || '')}</div>`,
  ].join('');
}

function appendChatTurn(subject, personaName, userMessage, responseText) {
  const item = document.createElement('article');
  item.className = 'log-item';
  item.innerHTML = [
    `<h3>${escapeHtml(subject === 'admin' ? 'Admin Subject' : 'Receiver Subject')}</h3>`,
    `<p><strong>You:</strong> ${escapeHtml(userMessage)}</p>`,
    `<p><strong>${escapeHtml(personaName || 'Persona')}:</strong> ${escapeHtml(responseText || '')}</p>`,
  ].join('');
  el.chatLog.prepend(item);
}

function renderChatReasoning(out) {
  el.chatReasoning.classList.remove('empty');
  el.chatReasoning.innerHTML = [
    `<p><strong>Subject:</strong> ${escapeHtml(out.subject || '')}</p>`,
    `<p><strong>Reasoning:</strong> ${escapeHtml(out.reasoning_background || '(none)')}</p>`,
    `<p><strong>Memories Used:</strong> ${escapeHtml((out.memories_used || []).join(', ') || '(none)')}</p>`,
    '<div><strong>Retrieved Memories</strong></div>',
    memoriesToHtml(out.retrieved_memories || []),
  ].join('');
}

function renderReceiverCard(step) {
  const receiver = step && step.receiver ? step.receiver : null;
  if (!receiver) {
    el.receiverCard.className = 'reasoning empty';
    el.receiverCard.textContent = 'No receiver reaction yet.';
    return;
  }
  el.receiverCard.className = 'reasoning';
  el.receiverCard.innerHTML = [
    `<p><strong>Responded:</strong> ${escapeHtml(String(Boolean(receiver.responded)))}</p>`,
    `<p><strong>Message:</strong> ${escapeHtml(receiver.message || '(silent)')}</p>`,
    `<p><strong>Reasoning:</strong> ${escapeHtml(receiver.reasoning_background || '(none)')}</p>`,
    `<p><strong>Distress:</strong> ${escapeHtml(Number(receiver.distress_level || 0).toFixed(2))}</p>`,
    `<p><strong>Memories Used:</strong> ${escapeHtml((receiver.memories_used || []).join(', ') || '(none)')}</p>`,
    '<div><strong>Retrieved Memories</strong></div>',
    memoriesToHtml(receiver.retrieved_memories || []),
  ].join('');
}

function renderAdminCard(step) {
  const admin = step && step.admin ? step.admin : null;
  if (!admin) {
    el.adminCard.className = 'reasoning empty';
    el.adminCard.textContent = 'No admin decision yet.';
    return;
  }
  el.adminCard.className = 'reasoning';
  el.adminCard.innerHTML = [
    `<p><strong>Action:</strong> ${escapeHtml(admin.action || '')}</p>`,
    `<p><strong>Confidence:</strong> ${escapeHtml(Number(admin.confidence || 0).toFixed(2))}</p>`,
    `<p><strong>Reasoning:</strong> ${escapeHtml(admin.reasoning_background || '(none)')}</p>`,
    `<p><strong>Memories Used:</strong> ${escapeHtml((admin.memories_used || []).join(', ') || '(none)')}</p>`,
    '<div><strong>Retrieved Memories</strong></div>',
    memoriesToHtml(admin.retrieved_memories || []),
  ].join('');
}

function appendExperimentLog(step) {
  const item = document.createElement('article');
  item.className = 'log-item';
  item.innerHTML = [
    `<h3>Turn ${escapeHtml(step.turn_index)} | ${escapeHtml(step.voltage)}V</h3>`,
    `<p><strong>Authority:</strong> ${escapeHtml(step.authority_command || '')}</p>`,
    `<p><strong>Learner cue baseline:</strong> ${escapeHtml(step.learner_cue || '')}</p>`,
    `<p><strong>Receiver responded:</strong> ${escapeHtml(String(Boolean(step.receiver && step.receiver.responded)))}</p>`,
    `<p><strong>Admin action:</strong> ${escapeHtml(step.admin ? step.admin.action : '')} (${escapeHtml(Number(step.admin ? step.admin.confidence : 0).toFixed(2))})</p>`,
  ].join('');
  el.experimentLog.prepend(item);
}

function downloadTrace(format) {
  if (!state.sessionId) {
    setStatus('Start a session first.', true);
    return;
  }
  const url = `/api/v2/export?session_id=${encodeURIComponent(state.sessionId)}&format=${encodeURIComponent(format)}`;
  window.open(url, '_blank', 'noopener,noreferrer');
}

el.downloadJsonl.addEventListener('click', () => downloadTrace('jsonl'));
el.downloadCsv.addEventListener('click', () => downloadTrace('csv'));

el.setupForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  try {
    el.startSessionBtn.disabled = true;
    setStatus('Starting dual session...');
    const out = await postJson('/api/v2/start_session', {
      persona_admin_path: el.personaAdminPath.value.trim(),
      memories_admin_path: el.memoriesAdminPath.value.trim(),
      chat_sessions_admin_path: el.chatSessionsAdminPath.value.trim(),
      shock_sessions_admin_path: el.shockSessionsAdminPath.value.trim(),
      persona_receiver_path: el.personaReceiverPath.value.trim(),
      memories_receiver_path: el.memoriesReceiverPath.value.trim(),
      chat_sessions_receiver_path: el.chatSessionsReceiverPath.value.trim(),
      shock_sessions_receiver_path: el.shockSessionsReceiverPath.value.trim(),
      chat_model: el.chatModel.value.trim(),
      embed_model: el.embedModel.value.trim(),
      top_k: Number(el.topK.value || 3),
    });
    state.sessionId = out.session_id;
    renderSessionMeta(out);
    el.chatLog.innerHTML = '';
    el.chatReasoning.className = 'reasoning empty';
    el.chatReasoning.textContent = 'No chat turns yet.';
    enableSessionUi();
    resetExperimentUi();
    setStatus('Dual session started. You can chat with either subject or run the experiment.');
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
  const subject = el.chatTarget.value;
  try {
    el.chatSendBtn.disabled = true;
    setStatus(`Sending chat to ${subject} subject...`);
    const out = await postJson('/api/v2/chat_turn', {
      session_id: state.sessionId,
      subject,
      message,
    });
    appendChatTurn(subject, out.persona_name, message, out.response_text);
    renderChatReasoning(out);
    el.finishChatBtn.disabled = false;
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

el.finishChatBtn.addEventListener('click', async () => {
  if (!state.sessionId) {
    setStatus('Start a session first.', true);
    return;
  }
  const subject = el.chatTarget.value;
  let success = false;
  try {
    el.finishChatBtn.disabled = true;
    setStatus(`Saving ${subject} chat summary...`);
    const out = await postJson('/api/v2/chat_finish', {
      session_id: state.sessionId,
      subject,
    });
    setStatus(
      `${subject} chat summary saved (${out.summary ? out.summary.id : 'no-id'}).`
    );
    success = true;
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    if (state.sessionId) {
      el.finishChatBtn.disabled = success;
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
    setStatus('Starting dual-subject experiment...');
    const out = await postJson('/api/v2/experiment_start', {
      session_id: state.sessionId,
      start_voltage: Number(el.expStart.value || 15),
      end_voltage: Number(el.expEnd.value || 450),
      step: Number(el.expStep.value || 15),
    });
    state.experimentStarted = true;
    state.experimentDone = false;
    el.nextStepBtn.disabled = false;
    el.finishExpBtn.disabled = true;
    setStatus(`Experiment started. Next command: ${out.next_authority_command}`);
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
    setStatus('Running next dual-subject step...');
    const step = await postJson('/api/v2/experiment_step', {
      session_id: state.sessionId,
      operator_note: el.operatorNote.value.trim(),
    });
    renderReceiverCard(step);
    renderAdminCard(step);
    appendExperimentLog(step);
    el.finishExpBtn.disabled = false;

    if (step.done) {
      state.experimentDone = true;
      setStatus(`Experiment ended at ${step.voltage}V. Admin action: ${step.admin.action}.`);
      return;
    }
    setStatus(`Step ${step.turn_index} done. Next level: ${step.next_voltage}V.`);
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    if (state.sessionId && state.experimentStarted && !state.experimentDone) {
      el.nextStepBtn.disabled = false;
    }
  }
});

el.finishExpBtn.addEventListener('click', async () => {
  if (!state.sessionId) {
    setStatus('Start a session first.', true);
    return;
  }
  let success = false;
  try {
    el.finishExpBtn.disabled = true;
    setStatus('Saving shock summaries for both subjects...');
    const out = await postJson('/api/v2/experiment_finish', {
      session_id: state.sessionId,
    });
    const adminId = out.admin_summary ? out.admin_summary.id : 'none';
    const receiverId = out.receiver_summary ? out.receiver_summary.id : 'none';
    setStatus(`Shock summaries saved. admin=${adminId} receiver=${receiverId}`);
    success = true;
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    if (state.sessionId) {
      el.finishExpBtn.disabled = success;
    }
  }
});
