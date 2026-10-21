(() => {
  const el = id => document.getElementById(id);
  const storageKey = 'dgrag-ask-poe-v1';
  let turns = [], sending = false;
  try {
    const saved = JSON.parse(localStorage.getItem(storageKey) || '[]');
    if (Array.isArray(saved)) turns = saved.filter(t => typeof t.question === 'string' && typeof t.answer === 'string').slice(-20);
  } catch (_) { /* Chat works when browser storage is unavailable. */ }

  function selectTab(name, focus = false) {
    for (const tab of ['library', 'poe']) {
      const selected = tab === name;
      el(`${tab}-tab`).setAttribute('aria-selected', String(selected));
      el(`${tab}-tab`).tabIndex = selected ? 0 : -1;
      el(`${tab}-panel`).hidden = !selected;
    }
    location.hash = name === 'poe' ? 'ask-poe' : 'library';
    if (focus) el(`${name}-tab`).focus();
  }
  for (const name of ['library', 'poe']) {
    el(`${name}-tab`).onclick = () => selectTab(name);
    el(`${name}-tab`).onkeydown = event => {
      if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
        event.preventDefault();
        selectTab(event.key === 'Home' ? 'library' : event.key === 'End' ? 'poe' : name === 'poe' ? 'library' : 'poe', true);
      }
    };
  }
  window.addEventListener('hashchange', () => selectTab(location.hash === '#ask-poe' ? 'poe' : 'library'));
  selectTab(location.hash === '#ask-poe' ? 'poe' : 'library');

  function addMessage(role, text, matches = []) {
    const message = document.createElement('article');
    message.className = `poe-message ${role}`;
    const label = document.createElement('span');
    label.className = 'speaker';
    label.textContent = role === 'user' ? 'You' : 'Edgar Allan Poe · literary simulation';
    const body = document.createElement('div');
    body.className = 'message-text';
    body.textContent = text;
    message.append(label, body);
    if (Array.isArray(matches) && matches.length) {
      const details = document.createElement('details');
      const summary = document.createElement('summary');
      summary.textContent = `Read the supporting passages (${matches.length})`;
      details.append(summary);
      matches.forEach((match, i) => {
        const metadata = match.metadata || {};
        const source = document.createElement('div');
        source.className = 'source';
        const link = document.createElement('a');
        const allowed = ['https://www.gutenberg.org/cache/epub/2147/pg2147.txt', 'https://www.gutenberg.org/cache/epub/2148/pg2148.txt'];
        if (allowed.includes(metadata.url)) { link.href = metadata.url; link.target = '_blank'; link.rel = 'noreferrer'; }
        link.textContent = `[${i + 1}] ${metadata.name || 'Source passage'}`;
        const passage = document.createElement('pre');
        passage.textContent = metadata.text || '';
        source.append(link, passage);
        details.append(source);
      });
      message.append(details);
    }
    el('poe-chat').append(message);
    el('poe-chat').scrollTop = el('poe-chat').scrollHeight;
    return message;
  }
  function render() {
    el('poe-chat').replaceChildren();
    if (!turns.length) addMessage('assistant', 'Come, let us turn the page together. Which character troubles you, which image lingers, or which feeling would you have us examine?');
    for (const turn of turns) {
      addMessage('user', turn.question);
      addMessage('assistant', turn.answer, turn.matches);
    }
  }
  function save() { try { localStorage.setItem(storageKey, JSON.stringify(turns.slice(-20))); } catch (_) {} }
  function setSending(value) {
    sending = value;
    el('poe-send').disabled = el('poe-new').disabled = value;
    el('poe-pending').hidden = !value;
    el('poe-chat').setAttribute('aria-busy', String(value));
    for (const button of document.querySelectorAll('[data-question]')) button.disabled = value;
  }
  el('poe-new').onclick = () => { turns = []; save(); render(); el('poe-error').textContent = ''; el('poe-question').value = ''; el('poe-question').focus(); };
  for (const button of document.querySelectorAll('[data-question]')) button.onclick = () => {
    el('poe-question').value = button.dataset.question;
    el('poe-question').focus();
  };
  el('poe-form').onsubmit = async event => {
    event.preventDefault();
    const question = el('poe-question').value.trim();
    if (sending || !question) return;
    el('poe-error').textContent = '';
    const history = turns.slice(-6).flatMap(turn => [{role:'user', content:turn.question.slice(0,6000)}, {role:'assistant', content:turn.answer.slice(0,6000)}]);
    const pendingMessage = addMessage('user', question);
    el('poe-question').value = '';
    setSending(true);
    try {
      const response = await fetch('/api/poe', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({question, history})});
      const data = await response.json();
      if (!response.ok) throw Error(data.error || 'Poe could not respond. Please try again.');
      turns.push({question, answer:data.answer, matches:data.matches});
      turns = turns.slice(-20);
      addMessage('assistant', data.answer, data.matches);
      save();
    } catch (error) {
      pendingMessage.remove();
      if (!el('poe-question').value) el('poe-question').value = question;
      el('poe-error').textContent = error.message;
    } finally { setSending(false); el('poe-question').focus(); }
  };
  window.addEventListener('dgrag:state', event => {
    const state = event.detail;
    el('poe-readiness').textContent = !state.active_namespace
      ? 'Load and index the Poe volumes in Library & pipeline to begin a conversation.'
      : !state.generation_ready
        ? 'The library is indexed. Download the local answer model using python app.py --download-generator.'
        : 'The library is ready. Ask about a character, a feeling, a theme, or an image.';
  });
  render();
})();
