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
        const allowed = ['https://www.gutenberg.org/cache/epub/2147/pg2147.txt', 'https://www.gutenberg.org/cache/epub/2148/pg2148.txt', 'https://www.eapoe.org/works/essays/philcomp.htm'];
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
  const modelOrder = ['local', 'minilm', 'openai'];
  const modelLabels = {local: 'Current local model', minilm: 'all-MiniLM-L6-v2', openai: 'OpenAI'};
  let modelSettings = {}, openaiConnected = null;

  function addComparison(outputs = [], matches = []) {
    const row = document.createElement('article');
    row.className = 'poe-comparison';
    const grid = document.createElement('div');
    grid.className = 'poe-output-grid';
    row.cells = {};
    for (const id of modelOrder) {
      const section = document.createElement('section');
      section.className = 'poe-output-card';
      const title = document.createElement('h2');
      title.textContent = modelLabels[id];
      const model = document.createElement('p');
      model.className = 'small model-name';
      model.textContent = modelSettings[id] || '';
      const status = document.createElement('p');
      status.className = 'small model-status';
      status.setAttribute('role', 'status');
      const box = document.createElement('textarea');
      box.readOnly = true;
      box.className = 'poe-output-text';
      box.setAttribute('aria-label', modelLabels[id] + ' output');
      box.spellcheck = false;
      const pending = id === 'openai' && openaiConnected === false ? 'not connected' : 'Waiting for response...';
      box.value = pending;
      status.textContent = id === 'minilm' ? 'Retrieved passages (embedding model)' : pending;
      section.append(title, model, status, box);
      grid.append(section);
      row.cells[id] = {model, status, box};
    }
    row.append(grid);
    el('poe-chat').append(row);
    for (const output of outputs) updateOutput(row, output);
    if (matches.length) showComparisonSources(row, matches);
    el('poe-chat').scrollTop = el('poe-chat').scrollHeight;
    return row;
  }
  function updateOutput(row, output) {
    const cell = row.cells[output.id];
    if (!cell) return;
    cell.model.textContent = output.model || modelSettings[output.id] || '';
    cell.box.value = output.text;
    cell.status.textContent = output.status === 'error' ? 'Could not respond'
      : output.status === 'not_connected' ? 'not connected'
      : output.status === 'not_requested' ? 'Not requested'
      : output.id === 'minilm' ? 'Retrieved passages (embedding model)' : 'Complete';
    cell.box.dataset.status = output.status;
  }
  function showComparisonSources(row, matches) {
    if (row.sources) row.sources.remove();
    // Reuse the safe source-link renderer beneath the three output boxes.
    const sources = addMessage('assistant', '', matches);
    sources.className = 'poe-comparison-sources';
    sources.querySelector('.speaker').remove();
    sources.querySelector('.message-text').remove();
    row.append(sources);
    row.sources = sources;
  }
  function render() {
    el('poe-chat').replaceChildren();
    if (!turns.length) addMessage('assistant', 'Ask a question to compare the current local model, MiniLM passages, and OpenAI side by side.');
    for (const turn of turns) {
      addMessage('user', turn.question);
      addComparison(turn.outputs || [{id:'local', text:turn.answer, status:'complete'},
        {id:'minilm', text:'Saved passages are available below.', status:'complete'},
        {id:'openai', text:'Not requested for this earlier prompt.', status:'not_requested'}], turn.matches || []);
    }
  }
  function modelHistory(id) {
    return turns.slice(-6).flatMap(turn => {
      const result = turn.outputs?.find(output => output.id === id);
      const answer = result?.status === 'complete' ? result.text : !turn.outputs && id === 'local' ? turn.answer : '';
      return [{role:'user', content:turn.question.slice(0,6000)},
        ...(answer ? [{role:'assistant', content:answer.slice(0,6000)}] : [])];
    });
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
    const history = modelHistory('local');
    const openai_history = modelHistory('openai');
    addMessage('user', question);
    const row = addComparison();
    const turn = {question, answer:'', outputs:[], matches:[]};
    el('poe-question').value = '';
    setSending(true);
    let complete = false;
    try {
      const response = await fetch('/api/poe', {method:'POST', headers:{'Content-Type':'application/json'},
        body:JSON.stringify({question, history, openai_history, stream:true})});
      if (!response.ok) {
        const error = await response.json();
        throw Error(error.error || 'Poe could not respond. Please try again.');
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      const receive = item => {
        if (item.type === 'sources') {
          turn.matches = item.matches;
          showComparisonSources(row, item.matches);
        } else if (item.type === 'model') {
          turn.outputs = turn.outputs.filter(output => output.id !== item.output.id).concat(item.output);
          updateOutput(row, item.output);
        } else if (item.type === 'complete') {
          Object.assign(turn, item.result);
          complete = true;
        } else if (item.type === 'error') throw Error(item.message);
      };
      while (true) {
        const {value, done} = await reader.read();
        buffer += decoder.decode(value, {stream:!done});
        const lines = buffer.split('\n');
        buffer = lines.pop();
        for (const line of lines) if (line.trim()) receive(JSON.parse(line));
        if (done) break;
      }
      if (buffer.trim()) receive(JSON.parse(buffer));
      if (!complete) throw Error('Connection ended before all outputs completed. Please try again.');
    } catch (error) {
      el('poe-error').textContent = error.message;
      for (const id of modelOrder) {
        if (!turn.outputs.some(output => output.id === id)) {
          const disconnected = id === 'openai' && openaiConnected === false;
          const output = {id, model:modelSettings[id], text:disconnected ? 'not connected' : error.message,
            status:disconnected ? 'not_connected' : 'error'};
          turn.outputs.push(output);
          updateOutput(row, output);
        }
      }
    } finally {
      turn.answer = turn.outputs.find(output => output.id === 'local')?.text || '';
      turns.push(turn);
      turns = turns.slice(-20);
      save();
      setSending(false);
      el('poe-question').focus();
    }
  };
  window.addEventListener('dgrag:state', event => {
    const state = event.detail;
    modelSettings = state.comparison_models || {};
    openaiConnected = modelSettings.openai_connected === true;
    el('poe-readiness').textContent = (!state.active_namespace || state.reindex_required)
      ? 'Load and index all three Poe sources in Library & pipeline to begin a conversation.'
      : !state.generation_ready
        ? 'The library is indexed. The current local model needs its download; MiniLM and connected OpenAI remain available.'
        : 'The library is ready. Each prompt compares local Qwen, MiniLM passages, and OpenAI.';
  });
  render();
})();
