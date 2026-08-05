/* M51 S5: voice-test.eu Frontend-Logik (Vanilla JS, keine Libraries) */

const State = {
  samples: [],
  texts: null,
  currentVariant: 'v1_simple',
  currentLang: '',
  currentProvider: '',
  currentGender: '',
  currentLicense: '',
  currentSort: 'provider',
  searchQuery: '',
  audioSingle: new Audio(),
  currentPlayingId: null,
  compareA: null,
  compareB: null,
  audioA: document.getElementById('audio-a'),
  audioB: document.getElementById('audio-b'),
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const PROVIDER_ORDER = ['google', 'aws', 'azure', 'openai', 'elevenlabs', 'edge', 'knowlez', 'piper', 'kokoro', 'bark', 'xtts'];

function providerDisplay(p) {
  const map = {
    google: 'Google Cloud TTS', aws: 'Amazon Polly', azure: 'Microsoft Azure',
    openai: 'OpenAI', elevenlabs: 'ElevenLabs', edge: 'Microsoft Edge TTS',
    knowlez: 'Knowlez', piper: 'Piper (rhasspy)',
    xtts: 'Coqui XTTS-v2', kokoro: 'Kokoro-82M', bark: 'Suno Bark',
  };
  return map[p] || p;
}

function providerOSS(p) {
  return ['piper', 'xtts', 'kokoro', 'bark'].includes(p);
}

async function loadData() {
  const [samplesRes, textsRes] = await Promise.all([
    fetch('/data/samples.json').then(r => r.json()),
    fetch('/data/texts.json').then(r => r.json()),
  ]);
  State.samples = samplesRes.samples || [];
  State.texts = textsRes;
  $('#total-count').textContent = State.samples.length.toLocaleString('de-DE');
}

function populateFilters() {
  // Sprachen
  const langs = [...new Set(State.samples.map(s => s.language))].sort();
  const langNames = {};
  State.texts.languages.forEach(l => langNames[l.code] = l.native || l.name);
  const langSelect = $('#f-lang');
  langs.forEach(code => {
    const opt = document.createElement('option');
    opt.value = code;
    opt.textContent = `${langNames[code] || code} (${code})`;
    langSelect.appendChild(opt);
  });

  // Provider
  const providers = [...new Set(State.samples.map(s => s.provider))]
    .sort((a, b) => PROVIDER_ORDER.indexOf(a) - PROVIDER_ORDER.indexOf(b));
  const provSelect = $('#f-provider');
  providers.forEach(p => {
    const opt = document.createElement('option');
    opt.value = p;
    opt.textContent = providerDisplay(p);
    provSelect.appendChild(opt);
  });
}

function getFilteredSamples() {
  let result = State.samples.filter(s => {
    if (s.variant !== State.currentVariant) return false;
    if (State.currentLang && s.language !== State.currentLang) return false;
    if (State.currentProvider && s.provider !== State.currentProvider) return false;
    if (State.currentGender && s.gender !== State.currentGender) return false;
    if (State.currentLicense === 'oss' && !providerOSS(s.provider)) return false;
    if (State.currentLicense === 'proprietary' && providerOSS(s.provider)) return false;
    if (State.searchQuery) {
      const q = State.searchQuery.toLowerCase();
      const hay = `${s.voice_name} ${s.voice_id} ${s.provider}`.toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });

  // Sort
  const sort = State.currentSort;
  if (sort === 'provider') {
    result.sort((a, b) => {
      const pd = PROVIDER_ORDER.indexOf(a.provider) - PROVIDER_ORDER.indexOf(b.provider);
      if (pd !== 0) return pd;
      return a.voice_name.localeCompare(b.voice_name);
    });
  } else if (sort === 'voice') {
    result.sort((a, b) => a.voice_name.localeCompare(b.voice_name));
  } else if (sort === 'language') {
    result.sort((a, b) => a.language.localeCompare(b.language));
  } else if (sort === 'random') {
    result.sort(() => Math.random() - 0.5);
  }
  return result;
}

function renderTextPreview() {
  const variant = State.currentVariant;
  const lang = State.currentLang || 'de';
  const text = State.texts?.texts?.[variant]?.[lang];
  const label = variant === 'v1_simple' ? 'Einfach' :
                variant === 'v2_medium' ? 'Mittel' : 'Emotional';
  $('#text-preview').textContent = text
    ? `[${label} · ${lang}] "${text}"`
    : `[${label}] Kein Text für ${lang}`;
}

function renderList() {
  const samples = getFilteredSamples();
  const list = $('#samples-list');
  const count = samples.length;

  $('#result-count').textContent = `${count} Stimmen`;

  if (count === 0) {
    list.innerHTML = `<div class="state"><div>Keine Stimmen gefunden für die aktuellen Filter.</div></div>`;
    return;
  }

  // Limit auf 200 fuer Performance (Paginierung folgt in spaterer Version)
  const display = samples.slice(0, 200);
  const langNames = {};
  State.texts.languages.forEach(l => langNames[l.code] = l.name);

  list.innerHTML = display.map(s => {
    const playingClass = State.currentPlayingId === s.id ? ' playing' : '';
    const cmpAClass = State.compareA?.id === s.id ? ' compare-a' : '';
    const cmpBClass = State.compareB?.id === s.id ? ' compare-b' : '';
    const abAActive = State.compareA?.id === s.id ? ' active-a' : '';
    const abBActive = State.compareB?.id === s.id ? ' active-b' : '';
    const gender_de = {female: 'Weiblich', male: 'Männlich', neutral: 'Neutral', unknown: '?'}[s.gender] || s.gender;
    return `
      <div class="sample-card${playingClass}${cmpAClass}${cmpBClass}" data-id="${s.id}">
        <button class="play-btn" data-action="play">▶</button>
        <div class="info">
          <div class="row1">
            <span class="voice-name">${escapeHtml(s.voice_name)}</span>
            <span class="voice-id">${escapeHtml(s.voice_id)}</span>
          </div>
          <div class="row2">
            <span class="tag">${providerDisplay(s.provider)}</span>
            <span class="tag">${langNames[s.language] || s.language} (${s.language})</span>
            <span class="tag">${gender_de}</span>
            <span class="tag">${escapeHtml(s.model_type || '')}</span>
            ${s.model_size_mb ? `<span class="tag">${s.model_size_mb} MB</span>` : ''}
            ${s.provider_url ? `<a class="provider-link" href="${s.provider_url}" target="_blank" rel="noopener">↗ Source</a>` : ''}
          </div>
        </div>
        <div class="actions">
          <button data-action="set-a"${abAActive}>A</button>
          <button data-action="set-b"${abBActive}>B</button>
        </div>
      </div>
    `;
  }).join('') + (samples.length > 200
    ? `<div class="state"><div>${samples.length - 200} weitere — verfeinere die Filter um alle zu sehen.</div></div>`
    : '');
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

// --- Playback ---
function playSample(sampleId) {
  const s = State.samples.find(x => x.id === sampleId);
  if (!s) return;
  // Toggle: wenn gerade spielt, stoppen
  if (State.currentPlayingId === sampleId && !State.audioSingle.paused) {
    State.audioSingle.pause();
    return;
  }
  State.audioSingle.src = '/' + s.audio_path;
  State.audioSingle.play().catch(e => console.warn('play failed:', e));
  State.currentPlayingId = sampleId;
  document.querySelectorAll('.sample-card').forEach(c => {
    c.classList.toggle('playing', c.dataset.id === sampleId);
    const btn = c.querySelector('[data-action="play"]');
    if (btn) btn.textContent = c.dataset.id === sampleId ? '⏸' : '▶';
  });
}

State.audioSingle.addEventListener('ended', () => {
  State.currentPlayingId = null;
  document.querySelectorAll('.sample-card').forEach(c => {
    c.classList.remove('playing');
    const btn = c.querySelector('[data-action="play"]');
    if (btn) btn.textContent = '▶';
  });
});
State.audioSingle.addEventListener('pause', () => {
  document.querySelectorAll('.sample-card.playing [data-action="play"]').forEach(btn => {
    btn.textContent = '▶';
  });
});

// --- A/B Vergleich ---
function setCompareSlot(slot, sampleId) {
  const s = State.samples.find(x => x.id === sampleId);
  if (!s) return;
  if (slot === 'a') {
    State.compareA = s;
    State.audioA.src = '/' + s.audio_path;
    $('#slot-a').classList.remove('empty');
    $('#slot-a').querySelector('.name').textContent = `${s.voice_name} (${providerDisplay(s.provider)})`;
  } else {
    State.compareB = s;
    State.audioB.src = '/' + s.audio_path;
    $('#slot-b').classList.remove('empty');
    $('#slot-b').querySelector('.name').textContent = `${s.voice_name} (${providerDisplay(s.provider)})`;
  }
  $('#compare-bar').classList.add('visible');
  renderList();  // um Active-Klassen zu updaten
}

function closeCompare() {
  State.compareA = null;
  State.compareB = null;
  State.audioA.pause();
  State.audioB.pause();
  State.audioA.src = '';
  State.audioB.src = '';
  $('#slot-a').classList.add('empty');
  $('#slot-b').classList.add('empty');
  $('#slot-a').querySelector('.name').textContent = 'Klicke [A] bei einer Stimme';
  $('#slot-b').querySelector('.name').textContent = 'Klicke [B] bei einer Stimme';
  $('#compare-bar').classList.remove('visible');
  renderList();
}

function playBoth() {
  State.audioA.currentTime = 0;
  State.audioA.play().catch(e => console.warn(e));
  State.audioA.onended = () => {
    State.audioB.currentTime = 0;
    State.audioB.play().catch(e => console.warn(e));
    State.audioA.onended = null;
  };
}

// --- Event Handler ---
function bindEvents() {
  $('#search').addEventListener('input', e => {
    State.searchQuery = e.target.value;
    renderList();
  });
  $('#f-lang').addEventListener('change', e => {
    State.currentLang = e.target.value;
    renderTextPreview();
    renderList();
  });
  $('#f-provider').addEventListener('change', e => {
    State.currentProvider = e.target.value;
    renderList();
  });
  $('#f-gender').addEventListener('change', e => {
    State.currentGender = e.target.value;
    renderList();
  });
  $('#f-license').addEventListener('change', e => {
    State.currentLicense = e.target.value;
    renderList();
  });
  $('#f-sort').addEventListener('change', e => {
    State.currentSort = e.target.value;
    renderList();
  });

  $$('.variant-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      $$('.variant-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      State.currentVariant = btn.dataset.variant;
      renderTextPreview();
      renderList();
    });
  });

  // Click-Delegation für Sample-Karten
  $('#samples-list').addEventListener('click', e => {
    const card = e.target.closest('.sample-card');
    if (!card) return;
    const id = card.dataset.id;
    const action = e.target.dataset.action;
    if (action === 'play') {
      playSample(id);
    } else if (action === 'set-a') {
      if (State.compareA?.id === id) {
        State.compareA = null;
        $('#slot-a').classList.add('empty');
        $('#slot-a').querySelector('.name').textContent = 'Klicke [A] bei einer Stimme';
        State.audioA.src = '';
        if (!State.compareB) $('#compare-bar').classList.remove('visible');
      } else {
        setCompareSlot('a', id);
      }
      renderList();
    } else if (action === 'set-b') {
      if (State.compareB?.id === id) {
        State.compareB = null;
        $('#slot-b').classList.add('empty');
        $('#slot-b').querySelector('.name').textContent = 'Klicke [B] bei einer Stimme';
        State.audioB.src = '';
        if (!State.compareA) $('#compare-bar').classList.remove('visible');
      } else {
        setCompareSlot('b', id);
      }
      renderList();
    }
  });

  $('#close-compare').addEventListener('click', closeCompare);
  $('#play-both').addEventListener('click', playBoth);
}

async function init() {
  try {
    await loadData();
    populateFilters();
    bindEvents();
    renderTextPreview();
    renderList();
    $('#loading').style.display = 'none';
  } catch (e) {
    $('#loading').innerHTML = `<div style="color: var(--red);">Fehler beim Laden: ${e.message}</div>`;
    console.error(e);
  }
}

init();
