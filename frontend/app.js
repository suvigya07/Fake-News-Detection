const textarea    = document.getElementById('newsInput');
const charCount   = document.getElementById('charCount');
const analyseBtn  = document.getElementById('analyseBtn');
const errorBox    = document.getElementById('errorBox');
const resultSec   = document.getElementById('resultSection');
const loadingSec  = document.getElementById('loadingSection');

textarea.addEventListener('input', () => {
  const n = textarea.value.length;
  charCount.textContent = `${n.toLocaleString()} character${n !== 1 ? 's' : ''}`;
});

textarea.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') analyse();
});

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.classList.remove('hidden');
  resultSec.classList.add('hidden');
  loadingSec.classList.add('hidden');
}

function setLoading(on) {
  if (on) {
    loadingSec.classList.remove('hidden');
    resultSec.classList.add('hidden');
    errorBox.classList.add('hidden');
    analyseBtn.disabled = true;
  } else {
    loadingSec.classList.add('hidden');
    analyseBtn.disabled = false;
  }
}

async function analyse() {
  const text = textarea.value.trim();
  if (text.length < 20) {
    showError('Please enter at least 20 characters.');
    return;
  }

  setLoading(true);

  try {
    const resp = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });

    const data = await resp.json();

    if (!resp.ok || data.error) {
      showError(data.error || 'Something went wrong. Please try again.');
      setLoading(false);
      return;
    }

    setLoading(false);
    renderResult(data, text);
  } catch (err) {
    showError('Could not reach the server. Make sure it is running on port 3000.');
    setLoading(false);
  }
}

function renderResult(data, originalText) {
  const isReal   = data.label === 'REAL';
  const cls      = isReal ? 'real' : 'fake';
  const conf     = data.confidence;

  // Verdict card
  const card = document.getElementById('verdictCard');
  card.className = `verdict-card ${cls}`;

  document.getElementById('verdictValue').textContent    = data.label;
  document.getElementById('verdictValue').className      = `verdict-value ${cls}`;
  document.getElementById('verdictSub').textContent      = `${conf}% confidence`;

  // Gauge — arc length is 173px for a half-circle
  const fill   = document.getElementById('gaugeFill');
  const pct    = document.getElementById('gaugePct');
  const offset = 173 - (conf / 100) * 173;
  fill.className = `gauge-fill ${cls}`;
  setTimeout(() => { fill.style.strokeDashoffset = offset; }, 50);
  pct.textContent = `${conf}%`;

  // Stats
  document.getElementById('fakePct').textContent  = `${data.fake_probability}%`;
  document.getElementById('realPct').textContent  = `${data.real_probability}%`;
  document.getElementById('wordCount').textContent = data.word_count;

  // Highlight tags
  const tagContainer = document.getElementById('highlightTags');
  tagContainer.innerHTML = '';
  (data.highlights || []).forEach(h => {
    const tag = document.createElement('span');
    tag.className = `tag ${h.direction}`;
    tag.textContent = h.word;
    tag.title = `strength: ${h.strength}`;
    tagContainer.appendChild(tag);
  });

  // Annotated text
  const annotated = document.getElementById('annotatedText');
  annotated.innerHTML = annotateText(originalText, data.highlights || []);

  resultSec.classList.remove('hidden');
  resultSec.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function annotateText(text, highlights) {
  if (!highlights.length) return escapeHtml(text);

  // Build a map of word → direction for fast lookup
  const wordMap = {};
  highlights.forEach(h => { wordMap[h.word.toLowerCase()] = h.direction; });

  // Tokenise preserving whitespace/punctuation
  const tokens = text.split(/(\s+|[.,!?;:'"()\-–—])/);

  return tokens.map(token => {
    const key = token.toLowerCase().replace(/[^a-z]/g, '');
    if (key && wordMap[key]) {
      return `<mark class="${wordMap[key]}">${escapeHtml(token)}</mark>`;
    }
    return escapeHtml(token);
  }).join('');
}

function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
