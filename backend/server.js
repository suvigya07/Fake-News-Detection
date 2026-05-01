const express = require('express');
const cors    = require('cors');
const { spawn } = require('child_process');
const path    = require('path');

const app  = express();
const PORT = 3000;

app.use(cors());
app.use(express.json());

// Serve static frontend files
app.use(express.static(path.join(__dirname, '..', 'frontend')));

const ML_DIR     = path.join(__dirname, '..', 'ml');
const PYTHON_CMD = process.platform === 'win32' ? 'python' : 'python3';

app.post('/api/predict', (req, res) => {
  const { text } = req.body;

  if (!text || text.trim().length < 20) {
    return res.status(400).json({ error: 'Please provide at least 20 characters of text.' });
  }

  if (text.length > 50000) {
    return res.status(400).json({ error: 'Text too long. Please keep it under 50,000 characters.' });
  }

  const payload = JSON.stringify({ text });
  const py = spawn(PYTHON_CMD, [path.join(ML_DIR, 'predict.py')], {
    cwd: ML_DIR
  });

  let stdout = '';
  let stderr = '';

  py.stdin.write(payload);
  py.stdin.end();

  py.stdout.on('data', chunk => { stdout += chunk.toString(); });
  py.stderr.on('data', chunk => { stderr += chunk.toString(); });

  py.on('close', code => {
    if (code !== 0) {
      console.error('Python error:', stderr);
      return res.status(500).json({ error: 'Prediction failed. Make sure train.py has been run.' });
    }
    try {
      const result = JSON.parse(stdout.trim());
      if (result.error) return res.status(400).json(result);
      res.json(result);
    } catch (e) {
      console.error('Parse error:', stdout, stderr);
      res.status(500).json({ error: 'Could not parse prediction output.' });
    }
  });

  py.on('error', err => {
    console.error('Failed to start Python:', err);
    res.status(500).json({ error: `Could not start Python. Is "${PYTHON_CMD}" in your PATH?` });
  });
});

app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, '..', 'frontend', 'index.html'));
});

app.listen(PORT, () => {
  console.log(`\n  Fake News Detector running at http://localhost:${PORT}\n`);
});
