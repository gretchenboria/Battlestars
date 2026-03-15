from flask import Flask, jsonify, request, render_template_string
import os
from pathlib import Path
import re
import json
import subprocess

app = Flask(__name__)
LOGS_DIR = Path("helion/logs")

def parse_benchmarks(content):
    pattern = r"Benchmark \d+: ([\d\.]+) ms .*?  (\{.*\})"
    matches = re.findall(pattern, content)
    chart_data = []
    for mean_ms_str, dict_str in matches:
        try:
            mean_ms = float(mean_ms_str)
            shape_dict = eval(dict_str)
            label = f"({shape_dict.get('B')},{shape_dict.get('T') or shape_dict.get('D') or shape_dict.get('num_tokens')})"
            # We want to format the shape nicely for the UI
            full_shape = "(" + ",".join(str(v) for k,v in shape_dict.items() if k != 'seed') + ")"
            chart_data.append({"shape": full_shape, "time": mean_ms, "metric": "Mean Latency", "done": True})
        except Exception as e:
            pass
    return chart_data

def parse_profiles(content):
    parts = re.split(r"Profile \d+: (\{.*?\})", content)
    chart_data = []
    if len(parts) > 1:
        for i in range(1, len(parts), 2):
            try:
                shape_dict = eval(parts[i])
                table_str = parts[i+1]
                full_shape = "(" + ",".join(str(v) for k,v in shape_dict.items() if k != 'seed') + ")"
                
                match = re.search(r"^\s*_helion_kernel\s+(?:[^\s]+\s+){5}([\d\.]+)(us|ms)\s+([\d\.]+)%", table_str, re.MULTILINE)
                if match:
                    time_val = float(match.group(1))
                    unit = match.group(2)
                    if unit == "us":
                        time_val = time_val / 1000.0
                    chart_data.append({"shape": full_shape, "time": time_val, "metric": f"{match.group(3)}% CUDA", "done": True})
                else:
                    chart_data.append({"shape": full_shape, "time": 0.0, "metric": "Error/Not Found", "done": True})
            except Exception as e:
                pass
    return chart_data

TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Helion Hackathon — GPU Kernel Profiler</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet" />
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    :root {
      --bg-primary:    #080c14;
      --bg-secondary:  #0d1525;
      --bg-card:       #111827;
      --bg-card-hover: #162033;
      --border:        #1e2d45;
      --border-glow:   #1e3a5f;
      --cyan:          #00e5ff;
      --cyan-dim:      #0099bb;
      --green:         #00ff88;
      --green-dim:     #00aa55;
      --purple:        #b57bff;
      --purple-dim:    #7c3aed;
      --orange:        #ff9500;
      --red:           #ff4757;
      --yellow:        #ffd700;
      --text-primary:  #e8f4f8;
      --text-secondary:#7fa8c0;
      --text-muted:    #3d5a70;
      --mono:          'JetBrains Mono', 'Fira Code', monospace;
      --sans:          'Inter', sans-serif;
    }

    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    html { scroll-behavior: smooth; }

    body {
      background: var(--bg-primary);
      color: var(--text-primary);
      font-family: var(--sans);
      min-height: 100vh;
      overflow-x: hidden;
    }

    body::before {
      content: '';
      position: fixed;
      inset: 0;
      background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,229,255,0.012) 2px,
        rgba(0,229,255,0.012) 4px
      );
      pointer-events: none;
      z-index: 9999;
    }

    body::after {
      content: '';
      position: fixed;
      inset: 0;
      background-image:
        linear-gradient(rgba(0,229,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,229,255,0.03) 1px, transparent 1px);
      background-size: 40px 40px;
      pointer-events: none;
      z-index: 0;
    }

    header {
      position: sticky;
      top: 0;
      z-index: 100;
      background: rgba(8,12,20,0.92);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--border);
      padding: 0 2rem;
      height: 64px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      box-shadow: 0 0 40px rgba(0,229,255,0.06);
    }

    .header-left { display: flex; align-items: center; gap: 1rem; }
    .logo-mark { width: 36px; height: 36px; border: 2px solid var(--cyan); border-radius: 6px; display: flex; align-items: center; justify-content: center; position: relative; overflow: hidden; }
    .logo-mark::before { content: ''; position: absolute; width: 100%; height: 2px; background: var(--cyan); animation: scan 2s linear infinite; }
    @keyframes scan { from { top: -2px; } to { top: 100%; } }
    .logo-mark svg { width: 18px; height: 18px; fill: var(--cyan); position: relative; z-index: 1; }
    .header-title { font-family: var(--mono); font-size: 0.9rem; font-weight: 600; color: var(--cyan); letter-spacing: 0.08em; text-transform: uppercase; }
    .header-subtitle { font-family: var(--mono); font-size: 0.65rem; color: var(--text-muted); letter-spacing: 0.12em; text-transform: uppercase; }
    .header-right { display: flex; align-items: center; gap: 2rem; }
    .live-indicator { display: flex; align-items: center; gap: 0.5rem; font-family: var(--mono); font-size: 0.75rem; color: var(--green); }
    .pulse-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); animation: pulse 1.4s ease-in-out infinite; }
    @keyframes pulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; transform: scale(0.7); } }
    .time-display { font-family: var(--mono); font-size: 0.85rem; color: var(--text-primary); letter-spacing: 0.05em; }

    .countdown-banner { background: linear-gradient(135deg, rgba(181,123,255,0.12), rgba(0,229,255,0.08)); border-bottom: 1px solid var(--border); padding: 0.6rem 2rem; display: flex; align-items: center; justify-content: center; gap: 2rem; font-family: var(--mono); position: relative; z-index: 10; }
    .countdown-label { font-size: 0.65rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em; }
    .countdown-time { font-size: 1.4rem; font-weight: 700; color: var(--purple); letter-spacing: 0.08em; text-shadow: 0 0 20px rgba(181,123,255,0.5); }
    .countdown-segment { text-align: center; }
    .countdown-sep { font-size: 1.4rem; color: var(--text-muted); margin-top: -6px; }
    .elapsed-display { font-size: 0.75rem; color: var(--text-secondary); }

    main { position: relative; z-index: 1; max-width: 1600px; margin: 0 auto; padding: 2rem; display: flex; flex-direction: column; gap: 2rem; }

    .section-header { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1.25rem; }
    .section-header h2 { font-family: var(--mono); font-size: 0.8rem; font-weight: 600; color: var(--cyan); text-transform: uppercase; letter-spacing: 0.15em; }
    .section-line { flex: 1; height: 1px; background: linear-gradient(90deg, var(--border), transparent); }
    .section-icon { width: 28px; height: 28px; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-size: 0.8rem; }

    .card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem; transition: border-color 0.3s, box-shadow 0.3s; animation: fadeInUp 0.5s ease both; }
    .card:hover { border-color: var(--border-glow); box-shadow: 0 0 30px rgba(0,229,255,0.06); }
    @keyframes fadeInUp { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: translateY(0); } }

    .leaderboard-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 1rem; }
    .problem-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px; padding: 1.25rem 1.5rem; transition: all 0.3s; animation: fadeInUp 0.5s ease both; position: relative; overflow: hidden; cursor: pointer; }
    .problem-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, transparent, var(--accent-color, var(--cyan)), transparent); }
    .problem-card:hover { border-color: var(--accent-color, var(--cyan)); box-shadow: 0 0 24px color-mix(in srgb, var(--accent-color, var(--cyan)) 20%, transparent); transform: translateY(-2px); }
    .problem-card.active { border-color: var(--cyan); box-shadow: 0 0 30px rgba(0,229,255,0.2); }
    
    .problem-card:nth-child(1) { --accent-color: var(--green); animation-delay: 0.05s; }
    .problem-card:nth-child(2) { --accent-color: var(--cyan);  animation-delay: 0.10s; }
    .problem-card:nth-child(3) { --accent-color: var(--purple);animation-delay: 0.15s; }
    .problem-card:nth-child(4) { --accent-color: var(--purple);animation-delay: 0.20s; }
    .problem-card:nth-child(5) { --accent-color: var(--orange);animation-delay: 0.25s; }

    .problem-name { font-family: var(--mono); font-size: 0.8rem; font-weight: 600; color: var(--text-primary); margin-bottom: 0.5rem; word-break: break-all; }
    .problem-meta { display: flex; align-items: center; justify-content: space-between; margin-top: 0.75rem; }
    .badge { font-family: var(--mono); font-size: 0.65rem; font-weight: 600; padding: 0.25em 0.75em; border-radius: 20px; text-transform: uppercase; letter-spacing: 0.08em; }
    .badge-submitted { background: rgba(0,255,136,0.12); color: var(--green); border: 1px solid rgba(0,255,136,0.3); }
    .badge-tuning { background: rgba(0,229,255,0.12); color: var(--cyan); border: 1px solid rgba(0,229,255,0.3); animation: glow-badge 2s ease-in-out infinite; }
    @keyframes glow-badge { 0%, 100% { box-shadow: none; } 50% { box-shadow: 0 0 10px rgba(0,229,255,0.3); } }
    .badge-progress { background: rgba(255,149,0,0.12); color: var(--orange); border: 1px solid rgba(255,149,0,0.3); }
    .badge-pending { background: rgba(61,90,112,0.2); color: var(--text-muted); border: 1px solid var(--border); }
    .score-display { font-family: var(--mono); font-size: 0.9rem; font-weight: 700; color: var(--accent-color, var(--cyan)); }
    .shapes-count { font-family: var(--mono); font-size: 0.65rem; color: var(--text-muted); margin-top: 0.25rem; }
    .progress-bar-wrap { margin-top: 0.75rem; background: var(--bg-secondary); border-radius: 4px; height: 4px; overflow: hidden; }
    .progress-bar-fill { height: 100%; border-radius: 4px; background: linear-gradient(90deg, var(--accent-color, var(--cyan)), color-mix(in srgb, var(--accent-color, var(--cyan)) 60%, white)); transition: width 1s ease; }

    .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
    @media (max-width: 1100px) { .two-col { grid-template-columns: 1fr; } }
    .chart-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem; animation: fadeInUp 0.5s ease 0.2s both; }
    .chart-wrapper { position: relative; height: 280px; margin-top: 1rem; }
    .autotune-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem; animation: fadeInUp 0.5s ease 0.25s both; overflow-x: auto; max-height: 400px; }
    
    table { width: 100%; border-collapse: collapse; font-family: var(--mono); font-size: 0.72rem; }
    thead tr th { text-align: left; padding: 0.5rem 0.75rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em; border-bottom: 1px solid var(--border); font-weight: 500; }
    tbody tr { transition: background 0.2s; }
    tbody tr:hover { background: var(--bg-card-hover); }
    tbody tr td { padding: 0.6rem 0.75rem; border-bottom: 1px solid rgba(30,45,69,0.5); vertical-align: middle; }
    .td-shape { color: var(--text-secondary); font-size: 0.68rem; }
    .td-time { font-weight: 600; }
    .td-time.best { color: var(--green); text-shadow: 0 0 8px rgba(0,255,136,0.4); }
    .td-time.tbd { color: var(--text-muted); }
    .td-config { color: var(--cyan-dim); }
    .td-acf { color: var(--purple); }
    .td-status-dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; margin-right: 0.35rem; }
    .dot-done { background: var(--green); }
    .dot-tuning { background: var(--cyan); animation: pulse 1.4s ease-in-out infinite; }
    .sparkbar { display: inline-block; height: 6px; border-radius: 3px; background: var(--green); vertical-align: middle; margin-left: 0.5rem; transition: width 1s ease; }
    .sparkbar.tbd { background: var(--border); width: 30px !important; }

    .run-panel { background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem; animation: fadeInUp 0.5s ease 0.35s both; display: flex; gap: 1rem; align-items: center; }
    .run-panel select, .run-panel button { padding: 0.5rem 1rem; font-family: var(--mono); font-size: 0.8rem; background: var(--bg-secondary); color: var(--text-primary); border: 1px solid var(--border); border-radius: 6px; }
    .run-panel button { background: var(--cyan); color: var(--bg-primary); font-weight: bold; cursor: pointer; transition: 0.2s; }
    .run-panel button:hover { background: var(--cyan-dim); }
    .run-panel button:disabled { background: var(--text-muted); cursor: not-allowed; }

    footer { position: relative; z-index: 1; text-align: center; padding: 1.5rem 2rem; font-family: var(--mono); font-size: 0.65rem; color: var(--text-muted); border-top: 1px solid var(--border); letter-spacing: 0.08em; }
    .glow-cyan { text-shadow: 0 0 12px rgba(0,229,255,0.6); }
    .glow-green { text-shadow: 0 0 12px rgba(0,255,136,0.6); }
    .glow-purple { text-shadow: 0 0 12px rgba(181,123,255,0.6); }
    .tag { display: inline-flex; align-items: center; gap: 0.35rem; font-family: var(--mono); font-size: 0.6rem; padding: 0.2em 0.6em; border-radius: 4px; text-transform: uppercase; letter-spacing: 0.08em; }
    .tag-b200 { background: rgba(181,123,255,0.12); color: var(--purple); border: 1px solid rgba(181,123,255,0.25); }
    .tag-triton { background: rgba(0,229,255,0.10); color: var(--cyan); border: 1px solid rgba(0,229,255,0.2); }
  </style>
</head>
<body>

<header>
  <div class="header-left">
    <div class="logo-mark">
      <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>
    </div>
    <div>
      <div class="header-title">GPU Kernel Profiler</div>
      <div class="header-subtitle">Helion Hackathon · B200 Target</div>
    </div>
  </div>
  <div class="header-right">
    <span class="tag tag-b200">NVIDIA B200</span>
    <span class="tag tag-triton">Triton</span>
    <div class="live-indicator">
      <span class="pulse-dot"></span>
      <span id="polling-status">LIVE</span>
    </div>
    <div class="time-display" id="clock">--:--:--</div>
  </div>
</header>

<div class="countdown-banner">
  <div>
    <div class="countdown-label">Hackathon Start</div>
    <div style="font-family:var(--mono);font-size:0.75rem;color:var(--text-secondary)" id="start-display">Mar 14 2026 · 09:00 AM PT</div>
  </div>
  <div style="display:flex;align-items:center;gap:0.5rem">
    <div class="countdown-segment"><div class="countdown-time" id="cd-h">--</div><div class="countdown-label">hrs</div></div>
    <div class="countdown-sep">:</div>
    <div class="countdown-segment"><div class="countdown-time" id="cd-m">--</div><div class="countdown-label">min</div></div>
    <div class="countdown-sep">:</div>
    <div class="countdown-segment"><div class="countdown-time" id="cd-s">--</div><div class="countdown-label">sec</div></div>
  </div>
  <div style="text-align:right">
    <div class="countdown-label">Deadline</div>
    <div style="font-family:var(--mono);font-size:0.75rem;color:var(--red)">Mar 14 2026 · 06:00 PM PT</div>
  </div>
</div>

<main>

  <div class="run-panel">
      <span style="font-family:var(--mono);font-size:0.8rem;color:var(--cyan);font-weight:bold;text-transform:uppercase;letter-spacing:0.1em;">Experiment Trigger</span>
      <select id="run-mode">
          <option value="benchmark">Benchmark</option>
          <option value="profile">Profile</option>
          <option value="test">Test Correctness</option>
      </select>
      <select id="run-problem" style="flex:1">
          <option value="causal_conv1d_py">causal_conv1d_py</option>
          <option value="fp8_quant_py">fp8_quant_py</option>
          <option value="gated_deltanet_chunk_fwd_h_py">gated_deltanet_chunk_fwd_h_py</option>
          <option value="gated_deltanet_chunk_fwd_o_py">gated_deltanet_chunk_fwd_o_py</option>
          <option value="gated_deltanet_recompute_w_u_py">gated_deltanet_recompute_w_u_py</option>
      </select>
      <button id="run-btn" onclick="runExperiment()">Deploy to B200</button>
      <span id="run-status" style="font-family:var(--mono);font-size:0.7rem;color:var(--text-muted);margin-left:1rem;"></span>
  </div>

  <section>
    <div class="section-header">
      <div class="section-icon" style="background:rgba(0,255,136,0.1);">🏆</div>
      <h2>Select Kernel to Analyze</h2>
      <div class="section-line"></div>
    </div>
    <div class="leaderboard-grid">
      <div class="problem-card" id="card-fp8" onclick="filterLogs('fp8')">
        <div class="problem-name">fp8_quant</div>
        <div class="shapes-count">View Logs</div>
      </div>
      <div class="problem-card" id="card-causal" onclick="filterLogs('causal')">
        <div class="problem-name">causal_conv1d</div>
        <div class="shapes-count">View Logs</div>
      </div>
      <div class="problem-card" id="card-gdn-o" onclick="filterLogs('gated_deltanet_chunk_fwd_o')">
        <div class="problem-name">gdn_chunk_fwd_o</div>
        <div class="shapes-count">View Logs</div>
      </div>
      <div class="problem-card" id="card-gdn-h" onclick="filterLogs('gated_deltanet_chunk_fwd_h')">
        <div class="problem-name">gdn_chunk_fwd_h</div>
        <div class="shapes-count">View Logs</div>
      </div>
      <div class="problem-card" id="card-gdn-wu" onclick="filterLogs('gated_deltanet_recompute')">
        <div class="problem-name">gdn_recompute_w_u</div>
        <div class="shapes-count">View Logs</div>
      </div>
    </div>
  </section>

  <div class="two-col">
    <div class="chart-card" id="chart-wrapper" style="display:none;">
      <div class="section-header" style="margin-bottom:0">
        <div class="section-icon" style="background:rgba(0,229,255,0.1);">📊</div>
        <h2 id="chart-title">Performance Metrics</h2>
        <div class="section-line"></div>
        <span id="chart-subtitle" style="font-family:var(--mono);font-size:0.6rem;color:var(--text-muted)"></span>
      </div>
      <div class="chart-wrapper">
        <canvas id="perfChart"></canvas>
      </div>
    </div>

    <div class="autotune-card" id="table-wrapper" style="display:none;">
      <table>
        <thead>
          <tr>
            <th>Shape Configuration</th>
            <th>Execution Time</th>
            <th>Metric</th>
          </tr>
        </thead>
        <tbody id="autotune-tbody">
        </tbody>
      </table>
    </div>
  </div>
</main>

<footer>
  GPU KERNEL PROFILER · HELION HACKATHON 2026 · NVIDIA B200 · Built with Chart.js + Vanilla JS
</footer>

<script>
let currentLog = null;
let chartInstance = null;
let currentFilter = '';

async function fetchLogs() {
    try {
        const res = await fetch('/api/logs');
        const data = await res.json();
        
        // Auto-select latest log if none selected or if matching filter
        if (data.logs.length > 0) {
            let logToSelect = data.logs[0];
            if (currentFilter) {
                const filtered = data.logs.filter(l => l.includes(currentFilter));
                if (filtered.length > 0) {
                    logToSelect = filtered[0];
                }
            }
            if (currentLog !== logToSelect) {
                selectLog(logToSelect);
            }
        }
    } catch (e) {}
}

function filterLogs(prefix) {
    currentFilter = prefix;
    document.querySelectorAll('.problem-card').forEach(c => c.classList.remove('active'));
    document.getElementById('card-' + prefix.split('_')[0])?.classList.add('active');
    fetchLogs();
}

async function selectLog(filename) {
    currentLog = filename;
    document.getElementById('chart-subtitle').textContent = filename;
    
    try {
        const res = await fetch(`/api/log/${filename}`);
        const data = await res.json();
        
        const chartWrapper = document.getElementById('chart-wrapper');
        const tableWrapper = document.getElementById('table-wrapper');
        
        if (data.parsed_data && data.parsed_data.length > 0) {
            chartWrapper.style.display = 'block';
            tableWrapper.style.display = 'block';
            renderTable(data.parsed_data);
            renderChart(data.parsed_data, filename.includes('profile'));
        } else {
            chartWrapper.style.display = 'none';
            tableWrapper.style.display = 'none';
            if (chartInstance) chartInstance.destroy();
        }
    } catch (e) {
        console.error(e);
    }
}

function renderTable(parsed_data) {
    const tbody = document.getElementById('autotune-tbody');
    tbody.innerHTML = '';
    
    let minTime = Math.min(...parsed_data.filter(d => d.time > 0).map(d => d.time));
    if (minTime === Infinity) minTime = 0.001;

    parsed_data.forEach((d, i) => {
        const tr = document.createElement('tr');
        tr.style.animationDelay = (i * 0.04) + 's';
        tr.style.animation = 'fadeInUp 0.4s ease both';
        
        const isBest = d.time > 0 && d.time <= minTime;
        const valStr = d.time > 0 ? d.time.toFixed(4) + ' ms' : 'N/A';
        
        tr.innerHTML = `
            <td class="td-shape">${d.shape}</td>
            <td class="td-time ${isBest ? 'best' : ''}">
                ${valStr} ${isBest ? '<span style="color:var(--yellow);margin-left:0.35rem;">★</span>' : ''}
            </td>
            <td class="td-config">${d.metric}</td>
        `;
        tbody.appendChild(tr);
    });
}

function renderChart(parsed_data, isProfile) {
    const ctx = document.getElementById('perfChart').getContext('2d');
    if (chartInstance) chartInstance.destroy();
    
    const minTime = Math.min(...parsed_data.filter(d => d.time > 0).map(d => d.time));
    
    const colors = parsed_data.map(d => 
        (d.time > 0 && d.time <= minTime) ? 'rgba(0,255,136,0.85)' : 'rgba(0,229,255,0.65)'
    );
    const borderColors = parsed_data.map(d => 
        (d.time > 0 && d.time <= minTime) ? '#00ff88' : '#00e5ff'
    );

    chartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: parsed_data.map(d => d.shape),
            datasets: [{
                label: isProfile ? '_helion_kernel ms' : 'Mean ms',
                data: parsed_data.map(d => d.time),
                backgroundColor: colors,
                borderColor: borderColors,
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 1200, easing: 'easeOutQuart' },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#162033',
                    borderColor: '#1e3a5f',
                    borderWidth: 1,
                    titleColor: '#00e5ff',
                    bodyColor: '#7fa8c0',
                    titleFont: { family: "'JetBrains Mono', monospace", size: 11 },
                    bodyFont:  { family: "'JetBrains Mono', monospace", size: 11 }
                }
            },
            scales: {
                x: { ticks: { color: '#7fa8c0', font: { family: "'JetBrains Mono', monospace", size: 9 } }, grid: { color: 'rgba(30,45,69,0.5)' } },
                y: { ticks: { color: '#7fa8c0', font: { family: "'JetBrains Mono', monospace", size: 10 } }, grid: { color: 'rgba(30,45,69,0.5)' }, border: { dash: [4, 4] }, beginAtZero: true }
            }
        }
    });
}

async function runExperiment() {
    const mode = document.getElementById('run-mode').value;
    const problem = document.getElementById('run-problem').value;
    const btn = document.getElementById('run-btn');
    const status = document.getElementById('run-status');
    
    btn.disabled = true;
    btn.textContent = 'Running...';
    status.textContent = 'Deploying job to B200 instance (Expect 30s - 2m)...';
    document.getElementById('polling-status').textContent = 'WORKING...';
    document.getElementById('polling-status').style.color = 'var(--orange)';
    
    try {
        const res = await fetch('/api/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode, problem })
        });
        status.textContent = 'Job complete! Syncing logs...';
        filterLogs(problem);
    } catch (e) {
        status.textContent = 'Failed to deploy experiment.';
    } finally {
        btn.disabled = false;
        btn.textContent = 'Deploy to B200';
        document.getElementById('polling-status').textContent = 'LIVE';
        document.getElementById('polling-status').style.color = 'var(--green)';
    }
}

// Clock logic
function pad2(n) { return String(n).padStart(2,'0'); }
function tick() {
    const now = new Date();
    document.getElementById('clock').textContent = pad2(now.getHours()) + ':' + pad2(now.getMinutes()) + ':' + pad2(now.getSeconds());
}
setInterval(tick, 1000);
setInterval(fetchLogs, 5000);
tick();
fetchLogs();
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(TEMPLATE)

@app.route('/api/logs')
def api_logs():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logs = sorted([f.name for f in LOGS_DIR.glob('*.log')], reverse=True)
    return jsonify({"logs": logs})

@app.route('/api/log/<filename>')
def api_log(filename):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    file_path = LOGS_DIR / filename
    if not file_path.exists():
        return jsonify({"error": "File not found"}), 404
        
    content = file_path.read_text()
    parsed_data = []
    if "profile" in filename.lower() or "Profile" in content:
        parsed_data = parse_profiles(content)
    elif "benchmark" in filename.lower() or "Benchmark" in content:
        parsed_data = parse_benchmarks(content)
    
    return jsonify({"content": content, "parsed_data": parsed_data})

@app.route('/api/run', methods=['POST'])
def run_experiment():
    data = request.json
    mode = data.get('mode')
    problem = data.get('problem')
    if not mode or not problem:
        return jsonify({"error": "Missing mode or problem"}), 400
    try:
        subprocess.run(["./run_experiment.sh", mode, problem], check=True)
        return jsonify({"status": "success"})
    except subprocess.CalledProcessError as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Advanced Helion UI...")
    print("Open http://127.0.0.1:8080 in your browser.")
    app.run(debug=True, port=8080)
