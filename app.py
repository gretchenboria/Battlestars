from flask import Flask, jsonify, request, render_template_string
import os
from pathlib import Path
import re
import json

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
            label = " | ".join(f"{k}={v}" for k, v in shape_dict.items() if k != 'seed')
            chart_data.append({"label": label, "value": mean_ms, "metric": "Mean ms"})
        except Exception as e:
            print(f"Error parsing benchmark line: {e}")
    return chart_data

def parse_profiles(content):
    # Split content by Profile headers
    parts = re.split(r"Profile \d+: (\{.*?\})", content)
    chart_data = []
    
    if len(parts) > 1:
        for i in range(1, len(parts), 2):
            try:
                shape_str = parts[i]
                table_str = parts[i+1]
                
                shape_dict = eval(shape_str)
                label = " | ".join(f"{k}={v}" for k, v in shape_dict.items() if k != 'seed')
                
                # Extract _helion_kernel CUDA time (it's usually in us or ms)
                # We look for the row starting with _helion_kernel
                match = re.search(r"^\s*_helion_kernel\s+(?:[\d\.]+%?\s+){4}[\d\.]+[a-z]+\s+([\d\.]+)(us|ms)\s+([\d\.]+)%", table_str, re.MULTILINE)
                
                if match:
                    time_val = float(match.group(1))
                    unit = match.group(2)
                    pct = float(match.group(3))
                    
                    # Convert to ms
                    if unit == "us":
                        time_val = time_val / 1000.0
                        
                    chart_data.append({
                        "label": label,
                        "value": time_val,
                        "metric": f"_helion_kernel ms ({pct}%)"
                    })
                else:
                    chart_data.append({
                        "label": label,
                        "value": 0,
                        "metric": "No _helion_kernel found"
                    })
            except Exception as e:
                print(f"Error parsing profile section: {e}")
                
    return chart_data

@app.route('/')
def index():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Helion Profiler & Eval Logs</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 20px; background: #f6f8fa; color: #24292f; }
        .header { margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; }
        .container { display: flex; gap: 20px; }
        .sidebar { width: 350px; background: white; padding: 15px; border-radius: 8px; border: 1px solid #d0d7de; height: 85vh; overflow-y: auto; }
        .content { flex-grow: 1; background: white; padding: 20px; border-radius: 8px; border: 1px solid #d0d7de; height: 85vh; overflow-y: auto; display: flex; flex-direction: column; gap: 20px; }
        ul { list-style-type: none; padding: 0; margin: 0; }
        li { margin-bottom: 5px; }
        a { text-decoration: none; color: #0969da; display: block; padding: 8px; border-radius: 6px; cursor: pointer; }
        a:hover { background: #f3f4f6; }
        a.active { background: #0969da; color: white; }
        pre { background: #1f2328; color: #f6f8fa; padding: 15px; border-radius: 6px; overflow-x: auto; font-family: ui-monospace, SFMono-Regular, monospace; font-size: 13px; line-height: 1.45; margin: 0; flex-grow: 1; }
        h1, h2 { margin-top: 0; }
        .chart-container { background: white; padding: 15px; border-radius: 6px; border: 1px solid #d0d7de; height: 300px; display: none; }
        table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
        th, td { padding: 10px; border: 1px solid #d0d7de; text-align: left; }
        th { background-color: #f6f8fa; }
        .table-container { display: none; }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>Helion Hackathon Viewer</h1>
            <p>Live polling enabled. Automatically parses Benchmarks and Profiles into interactive charts.</p>
        </div>
        <span id="status" style="color: #0969da; font-weight: bold;">Polling active...</span>
    </div>
    <div class="container">
        <div class="sidebar">
            <h2>Experiments</h2>
            <ul id="log-list"></ul>
        </div>
        <div class="content">
            <h2 id="log-title">Select an experiment to view</h2>
            
            <div class="table-container" id="table-wrapper">
                <h3 id="table-title">Extracted Data</h3>
                <table id="dataTable">
                    <thead>
                        <tr>
                            <th>Shape Configuration</th>
                            <th>Value</th>
                            <th>Metric Info</th>
                        </tr>
                    </thead>
                    <tbody></tbody>
                </table>
            </div>

            <div class="chart-container" id="chart-wrapper">
                <canvas id="experimentChart"></canvas>
            </div>
            
            <pre id="log-content">Run experiments locally using ./run_experiment.sh to generate new logs.</pre>
        </div>
    </div>
    <script>
        let currentLog = null;
        let chartInstance = null;

        async function fetchLogs() {
            try {
                const res = await fetch('/api/logs');
                const data = await res.json();
                renderLogList(data.logs);
            } catch (e) {
                console.error('Error fetching logs', e);
            }
        }

        function renderLogList(logs) {
            const ul = document.getElementById('log-list');
            ul.innerHTML = '';
            logs.forEach(log => {
                const li = document.createElement('li');
                const a = document.createElement('a');
                a.textContent = log;
                a.onclick = () => selectLog(log);
                if (log === currentLog) {
                    a.classList.add('active');
                }
                li.appendChild(a);
                ul.appendChild(li);
            });
        }

        async function selectLog(filename) {
            currentLog = filename;
            fetchLogs(); // Re-render list to highlight active
            document.getElementById('log-title').textContent = filename;
            document.getElementById('log-content').textContent = 'Loading...';
            
            try {
                const res = await fetch(`/api/log/${filename}`);
                const data = await res.json();
                
                document.getElementById('log-content').textContent = data.content;
                
                const chartWrapper = document.getElementById('chart-wrapper');
                const tableWrapper = document.getElementById('table-wrapper');
                
                if (data.parsed_data && data.parsed_data.length > 0) {
                    chartWrapper.style.display = 'block';
                    tableWrapper.style.display = 'block';
                    
                    document.getElementById('table-title').textContent = 
                        filename.includes('profile') ? 'Profile Results (_helion_kernel Execution Time)' : 'Benchmark Results (Mean Execution Time)';
                        
                    renderTable(data.parsed_data);
                    renderChart(data.parsed_data, filename.includes('profile'));
                } else {
                    chartWrapper.style.display = 'none';
                    tableWrapper.style.display = 'none';
                    if (chartInstance) {
                        chartInstance.destroy();
                        chartInstance = null;
                    }
                }
                
            } catch (e) {
                console.error('Error fetching log content', e);
                document.getElementById('log-content').textContent = 'Error loading file.';
            }
        }
        
        function renderTable(parsed_data) {
            const tbody = document.querySelector('#dataTable tbody');
            tbody.innerHTML = '';
            parsed_data.forEach(d => {
                const tr = document.createElement('tr');
                
                const tdLabel = document.createElement('td');
                tdLabel.textContent = d.label;
                
                const tdVal = document.createElement('td');
                tdVal.textContent = d.value.toFixed(4) + ' ms';
                
                const tdMetric = document.createElement('td');
                tdMetric.textContent = d.metric;
                
                tr.appendChild(tdLabel);
                tr.appendChild(tdVal);
                tr.appendChild(tdMetric);
                tbody.appendChild(tr);
            });
        }

        function renderChart(parsed_data, isProfile) {
            const ctx = document.getElementById('experimentChart').getContext('2d');
            if (chartInstance) {
                chartInstance.destroy();
            }
            
            const chartLabel = isProfile ? '_helion_kernel Time (ms)' : 'Mean Execution Time (ms)';
            const barColor = isProfile ? 'rgba(218, 9, 105, 0.6)' : 'rgba(9, 105, 218, 0.6)';
            const borderColor = isProfile ? 'rgba(218, 9, 105, 1)' : 'rgba(9, 105, 218, 1)';
            
            chartInstance = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: parsed_data.map(d => d.label),
                    datasets: [{
                        label: chartLabel,
                        data: parsed_data.map(d => d.value),
                        backgroundColor: barColor,
                        borderColor: borderColor,
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: { display: true, text: 'Milliseconds (Lower is Better)' }
                        },
                        x: {
                            title: { display: true, text: 'Tensor Shape' }
                        }
                    }
                }
            });
        }

        // Poll every 2 seconds
        setInterval(fetchLogs, 2000);
        // Initial fetch
        fetchLogs();
    </script>
</body>
</html>
"""

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
    
    return jsonify({
        "content": content,
        "parsed_data": parsed_data
    })

if __name__ == '__main__':
    print("Starting Helion Log Viewer with Chart.js support...")
    print("Open http://127.0.0.1:8080 in your browser.")
    app.run(debug=True, port=8080)
