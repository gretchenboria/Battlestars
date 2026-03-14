from flask import Flask, jsonify, request
import os
from pathlib import Path
import re

app = Flask(__name__)
LOGS_DIR = Path("helion/logs")

def parse_benchmarks(content):
    # Matches lines like: Benchmark 0: 0.0302 ms (min=0.0301, max=0.0305)  {'B': 1, 'D': 1536, 'S': 2048, 'W': 4, 'seed': 2146}
    pattern = r"Benchmark \d+: ([\d\.]+) ms .*?  (\{.*\})"
    matches = re.findall(pattern, content)
    chart_data = []
    for mean_ms_str, dict_str in matches:
        try:
            mean_ms = float(mean_ms_str)
            shape_dict = eval(dict_str)
            label = " | ".join(f"{k}={v}" for k, v in shape_dict.items() if k != 'seed')
            chart_data.append({"label": label, "mean_ms": mean_ms})
        except Exception as e:
            print(f"Error parsing line: {e}")
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
            <p>Live polling enabled. Run your experiments, and logs will appear automatically.</p>
        </div>
        <span id="status" style="color: #0969da; font-weight: bold;">Polling active...</span>
    </div>
    <div class="container">
        <div class="sidebar">
            <h2>Logs</h2>
            <ul id="log-list"></ul>
        </div>
        <div class="content">
            <h2 id="log-title">Select a log file to view</h2>
            
            <div class="table-container" id="table-wrapper">
                <h3>Benchmark Results</h3>
                <table id="benchmarkTable">
                    <thead>
                        <tr>
                            <th>Shape Configuration</th>
                            <th>Mean Execution Time (ms)</th>
                        </tr>
                    </thead>
                    <tbody></tbody>
                </table>
            </div>

            <div class="chart-container" id="chart-wrapper">
                <canvas id="benchmarkChart"></canvas>
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
                
                if (data.benchmarks && data.benchmarks.length > 0) {
                    chartWrapper.style.display = 'block';
                    tableWrapper.style.display = 'block';
                    renderTable(data.benchmarks);
                    renderChart(data.benchmarks);
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
        
        function renderTable(benchmarks) {
            const tbody = document.querySelector('#benchmarkTable tbody');
            tbody.innerHTML = '';
            benchmarks.forEach(b => {
                const tr = document.createElement('tr');
                const tdLabel = document.createElement('td');
                tdLabel.textContent = b.label;
                const tdTime = document.createElement('td');
                tdTime.textContent = b.mean_ms.toFixed(4);
                tr.appendChild(tdLabel);
                tr.appendChild(tdTime);
                tbody.appendChild(tr);
            });
        }

        function renderChart(benchmarks) {
            const ctx = document.getElementById('benchmarkChart').getContext('2d');
            if (chartInstance) {
                chartInstance.destroy();
            }
            
            chartInstance = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: benchmarks.map(d => d.label),
                    datasets: [{
                        label: 'Mean Execution Time (ms)',
                        data: benchmarks.map(d => d.mean_ms),
                        backgroundColor: 'rgba(9, 105, 218, 0.6)',
                        borderColor: 'rgba(9, 105, 218, 1)',
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
    benchmarks = parse_benchmarks(content)
    
    return jsonify({
        "content": content,
        "benchmarks": benchmarks
    })

if __name__ == '__main__':
    print("Starting Helion Log Viewer...")
    print("Open http://127.0.0.1:8080 in your browser.")
    app.run(debug=True, port=8080)