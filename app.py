from flask import Flask, render_template_string
import os
from pathlib import Path
import re
import json

app = Flask(__name__)
LOGS_DIR = Path("helion/logs")

TEMPLATE = """
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
        a { text-decoration: none; color: #0969da; display: block; padding: 8px; border-radius: 6px; }
        a:hover { background: #f3f4f6; text-decoration: none; }
        a.active { background: #0969da; color: white; }
        pre { background: #1f2328; color: #f6f8fa; padding: 15px; border-radius: 6px; overflow-x: auto; font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace; font-size: 13px; line-height: 1.45; margin: 0; flex-grow: 1; }
        h1, h2 { margin-top: 0; }
        .chart-container { background: white; padding: 15px; border-radius: 6px; border: 1px solid #d0d7de; height: 300px; display: {% if chart_data %}block{% else %}none{% endif %}; }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>Helion Hackathon Viewer</h1>
            <p>Because KernelBot is busy, we build our own tooling.</p>
        </div>
        <button onclick="window.location.reload();" style="padding: 10px 15px; background: #0969da; color: white; border: none; border-radius: 6px; cursor: pointer;">Refresh Logs</button>
    </div>
    <div class="container">
        <div class="sidebar">
            <h2>Logs</h2>
            <ul>
                {% for log in logs %}
                <li><a href="/log/{{ log }}" class="{% if log == selected_log %}active{% endif %}">{{ log }}</a></li>
                {% endfor %}
            </ul>
        </div>
        <div class="content">
            {% if selected_log %}
                <h2>{{ selected_log }}</h2>
                
                <div class="chart-container">
                    <canvas id="benchmarkChart"></canvas>
                </div>
                
                <pre>{{ log_content }}</pre>
                
                {% if chart_data %}
                <script>
                    const ctx = document.getElementById('benchmarkChart').getContext('2d');
                    const rawData = {{ chart_data | safe }};
                    
                    new Chart(ctx, {
                        type: 'bar',
                        data: {
                            labels: rawData.map(d => d.label),
                            datasets: [{
                                label: 'Mean Execution Time (ms)',
                                data: rawData.map(d => d.mean_ms),
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
                </script>
                {% endif %}
                
            {% else %}
                <h2>Select a log file to view</h2>
                <p>Run experiments locally using <code>./run_experiment.sh profile causal_conv1d_py</code> to generate new logs.</p>
            {% endif %}
        </div>
    </div>
</body>
</html>
"""

def parse_benchmarks(content):
    # Matches lines like: Benchmark 0: 0.0302 ms (min=0.0301, max=0.0305)  {'B': 1, 'D': 1536, 'S': 2048, 'W': 4, 'seed': 2146}
    pattern = r"Benchmark \d+: ([\d\.]+) ms .*?  (\{.*\})"
    matches = re.findall(pattern, content)
    
    chart_data = []
    for mean_ms_str, dict_str in matches:
        try:
            mean_ms = float(mean_ms_str)
            shape_dict = eval(dict_str) # dict_str is like {'B': 1, 'D': 1536, ...}
            # Create a label excluding the random seed
            label = " | ".join(f"{k}={v}" for k, v in shape_dict.items() if k != 'seed')
            chart_data.append({"label": label, "mean_ms": mean_ms})
        except Exception as e:
            print(f"Error parsing line: {e}")
            
    return chart_data

@app.route('/')
def index():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logs = sorted([f.name for f in LOGS_DIR.glob('*.log')], reverse=True)
    return render_template_string(TEMPLATE, logs=logs, selected_log=None, chart_data="null")

@app.route('/log/<filename>')
def view_log(filename):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logs = sorted([f.name for f in LOGS_DIR.glob('*.log')], reverse=True)
    file_path = LOGS_DIR / filename
    
    content = "File not found."
    chart_data_json = "null"
    
    if file_path.exists():
        content = file_path.read_text()
        if "benchmark" in filename.lower() or "Benchmark" in content:
            parsed_data = parse_benchmarks(content)
            if parsed_data:
                chart_data_json = json.dumps(parsed_data)
                
    return render_template_string(TEMPLATE, logs=logs, selected_log=filename, log_content=content, chart_data=chart_data_json)

if __name__ == '__main__':
    print("Starting Helion Log Viewer with Chart.js support...")
    print("Open http://127.0.0.1:8080 in your browser.")
    app.run(debug=True, port=8080)
