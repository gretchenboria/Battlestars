from flask import Flask, render_template_string
import os
from pathlib import Path

app = Flask(__name__)
LOGS_DIR = Path("helion/logs")

TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Helion Profiler & Eval Logs</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 20px; background: #f6f8fa; color: #24292f; }
        .header { margin-bottom: 20px; }
        .container { display: flex; gap: 20px; }
        .sidebar { width: 350px; background: white; padding: 15px; border-radius: 8px; border: 1px solid #d0d7de; height: 85vh; overflow-y: auto; }
        .content { flex-grow: 1; background: white; padding: 20px; border-radius: 8px; border: 1px solid #d0d7de; height: 85vh; overflow-y: auto; }
        ul { list-style-type: none; padding: 0; margin: 0; }
        li { margin-bottom: 5px; }
        a { text-decoration: none; color: #0969da; display: block; padding: 8px; border-radius: 6px; }
        a:hover { background: #f3f4f6; text-decoration: none; }
        a.active { background: #0969da; color: white; }
        pre { background: #1f2328; color: #f6f8fa; padding: 15px; border-radius: 6px; overflow-x: auto; font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace; font-size: 13px; line-height: 1.45; }
        h1, h2 { margin-top: 0; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Helion Hackathon Viewer</h1>
        <p>Because KernelBot is busy, we build our own tooling.</p>
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
                <pre>{{ log_content }}</pre>
            {% else %}
                <h2>Select a log file to view</h2>
                <p>Run experiments locally using <code>./run_experiment.sh profile causal_conv1d_py</code> to generate new logs.</p>
            {% endif %}
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logs = sorted([f.name for f in LOGS_DIR.glob('*.log')], reverse=True)
    return render_template_string(TEMPLATE, logs=logs, selected_log=None)

@app.route('/log/<filename>')
def view_log(filename):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logs = sorted([f.name for f in LOGS_DIR.glob('*.log')], reverse=True)
    file_path = LOGS_DIR / filename
    if file_path.exists():
        content = file_path.read_text()
    else:
        content = "File not found."
    return render_template_string(TEMPLATE, logs=logs, selected_log=filename, log_content=content)

if __name__ == '__main__':
    print("Starting Helion Log Viewer...")
    print("Open http://127.0.0.1:8080 in your browser.")
    app.run(debug=True, port=8080)
