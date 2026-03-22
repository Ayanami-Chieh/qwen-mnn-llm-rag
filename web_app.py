from flask import Flask, render_template, request, jsonify
import time

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    # Mock response, replace with actual model processing logic
    response = f'You said: {user_message}'
    return jsonify({'response': response})

@app.route('/api/status')
def status():
    # Mock data for model status, replace with actual metrics collection
    model_startup_time = time.time() - 60  # Assuming model started 1 minute ago
    inference_latency = 0.1  # Mock inference latency
    return jsonify({'model_startup_time': model_startup_time, 'inference_latency': inference_latency})

if __name__ == '__main__':
    app.run(debug=True)