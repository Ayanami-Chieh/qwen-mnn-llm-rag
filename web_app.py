from flask import Flask, request, jsonify
import logging

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)

# Dummy database for demonstration purposes
knowledge_base = {}

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    # Here we should integrate with the RAG system to get a response
    response = f"You said: {user_input}"  # Replace with RAG system logic
    return jsonify({'response': response})

@app.route('/add_document', methods=['POST'])
def add_document():
    doc_id = request.json.get('id')
    content = request.json.get('content')
    knowledge_base[doc_id] = content
    logging.info(f"Added document {doc_id}")
    return jsonify({'status': 'success', 'message': 'Document added.'})

@app.route('/get_document/<doc_id>', methods=['GET'])
def get_document(doc_id):
    document = knowledge_base.get(doc_id, 'Document not found.')
    return jsonify({'document': document})

@app.route('/metrics', methods=['GET'])
def metrics():
    # Here we could track various metrics, like number of chats, documents handled etc.
    return jsonify({'status': 'success', 'metrics': {'messages_handled': len(knowledge_base)}})

if __name__ == '__main__':
    app.run(debug=True)