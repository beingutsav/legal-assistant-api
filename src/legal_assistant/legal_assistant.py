from flask import Flask, request, jsonify
from .legal_assistant_v3 import handle_query, create_chat_session

app = Flask(__name__)

@app.route('/api/legal-assistant', methods=['POST'])
def legal_assistant_query():
    data = request.json
    chat_id = data.get('chat_id')
    query = data.get('query')

    if not query:
        return jsonify({"error": "Query is required"}), 400

    if chat_id is None:
        chat_id = create_chat_session()

    response = handle_query(chat_id, query)

    return jsonify({
        "chat_id": chat_id,
        "response": response
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)