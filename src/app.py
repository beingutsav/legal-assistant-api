from flask import Flask
from legal_assistant.api import api

app = Flask(__name__)
app.register_blueprint(api)

@app.route('/')
def home():
    return "Welcome to the Legal Assistant API!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)