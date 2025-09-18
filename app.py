# app.py  (Flask backend for Render)
import os, time
from flask import Flask
from flask_cors import CORS
from chat_api import bp as chat_bp  # your existing blueprint

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # open CORS for testing
app.register_blueprint(chat_bp)

@app.get("/ping")
def ping():
    return {"ok": True, "ts": int(time.time())}

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=False)
