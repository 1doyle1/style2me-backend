import os
from flask import Flask, request, jsonify
from flask_cors import CORS

# your existing CLIP/search blueprint (/chat, /chat/messages, /ping inside it)
from chat_api import bp as chat_bp

# try to import the agent; if it fails, we keep the app up and show a clear hint
try:
    from agent_orchestrator import run as agent_run
    HAS_AGENT = True
    AGENT_ERR = None
except Exception as e:
    HAS_AGENT = False
    AGENT_ERR = str(e)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# register blueprint routes
app.register_blueprint(chat_bp)

# root/health
@app.get("/")
def root():
    return {"ok": True, "service": "style2me-backend", "agent": HAS_AGENT, "agent_error": AGENT_ERR}

# universal chat endpoint the app uses
@app.route("/agent/chat", methods=["POST", "OPTIONS"])
def agent_chat():
    if request.method == "OPTIONS":
        return ("", 204)

    if not HAS_AGENT:
        return jsonify({"ok": False, "error": "agent_unavailable", "hint": AGENT_ERR}), 503

    data = request.get_json(force=True) or {}
    messages = data.get("messages")
    if not messages:
        msg = (data.get("message") or "").strip()
        if not msg:
            return jsonify({"ok": False, "error": "no_input"}), 400
        messages = [{"role": "user", "content": msg}]

    try:
        reply, items = agent_run(messages)
        return jsonify({"ok": True, "reply": reply, "items": items})
    except Exception as e:
        return jsonify({"ok": False, "error": "agent_error", "detail": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
