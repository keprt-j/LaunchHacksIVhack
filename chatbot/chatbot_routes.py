from flask import Blueprint, render_template, request, jsonify, session
from openai import OpenAI
import os

chatbot_bp = Blueprint('chatbot', __name__)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "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"))

@chatbot_bp.route("/chatbot", methods=["GET"])
def chatbot_page():
    """Display a very small chatbot interface."""
    return render_template("chatbot.html")

@chatbot_bp.route("/api/chat", methods=["POST"])
def api_chat():
    """Return a full response using the OpenAI API (no streaming)."""
    data = request.get_json()
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"response": "Please say something."})

    history = session.get("chat_history", [
        {"role": "system", "content": "You are a helpful assistant regarding nature of interrogation. Respond to user comments about interrogation using the context of an interrogation scene and emotions that are commonly seen."}
    ])
    history.append({"role": "user", "content": message})

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=history
        )
        response_text = completion.choices[0].message.content
    except Exception as e:
        response_text = f"Error: {e}"

    history.append({"role": "assistant", "content": response_text})
    session["chat_history"] = history[-6:]

    return jsonify({"response": response_text})
