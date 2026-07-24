from flask import Flask, request, jsonify
from groq import Groq
from flask_cors import CORS
import os
import time
import json
import threading
import requests

app = Flask(__name__)
CORS(app, origins=[
    "https://chatflow-ai-o3e6.onrender.com",
    "https://chatflow.com",
    "https://backend-1-liqz.onrender.com",
    "https://testback-4sru.onrender.com",
    "https://backend-vz58.onrender.com",
    "https://chatflow-ai-o3e6.onrender.com"
    
])

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ===== SITE KNOWLEDGE BASE (edit this to keep the assistant accurate) =====
SITE_CONTEXT = """
Site name: Kairos (Kairos.chat)
What it is: An automated messaging app. It replies to chats for you when you're
busy, suggests smart context-aware replies, and can be customized to sound like you.

Sections on the landing page:
- "home" (#home): Hero section — intro, "Start for free" and "See it in action" buttons.
- "about" (#about): "How it works" — explains Auto Reply, AI Mode, Smart Suggestions,
  demo videos.
- "signup" (/signup): Where a new user creates an account.

Key features:
- Auto Reply: one-tap toggle (⚡) that lets Kairos answer incoming messages automatically.
- AI Mode: user writes instructions (e.g. "reply politely", "say I'm in a meeting")
  and Kairos follows them for every auto-reply.
- Smart Suggestions: tap ⚡ to get AI-generated reply options without full automation.
- Your Voice, Always: Kairos adapts tone to sound like the user (formal/friendly/punchy).
- Stay in Control: every automated reply is logged; auto-reply can be turned off anytime.
- Phone App: native iOS/Android app is coming soon (not available yet).
- Pricing: not mentioned on the site — if asked, say pricing isn't published yet.
"""
# ===========================================================================

def safe_ai_call(messages, max_tokens, retries=2, use_json=False):
    for i in range(retries + 1):
        try:
            kwargs = dict(
                model="llama-3.1-8b-instant",
                messages=messages,
                temperature=0.8,
                max_completion_tokens=max_tokens,
                timeout=30
            )
            if use_json:
                kwargs["response_format"] = {"type": "json_object"}
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            if i == retries:
                raise e
            time.sleep(0.5)

@app.route("/ai", methods=["POST"])
def ai():
    data = request.get_json()
    text = data.get("text", "")
    mode = data.get("mode", "default")
    instructions = data.get("instructions", "")
    tone = data.get("tone", "")   # used by mode == "help_me_write" AND mode == "chat" (Auto AI)

    try:
        system = "Reply in 1 short natural sentence like in a phone call."

        # parse the same "|length:X|emoji:Y" packing used by help_me_write,
        # so Auto AI's Short/Long buttons actually change reply length, and the
        # model never invents extra facts just to pad out a "long" answer.
        def parse_length_emoji(raw_tone):
            parts = raw_tone.split("|") if raw_tone else []
            tone_label = parts[0].strip() if parts else ""
            length = "Medium"
            use_emoji = False
            for p in parts[1:]:
                if p.startswith("length:"):
                    length = p.split(":", 1)[1].strip()
                if p.startswith("emoji:"):
                    use_emoji = p.split(":", 1)[1].strip().lower() == "true"
            return tone_label, length, use_emoji

        # NEW — these are computed up front (not just inside the "chat" if-block)
        # so the persona/instructions block below can reuse them instead of
        # hardcoding its own separate length/emoji rules.
        chat_length = "Medium"
        chat_length_rule = "Reply in 1-2 short natural sentences like in a phone call."
        chat_emoji_rule = "Do not use any emoji."

        if mode == "chat":
            _, chat_length, chat_emoji = parse_length_emoji(tone)
            chat_emoji_rule = "You may use light, tasteful emoji." if chat_emoji else "Do not use any emoji."
            if chat_length == "Short":
                chat_length_rule = (
                    "Reply in EXACTLY 1 short sentence like in a phone call. "
                    "Never add facts, details, or reasons not present in the incoming message."
                )
            elif chat_length == "Long":
                chat_length_rule = (
                    "Reply in 3-4 sentences, adding natural conversational detail, but ONLY "
                    "elaborating on what is explicitly implied by the incoming message — never "
                    "invent names, numbers, or events."
                )
            else:
                chat_length_rule = "Reply in 1-2 short natural sentences like in a phone call."

            system = chat_length_rule + " " + chat_emoji_rule

        if mode == "summary":
            system = "Summarize in 2 short sentences."
        elif mode == "help_me_write":
            # tone now arrives packed as "Formal|length:Short|emoji:False"
            parts = tone.split("|") if tone else []
            tone_label = parts[0].strip() if parts else ""
            length = "Medium"
            use_emoji = False
            for p in parts[1:]:
                if p.startswith("length:"):
                    length = p.split(":", 1)[1].strip()
                if p.startswith("emoji:"):
                    use_emoji = p.split(":", 1)[1].strip().lower() == "true"

            tone_line = f"Tone: {tone_label}.\n" if tone_label else ""
            emoji_line = "You may use light, tasteful emoji.\n" if use_emoji else "Do not use any emoji.\n"

            if length == "Short":
                length_line = (
                    "Write EXACTLY 1 short sentence. Do not add details, reasons, or facts "
                    "that are not explicitly present in the user's prompt. If the prompt is vague, "
                    "keep the message equally vague rather than inventing specifics.\n"
                )
            elif length == "Long":
                length_line = (
                    "Write 4-6 sentences with more context and detail, but ONLY elaborate on what "
                    "is explicitly implied by the user's prompt — do not invent names, dates, numbers, "
                    "or events not present in the prompt.\n"
                )
            else:
                length_line = "Write 2-3 sentences.\n"

            system = (
                "You are helping someone write ONE chat message they are about to send. "
                "The user describes WHAT they want to say — you write it for them, as a natural "
                "message, not as a reply to them and not as an AI assistant. "
                f"{tone_line}{length_line}{emoji_line}"
                "Return ONLY the message text. No quotes, no explanation, no greeting like 'Sure, here is...'. "
                "Never state something as fact unless it was in the user's prompt."
            )
        elif mode == "ai_writer":
            system = (
                "You are a creative message rewriter for a chat app. "
                "The user gives you a message they want to SEND. "
                "Your job is to rewrite it in 4 different creative styles (casual, funny, formal, expressive). "
                "Do NOT answer the message. Do NOT reply to it. ONLY rewrite it in different ways. "
                "Example: if the user gives you 'how are you', return 4 ways to say 'how are you' — not answers to it. "
                "You MUST respond with ONLY this exact JSON format, nothing else:\n"
                "{\"results\": [\"version 1\", \"version 2\", \"version 3\", \"version 4\"]}\n"
                "Do NOT add any explanation, greeting, or text outside the JSON. "
                "Do NOT use markdown. Do NOT number the items. "
                "Each result must be a natural chat message that means the same thing as the input."
            )
        elif mode == "greeting":
            system = (
                "Transform the message into different greeting styles. "
                "Give 3-5 variations like casual, formal, friendly, slang."
            )

        # CHANGED — previously this block completely overwrote `system` with a
        # persona template that hardcoded "1 sentence" (old rule 3) and "no
        # emoji" (old rule 4), silently discarding whatever length/emoji the
        # user picked in Auto AI's Short/Long/Emoji controls. Now rule 3 and
        # rule 4 are built from chat_length_rule/chat_emoji_rule computed
        # above, so a persona ("brutal", "inspirational", etc.) combined with
        # "Long answer" or "Use Emoji" actually respects both at once.
        if instructions and instructions.strip() and mode != "ai_writer":
            length_instruction = chat_length_rule if mode == "chat" else "Keep replies short: 1 sentence, like a real text/phone reply."
            emoji_instruction = chat_emoji_rule if mode == "chat" else "Do not use any emoji unless the persona explicitly calls for it."

            system = f"""You are role-playing as the user in a chat conversation.
The user has defined a PERMANENT PERSONA/BEHAVIOR you must follow for every single reply, no exceptions:

PERSONA:
\"\"\"{instructions.strip()}\"\"\"

NON-NEGOTIABLE RULES:
1. Every reply MUST be written fully in this persona/tone/style — never drop it, soften it, or revert to a neutral/helpful-assistant tone.
2. This persona applies to EVERY incoming message, regardless of topic, length, or how the other person phrases things.
3. {length_instruction}
4. {emoji_instruction}
5. Never say you are an AI, never apologize for the tone, never explain the persona.
6. Do not invent facts, names, numbers, or events not present in the incoming message — only react to what was actually said, but IN THIS PERSONA.
7. If the incoming message is unrelated to anything the persona would normally discuss, still reply in-persona (a "{instructions.strip()}" person's reaction to that message).

Respond with ONLY the reply text — no labels, no quotes, no meta-commentary."""

        max_tokens = 400 if mode in ("ai_writer", "help_me_write") else 150

        completion = safe_ai_call(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": text}
            ],
            max_tokens=max_tokens,
            use_json=(mode == "ai_writer")
        )

        reply = completion.choices[0].message.content.strip()

        if mode == "ai_writer":
            try:
                parsed = json.loads(reply)
                if not isinstance(parsed.get("results"), list) or len(parsed["results"]) < 2:
                    raise ValueError("bad results")
            except Exception:
                clean = reply.replace('"', "'").strip()
                fallback = {
                    "results": [
                        clean,
                        clean + "! 😊",
                        "Hey, " + clean + "?",
                        clean + " — just checking in!"
                    ]
                }
                return jsonify({"reply": json.dumps(fallback)})

        return jsonify({"reply": reply or "..."})

    except Exception as e:
        print("AI ERROR:", e)
        if "rate" in str(e).lower() or "limit" in str(e).lower():
            return jsonify({"message": "⚠️ AI request limit reached."}), 429
        return jsonify({"message": "AI error"}), 500


@app.route("/site-ai", methods=["POST"])
def site_ai():
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get("text") or "").strip()[:300]
    if not text:
        return jsonify({"reply": "Ask me anything about Kairos!", "action": None, "target": None})

    system = f"""You are "Kairos Assistant", the helpful guide embedded on the Kairos
landing page. Answer ONLY using the site information below — never invent features,
pricing, or pages that aren't listed.

SITE INFORMATION:
{SITE_CONTEXT}

If the user is asking to be taken/navigated somewhere on the site (e.g. "take me to
sign up", "show me how it works", "go home", "I want to create an account"), respond
with strict JSON ONLY in this exact shape:
{{"reply": "<short friendly sentence, e.g. 'Sure, taking you to sign up!'>", "action": "navigate", "target": "home" | "about" | "signup"}}

For any other question (what does the app do, what is auto reply, etc.), respond with
strict JSON ONLY in this exact shape:
{{"reply": "<helpful answer based only on SITE INFORMATION>", "action": null, "target": null}}

Never output anything except that JSON object. No markdown, no extra text."""

    try:
        completion = safe_ai_call(
            [{"role": "system", "content": system},
             {"role": "user", "content": text}],
            max_tokens=220,
            use_json=True
        )
        raw = completion.choices[0].message.content.strip()
        parsed = json.loads(raw)
        if "reply" not in parsed:
            raise ValueError("bad shape")
        parsed.setdefault("action", None)
        parsed.setdefault("target", None)
        return jsonify(parsed)
    except Exception as e:
        print("SITE AI ERROR:", e)
        return jsonify({"reply": "Sorry, I couldn't process that — try rephrasing.", "action": None, "target": None}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

def self_ping():
    while True:
        time.sleep(13 * 60)
        try:
            url = os.getenv("SELF_URL", "https://chatflow-ai-1.onrender.com")
            requests.post(url + "/ai", json={
                "text": "hi",
                "mode": "chat",
                "instructions": ""
            }, timeout=10)
            print("✅ Self-ping sent")
        except Exception as e:
            print("⚠️ Self-ping failed:", e)

threading.Thread(target=self_ping, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
