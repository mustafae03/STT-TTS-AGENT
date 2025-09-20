import os
import json
from typing import Tuple, Dict, Any, Optional

import openai
import gradio as gr
from PIL import Image, ImageDraw
from dotenv import load_dotenv
load_dotenv(dotenv_path="llm_engineering/.env")

openai.api_key = os.getenv("OPENAI_API_KEY")

MODEL = "gpt-4o-mini"
WHISPER = "whisper-1"
TTS = "tts-1"
VOICE = "onyx"

SYSTEM_MESSAGE = "Sen bir seyahat asistanısın. Kısa ve net cevap ver."

tools: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_ticket_price",
            "description": "Bir şehrin yaklaşık bilet fiyatını döndür.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination_city": {
                        "type": "string",
                        "description": "Gidilecek şehir"
                    }
                },
                "required": ["destination_city"]
            }
        }
    }
]

def get_ticket_price(city: str) -> str:
    prices = {"istanbul": 1200, "ankara": 1400, "izmir": 1500, "antalya": 1350}
    if not city:
        return "Şehir belirtilmedi."
    price = prices.get(city.lower(), 1600)
    return f"{city.title()} için yaklaşık bilet fiyatı: {price} TL"

def handle_tool_call(message) -> Tuple[Dict[str, Any], Optional[str]]:
    tool_calls = getattr(message, "tool_calls", None) or message.get("tool_calls", [])
    if not tool_calls:
        return {"role": "tool", "content": json.dumps({"ok": True})}, None
    tool_call = tool_calls[0]
    if tool_call.get("type") != "function":
        return {"role": "tool", "content": json.dumps({"ok": True})}, None
    fn_name = tool_call["function"]["name"]
    raw_args = tool_call["function"].get("arguments", "{}")
    try:
        args = json.loads(raw_args)
    except json.JSONDecodeError:
        args = {}
    if fn_name == "get_ticket_price":
        city = args.get("destination_city")
        result_text = get_ticket_price(city)
        tool_msg = {
            "role": "tool",
            "tool_call_id": tool_call.get("id"),
            "content": result_text
        }
        return tool_msg, city
    return {"role": "tool", "content": json.dumps({"ok": True})}, None

def artist(city: Optional[str]) -> Optional[str]:
    if not city:
        return None
    img = Image.new("RGB", (900, 500), color=(235, 244, 255))
    d = ImageDraw.Draw(img)
    txt = city.title()
    d.text((50, 200), txt, fill=(15, 30, 60))
    out_path = "city_poster.png"
    img.save(out_path)
    return out_path

def transcribe_audio(audio_path: Optional[str]) -> str:
    if not audio_path:
        return ""
    with open(audio_path, "rb") as f:
        tr = openai.audio.transcriptions.create(model=WHISPER, file=f)
    return (tr.text or "").strip()

def run_assistant(user_text: str, history: list[dict[str, str]]) -> tuple[list[dict[str, str]], Optional[str], Optional[str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_MESSAGE}]
    messages += history
    messages.append({"role": "user", "content": user_text})
    image_path: Optional[str] = None
    response = openai.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    if response.choices[0].finish_reason == "tool_calls":
        tool_msg, city = handle_tool_call(response.choices[0].message)
        messages.append(tool_msg)
        try:
            if city:
                image_path = artist(city)
        except Exception:
            image_path = None
        response = openai.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools
        )
    assistant_reply: str = response.choices[0].message.content
    speech = openai.audio.speech.create(model=TTS, voice=VOICE, input=assistant_reply)
    audio_path = "assistant_reply.mp3"
    with open(audio_path, "wb") as f:
        f.write(speech.content)
    new_history = history + [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_reply},
    ]
    return new_history, image_path, audio_path

with gr.Blocks(title="STT + Tools + TTS Agent") as ui:
    gr.Markdown("## 🎤 STT + 🔧 Tools + 🗣 TTS + 🖼 Görsel — Demo")
    with gr.Row():
        chatbot = gr.Chatbot(height=480, type="messages", label="Sohbet")
        image_out = gr.Image(height=480, label="Üretilen Görsel (opsiyonel)")
    with gr.Row():
        mic = gr.Audio(sources=["microphone"], type="filepath", label="Mikrofon")
        tts_out = gr.Audio(label="Asistan Sesi", autoplay=True)
        clear = gr.Button("Temizle")
    def on_stop_recording(audio_path: Optional[str], history: Optional[list[dict[str, str]]]):
        text = transcribe_audio(audio_path)
        if not text:
            return history or [], None, None
        return run_assistant(text, history or [])
    mic.stop_recording(
        on_stop_recording,
        inputs=[mic, chatbot],
        outputs=[chatbot, image_out, tts_out]
    )
    clear.click(lambda: [], outputs=chatbot)

if __name__ == "__main__":
    ui.launch(inbrowser=True)
