import os
import re
import traceback
import httpx
import edge_tts
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from ddgs import DDGS
from google import genai
from google.genai import types

app = FastAPI()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
YEMOT_TOKEN = os.environ["YEMOT_TOKEN"]
YEMOT_API = "https://www.call2all.co.il/ym/api"

sessions: dict[str, list[types.Content]] = {}

def search_web(query: str) -> str:
    """חיפוש מידע עדכני באינטרנט."""
    results = DDGS().text(query, max_results=3)
    if not results:
        return "לא נמצאו תוצאות."
    return "\n".join(f"- {r['title']}: {r['body']}" for r in results)

@app.get("/")
async def health():
    return {"status": "ok"}

@app.api_route("/yemot", methods=["GET", "POST"])
async def yemot_handler(request: Request):
    try:
        params = dict(request.query_params)
        print("YEMOT PARAMS:", params, flush=True)
        call_id = re.sub(r"[^a-zA-Z0-9]", "", params.get("ApiCallId", "default"))[:16]

        if params.get("hangup") == "yes":
            sessions.pop(call_id, None)
            return PlainTextResponse("noop=ok")

        turns = []
        for k, v in params.items():
            m = re.fullmatch(r"turn_(\d+)", k)
            if m and v:
                turns.append((int(m.group(1)), v))

        if not turns:
            sessions[call_id] = []
            return PlainTextResponse(
                f"read=t-שלום אני העוזר הקולי שלך אפשר לדבר עכשיו ולסיים בסולמית="
                f"turn_0,,record,/5/rec,in_{call_id}_0,no,no,no,1,20"
            )

        last_idx, rec_path = max(turns, key=lambda x: x[0])
        next_idx = last_idx + 1

        # הוספת קידומת ivr2: אם חסרה
        dl_path = rec_path if rec_path.startswith("ivr2:") else f"ivr2:{rec_path}"
        if not dl_path.endswith(".wav"):
            dl_path += ".wav"

        async with httpx.AsyncClient(timeout=30) as http:
            dl = await http.get(
                f"{YEMOT_API}/DownloadFile",
                params={"token": YEMOT_TOKEN, "path": dl_path},
            )
            audio_bytes = dl.content

        history = sessions.setdefault(call_id, [])
        history.append(
            types.Content(
                role="user",
                parts=[types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")],
            )
        )

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=history,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "אתה עוזר קולי ישראלי בטלפון. ענה בעברית טבעית, ברורה וקצרה "
                    "(עד 2-3 משפטים בכל תור, בלי סימני עיצוב או כוכביות). "
                    "השתמש בכלי החיפוש search_web כשנדרש מידע עדכני מהרשת."
                ),
                tools=[search_web],
            ),
        )
        reply_text = (response.text or "סליחה לא הצלחתי להבין").strip()
        history.append(types.Content(role="model", parts=[types.Part.from_text(text=reply_text)]))

        tts = edge_tts.Communicate(reply_text, "he-IL-HilaNeural")
        mp3_bytes = b"".join([c["data"] async for c in tts.stream() if c["type"] == "audio"])

        out_name = f"out_{call_id}_{last_idx}"
        async with httpx.AsyncClient(timeout=30) as http:
            up = await http.post(
                f"{YEMOT_API}/UploadFile",
                data={"token": YEMOT_TOKEN, "path": f"ivr2:/5/tts/{out_name}.mp3", "convertAudio": "1"},
                files={"file": (f"{out_name}.mp3", mp3_bytes, "audio/mpeg")},
            )
            print("UPLOAD RESULT:", up.text, flush=True)

        return PlainTextResponse(
            f"read=f-/5/tts/{out_name}=turn_{next_idx},,record,/5/rec,in_{call_id}_{next_idx},no,no,no,1,20"
        )
    except Exception:
        print("ERROR:", traceback.format_exc(), flush=True)
        return PlainTextResponse("id_list_message=t-ארעה שגיאה בשרת אנא נסה שוב")
