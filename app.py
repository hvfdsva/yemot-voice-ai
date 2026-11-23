import os
import re
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

# היסטוריית שיחה זמנית לפי מזהה שיחה
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
    params = dict(request.query_params)
    call_id = re.sub(r"[^a-zA-Z0-9_-]", "", params.get("ApiCallId", "default"))

    if params.get("hangup") == "yes":
        sessions.pop(call_id, None)
        return PlainTextResponse("noop=ok")

    # איתור התור האחרון שהוקלט
    turns = []
    for k, v in params.items():
        m = re.fullmatch(r"turn_(\d+)", k)
        if m and v:
            turns.append((int(m.group(1)), v))

    if not turns:
        sessions[call_id] = []
        return PlainTextResponse(
            f"read=t-שלום, אני העוזר הקולי שלך. אפשר לדבר אחרי הביפ ולסיים בסולמית="
            f"turn_0,no,record,/5/rec,in_{call_id}_0,no,no,no,1,20"
        )

    last_idx, rec_path = max(turns, key=lambda x: x[0])
    next_idx = last_idx + 1

    # 1. הורדת ההקלטה מימות המשיח
    async with httpx.AsyncClient(timeout=30) as http:
        dl = await http.get(f"{YEMOT_API}/DownloadFile", params={"token": YEMOT_TOKEN, "path": rec_path})
        audio_bytes = dl.content

    # 2. שליחה ל-Gemini (שמיעת האודיו + חיפוש אינטרנט בעת הצורך)
    history = sessions.setdefault(call_id, [])
    user_msg = types.Content(
        role="user",
        parts=[types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")],
    )
    history.append(user_msg)

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
    reply_text = (response.text or "סליחה, לא הצלחתי לעבד את הבקשה.").strip()
    history.append(types.Content(role="model", parts=[types.Part.from_text(text=reply_text)]))

    # 3. המרה לדיבור עברי טבעי (edge-tts) והעלאה לימות המשיח
    tts = edge_tts.Communicate(reply_text, "he-IL-HilaNeural")
    mp3_chunks = [c["data"] async for c in tts.stream() if c["type"] == "audio"]
    mp3_bytes = b"".join(mp3_chunks)

    out_name = f"out_{call_id}_{last_idx}"
    async with httpx.AsyncClient(timeout=30) as http:
        await http.post(
            f"{YEMOT_API}/UploadFile",
            data={"token": YEMOT_TOKEN, "path": f"ivr2:/5/tts/{out_name}.mp3", "convertAudio": "1"},
            files={"file": (f"{out_name}.mp3", mp3_bytes, "audio/mpeg")},
        )

    return PlainTextResponse(
        f"read=f-/5/tts/{out_name}=turn_{next_idx},no,record,/5/rec,in_{call_id}_{next_idx},no,no,no,1,20"
    )
