import os
import re
import traceback
from contextlib import asynccontextmanager
import httpx
import edge_tts
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from ddgs import DDGS
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
YEMOT_TOKEN = os.environ["YEMOT_TOKEN"]
YEMOT_API = "https://www.call2all.co.il/ym/api"
http_client: httpx.AsyncClient | None = None

# שומר היסטוריה טקסטואלית בלבד (לא קבצי WAV כבדים) עד 6 תורות אחרונים
sessions: dict[str, list[types.Content]] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=25)
    yield
    await http_client.aclose()

app = FastAPI(lifespan=lifespan)

def search_web(query: str) -> str:
    """חיפוש מידע עדכני באינטרנט."""
    try:
        results = DDGS().text(query, max_results=3)
        if not results:
            return "לא נמצאו תוצאות."
        return "\n".join(f"- {r['title']}: {r['body']}" for r in results)
    except Exception:
        return "החיפוש נכשל זמנית."

@app.get("/")
async def health():
    return {"status": "ok"}

@app.api_route("/yemot", methods=["GET", "POST"])
async def yemot_handler(request: Request):
    params = dict(request.query_params)
    call_id = re.sub(r"[^a-zA-Z0-9]", "", params.get("ApiCallId", "default"))[:16]
    next_idx = 0

    try:
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

        dl_path = rec_path if rec_path.startswith("ivr2:") else f"ivr2:{rec_path}"
        if not dl_path.endswith(".wav"):
            dl_path += ".wav"

        dl = await http_client.get(
            f"{YEMOT_API}/DownloadFile",
            params={"token": YEMOT_TOKEN, "path": dl_path},
        )
        audio_bytes = dl.content

        # בדיקה שההקלטה תקינה ולא ריקה
        if len(audio_bytes) < 500 or not audio_bytes.startswith(b"RIFF"):
            return PlainTextResponse(
                f"read=t-לא שמעתי הקלטה ברורה אפשר לנסות שוב ולסיים בסולמית="
                f"turn_{next_idx},,record,/5/rec,in_{call_id}_{next_idx},no,no,no,1,20"
            )

        history = sessions.setdefault(call_id, [])
        contents = history[-12:] + [
            types.Content(
                role="user",
                parts=[types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")],
            )
        ]

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "אתה עוזר קולי ישראלי בשיחת טלפון. ענה בעברית טבעית, ברורה וקצרה "
                    "(עד 2-3 משפטים, בלי כוכביות, בלי סימני עיצוב ובלי קישורים). "
                    "השתמש בכלי search_web רק כשצריך מידע עדכני מהרשת."
                ),
                tools=[search_web],
            ),
        )
        reply_text = (response.text or "סליחה לא הצלחתי להבין").strip()
        reply_text = re.sub(r"[*_#`]", "", reply_text)

        # שומרים בהיסטוריה רק טקסט קצר כדי לא לשלוח אודיו ישן בתורים הבאים
        history.append(types.Content(role="user", parts=[types.Part.from_text(text="[שאלה קולית של המשתמש]")]))
        history.append(types.Content(role="model", parts=[types.Part.from_text(text=reply_text)]))
        if len(history) > 12:
            sessions[call_id] = history[-12:]

        tts = edge_tts.Communicate(reply_text, "he-IL-HilaNeural", rate="+10%")
        mp3_bytes = b"".join([c["data"] async for c in tts.stream() if c["type"] == "audio"])

        out_name = f"out_{call_id}_{last_idx}"
        await http_client.post(
            f"{YEMOT_API}/UploadFile",
            data={"token": YEMOT_TOKEN, "path": f"ivr2:/5/tts/{out_name}.mp3", "convertAudio": "1"},
            files={"file": (f"{out_name}.mp3", mp3_bytes, "audio/mpeg")},
        )

        return PlainTextResponse(
            f"read=f-/5/tts/{out_name}=turn_{next_idx},,record,/5/rec,in_{call_id}_{next_idx},no,no,no,1,20"
        )
    except Exception:
        print("ERROR:", traceback.format_exc(), flush=True)
        return PlainTextResponse(
            f"read=t-הייתה תקלה קצרה אפשר לחזור על השאלה ולסיים בסולמית="
            f"turn_{next_idx},,record,/5/rec,in_{call_id}_{next_idx},no,no,no,1,20"
        )
