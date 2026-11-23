import os
import re
import time
import traceback
from contextlib import asynccontextmanager
from datetime import datetime
from zoneinfo import ZoneInfo
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

sessions: dict[str, list[types.Content]] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=20)
    yield
    await http_client.aclose()

app = FastAPI(lifespan=lifespan)

def search_web(query: str) -> str:
    """חיפוש מידע עדכני ברשת (מזג אוויר, מחירים, שערים, זמנים, עובדות משתנות)."""
    try:
        results = DDGS(timeout=4).text(query, region="il-he", max_results=4)
        if not results:
            results = DDGS(timeout=4).text(query, max_results=3)
        if not results:
            return "לא נמצאו תוצאות."
        return "\n".join(f"- {r.get('title', '')}: {r.get('body', '')}" for r in results)
    except Exception:
        return "החיפוש נכשל זמנית."

def search_news(query: str) -> str:
    """חיפוש חדשות ואירועים אחרונים מישראל ומהעולם מהימים האחרונים."""
    try:
        results = DDGS(timeout=4).news(query, region="il-he", timelimit="w", max_results=4)
        if not results:
            results = DDGS(timeout=4).news(query, timelimit="w", max_results=3)
        if not results:
            return "לא נמצאו חדשות."
        return "\n".join(
            f"- [{r.get('date', '')}] {r.get('title', '')}: {r.get('body', '')}"
            for r in results
        )
    except Exception:
        return "חיפוש החדשות נכשל זמנית."

@app.get("/")
async def health():
    return {"status": "ok"}

@app.api_route("/yemot", methods=["GET", "POST"])
async def yemot_handler(request: Request):
    t0 = time.monotonic()
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
        t_dl = time.monotonic() - t0

        if len(audio_bytes) < 500 or not audio_bytes.startswith(b"RIFF"):
            return PlainTextResponse(
                f"read=t-לא שמעתי הקלטה ברורה אפשר לנסות שוב ולסיים בסולמית="
                f"turn_{next_idx},,record,/5/rec,in_{call_id}_{next_idx},no,no,no,1,20"
            )

        history = sessions.setdefault(call_id, [])
        contents = history[-10:] + [
            types.Content(
                role="user",
                parts=[types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")],
            )
        ]

        now_il = datetime.now(ZoneInfo("Asia/Jerusalem")).strftime("%d/%m/%Y %H:%M")
        t1 = time.monotonic()
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=(
                    f"התאריך והשעה עכשיו בישראל: {now_il}. "
                    "אתה עוזר קולי ישראלי ידידותי, חכם וטבעי לשיחה כללית בטלפון. "
                    "כללים: "
                    "1. בשיחה רגילה, שאלות ידע כללי, ייעוץ, הסברים או רעיונות — ענה ישירות ומהר בלי להפעיל חיפוש. "
                    "2. רק כשהמשתמש שואל על משהו שדורש בדיקה עדכנית (חדשות, מזג אוויר, מחירים, שערי מטבע, זמני היום, ספורט, בעלי תפקידים או אירועים אחרונים) — הפעל מיד את search_news או search_web עם שאילתה ממוקדת בעברית וענה לפי התוצאות. לעולם אל תאמר שאין לך גישה לאינטרנט. "
                    "3. דבר בעברית טבעית וקולחת, בתשובות קצרות וברורות (בדרך כלל 1 עד 3 משפטים, אלא אם התבקשת להרחיב), בלי כוכביות, בלי סימני עיצוב ובלי קישורים."
                ),
                tools=[search_news, search_web],
                max_output_tokens=260,
                thinking_config=types.ThinkingConfig(thinking_level="low"),
            ),
        )
        t_llm = time.monotonic() - t1

        reply_text = (response.text or "סליחה לא הצלחתי להבין").strip()
        reply_text = re.sub(r"[*_#`]", "", reply_text)

        history.append(types.Content(role="user", parts=[types.Part.from_text(text="[שאלה קולית]")]))
        history.append(types.Content(role="model", parts=[types.Part.from_text(text=reply_text)]))
        if len(history) > 10:
            sessions[call_id] = history[-10:]

        t2 = time.monotonic()
        tts = edge_tts.Communicate(reply_text, "he-IL-HilaNeural", rate="+15%")
        mp3_bytes = b"".join([c["data"] async for c in tts.stream() if c["type"] == "audio"])
        t_tts = time.monotonic() - t2

        t3 = time.monotonic()
        out_name = f"out_{call_id}_{last_idx}"
        await http_client.post(
            f"{YEMOT_API}/UploadFile",
            data={"token": YEMOT_TOKEN, "path": f"ivr2:/5/tts/{out_name}.mp3", "convertAudio": "1"},
            files={"file": (f"{out_name}.mp3", mp3_bytes, "audio/mpeg")},
        )
        t_up = time.monotonic() - t3
        print(f"TIMING dl={t_dl:.2f}s llm={t_llm:.2f}s tts={t_tts:.2f}s up={t_up:.2f}s total={time.monotonic()-t0:.2f}s", flush=True)

        return PlainTextResponse(
            f"read=f-/5/tts/{out_name}=turn_{next_idx},,record,/5/rec,in_{call_id}_{next_idx},no,no,no,1,20"
        )
    except Exception:
        print("ERROR:", traceback.format_exc(), flush=True)
        return PlainTextResponse(
            f"read=t-הייתה תקלה קצרה אפשר לחזור על השאלה ולסיים בסולמית="
            f"turn_{next_idx},,record,/5/rec,in_{call_id}_{next_idx},no,no,no,1,20"
        )
