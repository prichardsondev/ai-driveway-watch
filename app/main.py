from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.pipeline import Pipeline


settings = get_settings()
pipeline = Pipeline(settings)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    pipeline.start()
    yield
    pipeline.stop()


app = FastAPI(title="Driveway Watch", lifespan=lifespan)
app.mount("/event-images", StaticFiles(directory=settings.output_dir / "events"), name="event-images")


@app.get("/", response_class=HTMLResponse)
async def index():
    return """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>
<title>Driveway Watch</title><style>
:root{font-family:Inter,system-ui,sans-serif;color:#ecf2ed;background:#101713}body{margin:0}header{padding:18px 24px;border-bottom:1px solid #304038;display:flex;justify-content:space-between;align-items:center}h1{font-size:20px;margin:0}.pill{padding:6px 11px;border-radius:20px;background:#593d20;color:#ffd79a}.live{background:#194d31;color:#a4f0c2}main{padding:20px;display:grid;grid-template-columns:minmax(0,1fr) 360px;gap:18px}.card{background:#18221d;border:1px solid #304038;border-radius:12px;overflow:hidden}.feed{width:100%;display:block;aspect-ratio:16/9;object-fit:contain;background:#050806}.side{display:grid;gap:14px;align-content:start}section{padding:16px}h2{font-size:14px;margin:0 0 12px;color:#a9b9b0}dl{display:grid;grid-template-columns:1fr 1fr;gap:9px;margin:0;font-size:13px}dt{color:#91a198}dd{margin:0;text-align:right}.event{display:grid;grid-template-columns:70px 1fr;gap:10px;padding:10px 0;border-top:1px solid #304038}.event img{width:70px;height:48px;object-fit:cover;border-radius:5px}.empty{color:#91a198;font-size:13px}@media(max-width:850px){main{grid-template-columns:1fr}}
</style></head><body><header><h1>Driveway Watch</h1><span id='state' class='pill'>Starting</span></header><main><div class='card'><img class='feed' src='/video.mjpg' alt='Live driveway camera'></div><div class='side'><section class='card'><h2>OpenCV 5 test</h2><dl id='metrics'></dl></section><section class='card'><h2>Recent driveway events</h2><div id='events' class='empty'>No events yet</div></section></div></main><script>
const state=document.querySelector('#state'),metrics=document.querySelector('#metrics'),events=document.querySelector('#events');
async function refresh(){try{const d=await fetch('/api/status',{cache:'no-store'}).then(r=>r.json());state.textContent=d.connected?'Live':'Reconnecting';state.className='pill '+(d.connected?'live':'');const rows=[['Source',d.source],['OpenCV',d.opencv_version],['Camera FPS',d.capture_fps],['Dashboard FPS',d.publish_fps],['CPU threads',d.opencv_threads],['Inference',d.inference_ms_last+' ms'],['Average',d.inference_ms_avg+' ms'],['Every',d.inference_interval+' frames'],['Frames',d.frames],['Status',d.error||'Healthy']];metrics.innerHTML=rows.map(x=>`<dt>${x[0]}</dt><dd>${x[1]}</dd>`).join('');events.innerHTML=d.events.length?d.events.map(e=>`<div class='event'>${e.snapshot_path?`<a href='/event-images/${encodeURIComponent(e.snapshot_path)}'><img src='/event-images/${encodeURIComponent(e.snapshot_path)}'></a>`:'<span></span>'}<div><strong>${e.class_name}</strong><br><small>${Math.round(e.confidence*100)}% · ${new Date(e.timestamp).toLocaleString()}</small></div></div>`).join(''):'No events yet';}catch(e){state.textContent='Offline'}}refresh();setInterval(refresh,1000);
</script></body></html>"""


@app.get("/video.mjpg")
async def video():
    async def frames():
        while True:
            jpeg = pipeline.get_jpeg()
            if jpeg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            await asyncio.sleep(1 / settings.stream_fps)
    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/status")
async def status():
    return pipeline.status()


@app.get("/health")
async def health():
    return {"ok": True, "connected": pipeline.status()["connected"]}
