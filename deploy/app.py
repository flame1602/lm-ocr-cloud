"""
GLM-OCR Web App — Standalone Flask server with password protection.
Designed to run alongside vLLM server on Google Cloud Compute Engine.
"""
import os
import sys
import time
import glob
import json
import base64
import shutil
import threading
import traceback
import secrets
from pathlib import Path
from functools import wraps

from flask import (
    Flask, request, jsonify, send_file, send_from_directory,
    session, redirect, url_for, make_response
)
from werkzeug.middleware.proxy_fix import ProxyFix
import fitz
from openai import OpenAI

# === Configuration ===
APP_PASSWORD = os.environ.get("APP_PASSWORD", "glm-ocr-2024")
SECRET_KEY = os.environ.get("SECRET_KEY", secrets.token_hex(32))
VLLM_PORT = int(os.environ.get("VLLM_PORT", "8899"))
PORT = int(os.environ.get("PORT", "8080"))
INPUT_DIR = os.environ.get("INPUT_DIR", "/data/input_pdfs")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/data/output_markdown")
DPI = 200

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === OCR State ===
ocr_state = {
    "running": False, "current_file": "", "current_page": 0, "total_pages": 0,
    "files_done": 0, "files_total": 0, "results": [], "error": None, "start_time": 0
}
state_lock = threading.Lock()


# === Auth ===
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("authenticated"):
            if request.is_json or request.path.startswith("/api/"):
                return jsonify({"error": "Unauthorized"}), 401
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated


# === PDF & OCR ===
def pdf_to_images(pdf_path):
    doc = fitz.open(pdf_path)
    imgs = []
    td = f"/tmp/pdf_pages/{Path(pdf_path).stem}"
    os.makedirs(td, exist_ok=True)
    for i in range(len(doc)):
        z = DPI / 72
        pix = doc[i].get_pixmap(matrix=fitz.Matrix(z, z))
        p = os.path.join(td, f"p_{i+1:04d}.png")
        pix.save(p)
        imgs.append(p)
    doc.close()
    return imgs


def ocr_api(imgs, on_page=None):
    cl = OpenAI(api_key="EMPTY", base_url=f"http://localhost:{VLLM_PORT}/v1")
    mds = []
    page_times = []
    for i, ip in enumerate(imgs):
        t0 = time.time()
        if on_page:
            on_page(i + 1, len(imgs))
        with open(ip, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        r = cl.chat.completions.create(
            model="glm-ocr",
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": "Text Recognition:"}
            ]}],
            max_tokens=8192, temperature=0.01
        )
        mds.append(r.choices[0].message.content)
        page_times.append(round(time.time() - t0, 2))
    return "\n\n---\n\n".join(mds), page_times


def ocr_worker(pdfs):
    global ocr_state
    with state_lock:
        ocr_state.update({
            "running": True, "files_done": 0, "files_total": len(pdfs),
            "results": [], "error": None, "start_time": time.time()
        })
    for idx, pdf in enumerate(pdfs):
        nm = Path(pdf).stem
        with state_lock:
            ocr_state["current_file"] = Path(pdf).name
            ocr_state["current_page"] = 0
            ocr_state["total_pages"] = 0
        try:
            t0 = time.time()
            imgs = pdf_to_images(pdf)
            t_img = time.time() - t0
            with state_lock:
                ocr_state["total_pages"] = len(imgs)

            def on_pg(p, t):
                with state_lock:
                    ocr_state["current_page"] = p

            t1 = time.time()
            md, page_times = ocr_api(imgs, on_page=on_pg)
            t_ocr = time.time() - t1
            out = os.path.join(OUTPUT_DIR, f"{nm}.md")
            with open(out, "w", encoding="utf-8") as f:
                f.write(f"<!-- OCR by GLM-OCR | {Path(pdf).name} -->\n\n{md}")
            with state_lock:
                ocr_state["files_done"] = idx + 1
                ocr_state["results"].append({
                    "file": Path(pdf).name, "pages": len(imgs),
                    "output": f"{nm}.md", "status": "ok",
                    "time_ocr": round(t_ocr, 2), "time_img": round(t_img, 2),
                    "avg_per_page": round(t_ocr / max(len(imgs), 1), 2)
                })
            shutil.rmtree(f"/tmp/pdf_pages/{nm}", ignore_errors=True)
        except Exception as e:
            traceback.print_exc()
            with state_lock:
                ocr_state["files_done"] = idx + 1
                ocr_state["results"].append({
                    "file": Path(pdf).name, "pages": 0, "output": "",
                    "status": f"error:{e}", "time_ocr": 0, "time_img": 0, "avg_per_page": 0
                })
    with state_lock:
        ocr_state["running"] = False
        ocr_state["current_file"] = ""


# === Flask App ===
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
app.secret_key = SECRET_KEY
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = False  # Cloud Run terminates TLS at LB
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB max upload


# --- Auth Routes ---
@app.route("/login", methods=["GET"])
def login_page():
    return LOGIN_HTML


@app.route("/login", methods=["POST"])
def login_submit():
    data = request.get_json(silent=True)
    pw = data.get("password", "") if data else request.form.get("password", "")
    if pw == APP_PASSWORD:
        session["authenticated"] = True
        session.permanent = True
        return jsonify({"ok": True})
    return jsonify({"error": "Mật khẩu sai"}), 401


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


# --- App Routes ---
@app.route("/")
@login_required
def index():
    return APP_HTML


@app.route("/api/upload", methods=["POST"])
@login_required
def upload():
    ups = []
    for f in request.files.getlist("files"):
        if f.filename and f.filename.lower().endswith(".pdf"):
            dest = os.path.join(INPUT_DIR, f.filename)
            f.save(dest)
            ups.append({"name": f.filename, "size": os.path.getsize(dest)})
    return jsonify({"uploaded": ups})


@app.route("/api/files")
@login_required
def list_files():
    pdfs = [{"name": f, "size": os.path.getsize(os.path.join(INPUT_DIR, f))}
            for f in sorted(os.listdir(INPUT_DIR)) if f.lower().endswith(".pdf")]
    mds = [{"name": f, "size": os.path.getsize(os.path.join(OUTPUT_DIR, f))}
           for f in sorted(os.listdir(OUTPUT_DIR)) if f.lower().endswith(".md")]
    return jsonify({"pdfs": pdfs, "markdowns": mds})


@app.route("/api/ocr", methods=["POST"])
@login_required
def start_ocr():
    if ocr_state["running"]:
        return jsonify({"error": "OCR đang chạy"}), 409
    pdfs = sorted(glob.glob(os.path.join(INPUT_DIR, "*.pdf")))
    if not pdfs:
        return jsonify({"error": "Không có file PDF"}), 400
    threading.Thread(target=ocr_worker, args=(pdfs,), daemon=True).start()
    return jsonify({"started": len(pdfs)})


@app.route("/api/status")
@login_required
def status():
    with state_lock:
        el = time.time() - ocr_state["start_time"] if ocr_state["start_time"] else 0
        return jsonify({**ocr_state, "elapsed": round(el, 1)})


@app.route("/api/preview/<fn>")
@login_required
def preview(fn):
    p = os.path.join(OUTPUT_DIR, fn)
    if not os.path.exists(p):
        return jsonify({"error": "Not found"}), 404
    return jsonify({"content": open(p, "r", encoding="utf-8").read(), "name": fn})


@app.route("/api/download/<fn>")
@login_required
def download(fn):
    return send_from_directory(OUTPUT_DIR, fn, as_attachment=True)


@app.route("/api/download_all")
@login_required
def download_all():
    shutil.make_archive("/tmp/ocr_results", "zip", OUTPUT_DIR)
    return send_file("/tmp/ocr_results.zip", as_attachment=True, download_name="ocr_results.zip")


@app.route("/api/delete/<fn>", methods=["DELETE"])
@login_required
def delete_file(fn):
    for d in [INPUT_DIR, OUTPUT_DIR]:
        p = os.path.join(d, fn)
        if os.path.exists(p):
            os.remove(p)
    return jsonify({"deleted": fn})


@app.route("/health")
def health():
    return jsonify({"status": "ok", "ocr_running": ocr_state["running"]})


@app.route("/api/vllm-status")
def vllm_status():
    import urllib.request
    try:
        req = urllib.request.urlopen(f"http://localhost:{VLLM_PORT}/health", timeout=3)
        return jsonify({"ready": True, "status": "online"})
    except Exception:
        return jsonify({"ready": False, "status": "loading"})


# === HTML Templates (inline) ===

LOGIN_HTML = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Đăng nhập — GLM-OCR</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{--bg:#0a0e17;--card:rgba(17,24,39,0.7);--border:rgba(255,255,255,0.08);--text:#e5e7eb;--text2:#9ca3af;--accent:#6366f1;--glow:rgba(99,102,241,0.3);--err:#ef4444}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Inter',system-ui,sans-serif;background:var(--bg);color:var(--text);min-height:100vh;display:flex;align-items:center;justify-content:center}
body::before{content:'';position:fixed;inset:0;background:radial-gradient(circle at 50% 40%,rgba(99,102,241,0.1),transparent 60%);z-index:0}
.box{background:var(--card);border:1px solid var(--border);border-radius:20px;padding:40px;width:100%;max-width:380px;backdrop-filter:blur(20px);position:relative;z-index:1;text-align:center}
h1{font-size:22px;margin-bottom:6px;background:linear-gradient(135deg,#818cf8,#6366f1);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
p{color:var(--text2);font-size:13px;margin-bottom:24px}
input{width:100%;padding:12px 16px;border-radius:10px;border:1px solid var(--border);background:rgba(255,255,255,0.05);color:var(--text);font-family:inherit;font-size:14px;margin-bottom:14px;outline:none;transition:border .2s}
input:focus{border-color:var(--accent);box-shadow:0 0 15px var(--glow)}
button{width:100%;padding:12px;border-radius:10px;border:none;background:linear-gradient(135deg,var(--accent),#4f46e5);color:#fff;font-family:inherit;font-size:14px;font-weight:600;cursor:pointer;transition:all .2s;box-shadow:0 4px 15px var(--glow)}
button:hover{transform:translateY(-1px)}
.err{color:var(--err);font-size:12px;margin-top:10px;display:none}
</style>
</head>
<body>
<div class="box">
  <h1>🔐 GLM-OCR</h1>
  <p>Nhập mật khẩu để truy cập</p>
  <form onsubmit="return doLogin(event)">
    <input type="password" id="pw" placeholder="Mật khẩu" autofocus>
    <button type="submit">Đăng nhập</button>
  </form>
  <div class="err" id="err">Mật khẩu không đúng</div>
</div>
<script>
async function doLogin(e){
  e.preventDefault();
  const pw=document.getElementById('pw').value;
  try{
    const r=await fetch('/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({password:pw})});
    const d=await r.json();
    if(d.ok){window.location='/';}
    else{document.getElementById('err').style.display='block';}
  }catch(ex){document.getElementById('err').textContent='Lỗi kết nối';document.getElementById('err').style.display='block';}
  return false;
}
</script>
</body></html>"""

APP_HTML = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GLM-OCR — PDF to Markdown</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{--bg:#0a0e17;--bg2:#111827;--card:rgba(17,24,39,0.7);--border:rgba(255,255,255,0.08);--text:#e5e7eb;--text2:#9ca3af;--accent:#6366f1;--accent2:#818cf8;--glow:rgba(99,102,241,0.3);--ok:#10b981;--err:#ef4444}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Inter',system-ui,sans-serif;background:var(--bg);color:var(--text);min-height:100vh}
body::before{content:'';position:fixed;inset:0;background:radial-gradient(circle at 30% 20%,rgba(99,102,241,0.08),transparent 50%),radial-gradient(circle at 70% 80%,rgba(16,185,129,0.06),transparent 50%);z-index:0}
.ct{max-width:1100px;margin:0 auto;padding:24px 20px;position:relative;z-index:1}
.hdr{text-align:center;margin-bottom:28px;padding:24px;background:var(--card);border:1px solid var(--border);border-radius:20px;backdrop-filter:blur(20px);position:relative}
.hdr h1{font-size:26px;font-weight:700;background:linear-gradient(135deg,#818cf8,#6366f1,#10b981);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.hdr p{color:var(--text2);margin-top:4px;font-size:13px}
.lo{position:absolute;top:16px;right:20px;padding:6px 14px;border-radius:8px;border:1px solid var(--border);background:rgba(255,255,255,0.04);color:var(--text2);font-size:11px;cursor:pointer;text-decoration:none;transition:all .2s}
.lo:hover{background:rgba(255,255,255,0.08);color:var(--text)}
.cd{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:20px;margin-bottom:16px;backdrop-filter:blur(20px)}
.cd h2{font-size:15px;font-weight:600;margin-bottom:14px;display:flex;align-items:center;gap:8px}
.uz{border:2px dashed rgba(99,102,241,0.3);border-radius:12px;padding:36px;text-align:center;cursor:pointer;transition:all .3s}
.uz:hover,.uz.dg{border-color:var(--accent);background:rgba(99,102,241,0.05);box-shadow:0 0 30px var(--glow)}
.uz .em{font-size:36px;margin-bottom:8px}
.uz p{color:var(--text2);font-size:13px}
.uz input{display:none}
.gr{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:768px){.gr{grid-template-columns:1fr}}
.fl{max-height:240px;overflow-y:auto}
.fi{display:flex;align-items:center;justify-content:space-between;padding:9px 12px;border-radius:8px;margin-bottom:5px;background:rgba(255,255,255,0.03);border:1px solid var(--border);font-size:12px}
.fi:hover{background:rgba(255,255,255,0.06)}
.fn{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.fs{color:var(--text2);font-size:11px;margin-left:8px}
.fa{display:flex;gap:4px;margin-left:8px}
.btn{display:inline-flex;align-items:center;gap:5px;padding:7px 14px;border-radius:8px;border:none;font-family:inherit;font-size:12px;font-weight:500;cursor:pointer;transition:all .2s;color:#fff}
.bp{background:linear-gradient(135deg,var(--accent),#4f46e5);box-shadow:0 4px 15px var(--glow)}
.bp:hover{transform:translateY(-1px)}
.bp:disabled{opacity:.4;cursor:not-allowed;transform:none}
.bs{padding:4px 9px;font-size:11px;border-radius:6px}
.bg{background:rgba(255,255,255,0.06);color:var(--text2)}
.bg:hover{background:rgba(255,255,255,0.1);color:var(--text)}
.bd{background:rgba(239,68,68,0.15);color:var(--err)}
.bk{background:rgba(16,185,129,0.15);color:var(--ok)}
.bf{width:100%;justify-content:center;padding:11px;font-size:14px}
.pb{display:none}.pb.on{display:block}
.pbw{height:8px;background:rgba(255,255,255,0.06);border-radius:4px;overflow:hidden;margin:10px 0}
.pbb{height:100%;border-radius:4px;transition:width .5s;background:linear-gradient(90deg,var(--accent),var(--ok));box-shadow:0 0 12px var(--glow)}
.pi{display:flex;justify-content:space-between;font-size:11px;color:var(--text2)}
.pf{font-size:12px;color:var(--accent2);margin-bottom:6px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.sr{display:flex;gap:12px;margin-top:10px;flex-wrap:wrap}
.st{padding:6px 12px;border-radius:8px;background:rgba(255,255,255,0.03);border:1px solid var(--border);font-size:11px}
.st b{color:var(--accent2)}
.mo{display:none;position:fixed;inset:0;background:rgba(0,0,0,0.7);z-index:1000;backdrop-filter:blur(4px);justify-content:center;align-items:center;padding:20px}
.mo.on{display:flex}
.ml{background:var(--bg2);border:1px solid var(--border);border-radius:16px;width:100%;max-width:800px;max-height:85vh;display:flex;flex-direction:column}
.mh{display:flex;justify-content:space-between;align-items:center;padding:14px 18px;border-bottom:1px solid var(--border)}
.mc{background:none;border:none;color:var(--text2);font-size:18px;cursor:pointer;padding:4px 8px;border-radius:6px}
.mc:hover{background:rgba(255,255,255,0.1)}
.mb{padding:18px;overflow-y:auto;flex:1;font-size:13px;line-height:1.7;white-space:pre-wrap;word-wrap:break-word}
.emp{text-align:center;padding:20px;color:var(--text2);font-size:12px}
::-webkit-scrollbar{width:5px}::-webkit-scrollbar-thumb{background:rgba(255,255,255,0.1);border-radius:3px}
.tt{position:fixed;bottom:20px;right:20px;z-index:2000;padding:10px 18px;border-radius:10px;font-size:12px;background:var(--bg2);border:1px solid var(--border);box-shadow:0 8px 32px rgba(0,0,0,.4);transform:translateY(80px);opacity:0;transition:all .3s}
.tt.on{transform:translateY(0);opacity:1}
@keyframes pl{0%,100%{opacity:1}50%{opacity:.5}}
.pls{animation:pl 1.5s ease-in-out infinite}
.tr{margin-top:16px}.tr table{width:100%;border-collapse:collapse;font-size:11px}.tr th{text-align:left;padding:8px 10px;border-bottom:2px solid var(--border);color:var(--text2);font-weight:600}.tr td{padding:7px 10px;border-bottom:1px solid var(--border)}.tr tr:hover td{background:rgba(255,255,255,0.03)}.tr tfoot td{border-top:2px solid var(--border);font-weight:600}
</style>
</head>
<body>
<div class="ct">
  <div class="hdr">
    <h1>🔵 GLM-OCR — PDF to Markdown</h1>
    <p>Upload PDF → OCR tự động → Tải Markdown</p>
    <div id="vllm-badge" style="position:absolute;top:16px;left:20px;padding:6px 14px;border-radius:8px;font-size:11px;border:1px solid var(--border);background:rgba(255,255,255,0.04);color:var(--text2)">vLLM: ...</div>
    <a href="/logout" class="lo">🔒 Đăng xuất</a>
  </div>
  <div class="cd">
    <h2>📤 Upload PDF</h2>
    <label for="fi" class="uz" id="uz"><div class="em">📁</div><p>Kéo thả file PDF hoặc <b>nhấn để chọn</b></p><input type="file" id="fi" multiple accept=".pdf"></label>
  </div>
  <div class="gr">
    <div class="cd">
      <h2>📋 File PDF <span id="pc" style="color:var(--text2);font-weight:400"></span></h2>
      <div id="pl" class="fl"><div class="emp">Chưa có file</div></div>
      <div style="margin-top:12px"><button class="btn bp bf" id="bo">🔍 Bắt đầu OCR</button></div>
    </div>
    <div class="cd">
      <h2>✅ Kết quả <span id="mc2" style="color:var(--text2);font-weight:400"></span></h2>
      <div id="ml" class="fl"><div class="emp">Chưa có kết quả</div></div>
      <div style="margin-top:12px"><button class="btn bk bf" id="btnAll">📦 Tải tất cả (ZIP)</button></div>
    </div>
  </div>
  <div class="cd pb" id="pbx">
    <h2><span class="pls">⚡</span> Đang OCR...</h2>
    <div class="pf" id="pfl">—</div>
    <div class="pbw"><div class="pbb" id="pba" style="width:0%"></div></div>
    <div class="pi"><span id="ptx">0%</span><span id="ptm">0s</span></div>
    <div class="sr"><div class="st">File: <b id="sf">0/0</b></div><div class="st">Trang: <b id="sp">0/0</b></div><div class="st">⏱ <b id="se">0s</b></div></div>
  </div>
  <div class="cd pb" id="trx"><h2>📊 Thời gian xử lý</h2><div id="trbody"></div></div>
</div>
<div class="mo" id="mov"><div class="ml"><div class="mh"><h3 id="mt" style="font-size:14px">Preview</h3><button class="mc" id="btnCloseMo">✕</button></div><div class="mb" id="mbd"></div></div></div>
<div class="tt" id="tt"></div>
<script>
var pt=null;
function fm(b){if(b<1024)return b+"B";if(b<1048576)return(b/1024).toFixed(1)+"KB";return(b/1048576).toFixed(1)+"MB";}
function tw(m){var e=document.getElementById("tt");e.textContent=m;e.className="tt on";setTimeout(function(){e.classList.remove("on");},3000);}
var uz=document.getElementById("uz"),fi=document.getElementById("fi");
uz.addEventListener("dragover",function(e){e.preventDefault();uz.classList.add("dg");});
uz.addEventListener("dragleave",function(){uz.classList.remove("dg");});
uz.addEventListener("drop",function(e){e.preventDefault();uz.classList.remove("dg");uF(e.dataTransfer.files);});
fi.addEventListener("change",function(){if(fi.files.length)uF(fi.files);});
function uF(fs){var fd=new FormData();var c=0;for(var i=0;i<fs.length;i++){if(fs[i].name.toLowerCase().endsWith(".pdf")){fd.append("files",fs[i]);c++;}}if(!c){tw("Chi ho tro PDF");return;}fetch("/api/upload",{method:"POST",body:fd}).then(function(r){return r.json();}).then(function(r){if(r.error==="Unauthorized"){window.location="/login";return;}tw("Upload "+r.uploaded.length+" file");rF();}).catch(function(e){tw("Loi: "+e.message);});}
function rF(){fetch("/api/files").then(function(r){if(r.status===401){window.location="/login";return;}return r.json();}).then(function(d){if(d){rP(d.pdfs);rM(d.markdowns);}}).catch(function(e){console.error(e);});}
function rP(fs){var e=document.getElementById("pl");document.getElementById("pc").textContent=fs.length?"("+fs.length+")":"";if(!fs.length){e.innerHTML="<div class='emp'>Chua co file</div>";return;}var h="";for(var i=0;i<fs.length;i++){h+="<div class='fi'><span class='fn'>"+fs[i].name+"</span><span class='fs'>"+fm(fs[i].size)+"</span><div class='fa'><button class='btn bs bd' data-del='"+fs[i].name+"'>x</button></div></div>";}e.innerHTML=h;e.querySelectorAll("[data-del]").forEach(function(b){b.addEventListener("click",function(){dF(b.getAttribute("data-del"));});});}
function rM(fs){var e=document.getElementById("ml");document.getElementById("mc2").textContent=fs.length?"("+fs.length+")":"";if(!fs.length){e.innerHTML="<div class='emp'>Chua co ket qua</div>";return;}var h="";for(var i=0;i<fs.length;i++){h+="<div class='fi'><span class='fn'>"+fs[i].name+"</span><span class='fs'>"+fm(fs[i].size)+"</span><div class='fa'><button class='btn bs bg' data-pv='"+fs[i].name+"'>&#9776;</button><button class='btn bs bk' data-dl='"+fs[i].name+"'>&#8595;</button></div></div>";}e.innerHTML=h;e.querySelectorAll("[data-pv]").forEach(function(b){b.addEventListener("click",function(){pV(b.getAttribute("data-pv"));});});e.querySelectorAll("[data-dl]").forEach(function(b){b.addEventListener("click",function(){dD(b.getAttribute("data-dl"));});});}
function sOcr(){fetch("/api/ocr",{method:"POST",headers:{"Content-Type":"application/json"}}).then(function(r){return r.json();}).then(function(r){if(r.error){tw(r.error);return;}tw("OCR "+r.started+" file");document.getElementById("pbx").classList.add("on");document.getElementById("bo").disabled=true;sP();}).catch(function(e){tw("Loi: "+e.message);});}
document.getElementById("bo").addEventListener("click",sOcr);
function sP(){if(pt)clearInterval(pt);pt=setInterval(pS,1000);}
function pS(){fetch("/api/status").then(function(r){if(r.status===401){window.location="/login";return null;}return r.json();}).then(function(s){if(!s)return;var fd=s.files_done||0,ft=s.files_total||0,cp=s.current_page||0,tp=s.total_pages||0,el=s.elapsed||0;var p=ft>0?Math.round(((fd+(tp>0?cp/tp:0))/ft)*100):0;document.getElementById("pba").style.width=p+"%";document.getElementById("ptx").textContent=p+"%";document.getElementById("pfl").textContent=s.current_file||"--";document.getElementById("ptm").textContent=el+"s";document.getElementById("sf").textContent=fd+"/"+ft;document.getElementById("sp").textContent=cp+"/"+tp;document.getElementById("se").textContent=el+"s";if(!s.running&&ft>0){clearInterval(pt);pt=null;document.getElementById("pba").style.width="100%";document.getElementById("ptx").textContent="100%";document.getElementById("bo").disabled=false;var ok=s.results?s.results.filter(function(r){return r.status==="ok";}).length:0;tw("Xong! "+ok+"/"+ft+" OK ("+el+"s)");buildReport(s);setTimeout(function(){document.getElementById("pbx").classList.remove("on");rF();},2000);}}).catch(function(e){console.error(e);});}
function buildReport(s){if(!s.results||!s.results.length)return;var box=document.getElementById("trx");box.classList.add("on");var h="<div class='tr'><table><thead><tr><th>File</th><th>Trang</th><th>PDF-Anh</th><th>OCR</th><th>TB/trang</th></tr></thead><tbody>";var tP=0,tO=0,tI=0;for(var i=0;i<s.results.length;i++){var r=s.results[i];if(r.status==="ok"){h+="<tr><td>"+r.file+"</td><td>"+r.pages+"</td><td>"+r.time_img+"s</td><td>"+r.time_ocr+"s</td><td>"+r.avg_per_page+"s</td></tr>";tP+=r.pages;tO+=r.time_ocr;tI+=r.time_img;}}h+="</tbody><tfoot><tr><td><b>TONG</b></td><td>"+tP+"</td><td>"+tI.toFixed(1)+"s</td><td>"+tO.toFixed(1)+"s</td><td>"+(tO/Math.max(tP,1)).toFixed(2)+"s</td></tr></tfoot></table></div>";document.getElementById("trbody").innerHTML=h;}
function dF(n){fetch("/api/delete/"+n,{method:"DELETE"}).then(function(){rF();});}
function pV(n){fetch("/api/preview/"+n).then(function(r){if(r.status===401){window.location="/login";return;}return r.json();}).then(function(d){if(d){document.getElementById("mt").textContent=n;document.getElementById("mbd").textContent=d.content;document.getElementById("mov").classList.add("on");}}).catch(function(){tw("Loi");});}
function cMo(){document.getElementById("mov").classList.remove("on");}
function dD(n){window.open("/api/download/"+n);}
function dAll(){window.open("/api/download_all");}
document.getElementById("btnAll").addEventListener("click",dAll);
document.getElementById("btnCloseMo").addEventListener("click",cMo);
document.getElementById("mov").addEventListener("click",function(e){if(e.target===document.getElementById("mov"))cMo();});
function checkVllm(){fetch("/api/vllm-status").then(function(r){return r.json();}).then(function(d){var b=document.getElementById("vllm-badge");if(d.ready){b.innerHTML="vLLM: <b style='color:#10b981'>Online</b>";b.style.borderColor="rgba(16,185,129,0.3)";document.getElementById("bo").disabled=false;}else{b.innerHTML="vLLM: <b style='color:#f59e0b'>Loading...</b>";b.style.borderColor="rgba(245,158,11,0.3)";document.getElementById("bo").disabled=true;setTimeout(checkVllm,5000);}}).catch(function(){var b=document.getElementById("vllm-badge");b.innerHTML="vLLM: <b style='color:#ef4444'>Offline</b>";setTimeout(checkVllm,10000);});}
checkVllm();
rF();
fetch("/api/status").then(function(r){return r.json();}).then(function(s){if(s&&s.running){document.getElementById("pbx").classList.add("on");document.getElementById("bo").disabled=true;sP();}}).catch(function(){});
</script>
</body></html>"""


if __name__ == "__main__":
    print(f"🔐 Password: {APP_PASSWORD}")
    print(f"🌐 Starting on port {PORT}...")
    app.run(host="0.0.0.0", port=PORT, debug=False)
