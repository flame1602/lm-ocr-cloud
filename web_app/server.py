"""
GLM-OCR Web Server — Flask backend chạy trên Google Colab.
Cung cấp API để upload PDF, chạy OCR, theo dõi tiến trình, và tải kết quả.
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
from pathlib import Path
from flask import Flask, request, jsonify, send_file, send_from_directory

import fitz  # PyMuPDF
from openai import OpenAI

# ===== CẤU HÌNH =====
INPUT_DIR = "/content/input_pdfs"
OUTPUT_DIR = "/content/output_markdown"
VLLM_PORT = 8899
WEB_PORT = 5000
DPI = 200

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== TRẠNG THÁI OCR =====
ocr_state = {
    "running": False,
    "current_file": "",
    "current_page": 0,
    "total_pages": 0,
    "files_done": 0,
    "files_total": 0,
    "results": [],
    "error": None,
    "start_time": 0,
}
state_lock = threading.Lock()

# ===== OCR FUNCTIONS =====

def pdf_to_images(pdf_path, dpi=200):
    doc = fitz.open(pdf_path)
    image_paths = []
    temp_dir = f"/tmp/pdf_pages/{Path(pdf_path).stem}"
    os.makedirs(temp_dir, exist_ok=True)
    for i in range(len(doc)):
        page = doc[i]
        zoom = dpi / 72
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        img_path = os.path.join(temp_dir, f"page_{i+1:04d}.png")
        pix.save(img_path)
        image_paths.append(img_path)
    doc.close()
    return image_paths


def ocr_with_api(image_paths, on_page=None):
    client = OpenAI(api_key="EMPTY", base_url=f"http://localhost:{VLLM_PORT}/v1")
    all_md = []
    for i, img_path in enumerate(image_paths):
        if on_page:
            on_page(i + 1, len(image_paths))
        with open(img_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        resp = client.chat.completions.create(
            model="glm-ocr",
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": "Text Recognition:"},
            ]}],
            max_tokens=8192, temperature=0.01,
        )
        all_md.append(resp.choices[0].message.content)
    return "\n\n---\n\n".join(all_md)


def ocr_worker(pdf_files):
    global ocr_state
    with state_lock:
        ocr_state.update({
            "running": True, "files_done": 0, "files_total": len(pdf_files),
            "results": [], "error": None, "start_time": time.time(),
        })

    for idx, pdf_path in enumerate(pdf_files):
        pdf_name = Path(pdf_path).stem
        with state_lock:
            ocr_state["current_file"] = Path(pdf_path).name
            ocr_state["current_page"] = 0
            ocr_state["total_pages"] = 0

        try:
            images = pdf_to_images(pdf_path, dpi=DPI)
            with state_lock:
                ocr_state["total_pages"] = len(images)

            def on_page(page, total):
                with state_lock:
                    ocr_state["current_page"] = page

            # Try SDK first, fallback to API
            md_text = ""
            try:
                from glmocr import parse
                result = parse(images)
                out_dir = os.path.join(OUTPUT_DIR, pdf_name)
                os.makedirs(out_dir, exist_ok=True)
                result.save(output_dir=out_dir)
                md_file = os.path.join(out_dir, "result.md")
                if os.path.exists(md_file):
                    md_text = open(md_file, 'r', encoding='utf-8').read()
                else:
                    md_text = str(result)
                with state_lock:
                    ocr_state["current_page"] = len(images)
            except:
                md_text = ocr_with_api(images, on_page=on_page)

            md_out = os.path.join(OUTPUT_DIR, f"{pdf_name}.md")
            with open(md_out, 'w', encoding='utf-8') as f:
                f.write(f"<!-- OCR by GLM-OCR | Source: {Path(pdf_path).name} -->\n\n{md_text}")

            with state_lock:
                ocr_state["files_done"] = idx + 1
                ocr_state["results"].append({
                    "file": Path(pdf_path).name, "pages": len(images),
                    "output": f"{pdf_name}.md", "status": "ok"
                })

            shutil.rmtree(f"/tmp/pdf_pages/{pdf_name}", ignore_errors=True)

        except Exception as e:
            with state_lock:
                ocr_state["files_done"] = idx + 1
                ocr_state["results"].append({
                    "file": Path(pdf_path).name, "pages": 0,
                    "output": "", "status": f"error: {str(e)[:100]}"
                })
            traceback.print_exc()

    with state_lock:
        ocr_state["running"] = False
        ocr_state["current_file"] = ""


# ===== FLASK APP =====
app = Flask(__name__)

@app.route("/")
def index():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        return send_file(html_path)
    return "<h1>GLM-OCR Web</h1><p>index.html not found</p>", 404


@app.route("/api/upload", methods=["POST"])
def upload():
    if "files" not in request.files:
        return jsonify({"error": "No files"}), 400
    uploaded = []
    for f in request.files.getlist("files"):
        if f.filename and f.filename.lower().endswith(".pdf"):
            dest = os.path.join(INPUT_DIR, f.filename)
            f.save(dest)
            uploaded.append({"name": f.filename, "size": os.path.getsize(dest)})
    return jsonify({"uploaded": uploaded})


@app.route("/api/files")
def list_files():
    pdfs = []
    for f in sorted(os.listdir(INPUT_DIR)):
        if f.lower().endswith(".pdf"):
            path = os.path.join(INPUT_DIR, f)
            pdfs.append({"name": f, "size": os.path.getsize(path)})

    mds = []
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.lower().endswith(".md"):
            path = os.path.join(OUTPUT_DIR, f)
            mds.append({"name": f, "size": os.path.getsize(path)})

    return jsonify({"pdfs": pdfs, "markdowns": mds})


@app.route("/api/ocr", methods=["POST"])
def start_ocr():
    if ocr_state["running"]:
        return jsonify({"error": "OCR đang chạy"}), 409

    data = request.get_json(silent=True) or {}
    selected = data.get("files", None)

    if selected:
        pdf_files = [os.path.join(INPUT_DIR, f) for f in selected
                     if os.path.exists(os.path.join(INPUT_DIR, f))]
    else:
        pdf_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.pdf")))

    if not pdf_files:
        return jsonify({"error": "Không có file PDF"}), 400

    thread = threading.Thread(target=ocr_worker, args=(pdf_files,), daemon=True)
    thread.start()
    return jsonify({"started": len(pdf_files)})


@app.route("/api/status")
def status():
    with state_lock:
        elapsed = time.time() - ocr_state["start_time"] if ocr_state["start_time"] else 0
        return jsonify({**ocr_state, "elapsed": round(elapsed, 1)})


@app.route("/api/preview/<filename>")
def preview(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return jsonify({"error": "Not found"}), 404
    with open(path, "r", encoding="utf-8") as f:
        return jsonify({"content": f.read(), "name": filename})


@app.route("/api/download/<filename>")
def download(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


@app.route("/api/download_all")
def download_all():
    zip_path = "/tmp/ocr_results"
    shutil.make_archive(zip_path, "zip", OUTPUT_DIR)
    return send_file(f"{zip_path}.zip", as_attachment=True, download_name="ocr_results.zip")


@app.route("/api/delete/<filename>", methods=["DELETE"])
def delete_file(filename):
    for d in [INPUT_DIR, OUTPUT_DIR]:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            os.remove(p)
    return jsonify({"deleted": filename})


def run_server():
    app.run(host="0.0.0.0", port=WEB_PORT, debug=False, use_reloader=False)
