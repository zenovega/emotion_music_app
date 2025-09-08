from flask import Flask, request, jsonify, render_template
from deepface import DeepFace
import numpy as np
import cv2

app = Flask(__name__)

# 感情→YouTube(新タブ用) URL
def get_watch_url(emotion: str) -> str:
    vids = {
        'happy':    '1k8craCGpgs',   # Don't Stop Believin'
        'sad':      '4UQrcMVAkOI',   # Ask the Lonely
        'angry':    'LatorN4P9aA',   # Separate Ways
        'surprise': 'atxUuldUcfI',   # Any Way You Want It
        'neutral':  'tNG62fULYgI',   # Lights
        'fear':     'OMD8hBsA-RI',   # Faithfully
        'disgust':  'MxGEVIvSFeY',   # Wheel in the Sky
    }
    video_id = vids.get(emotion, vids['neutral'])
    return f"https://www.youtube.com/watch?v={video_id}&autoplay=1"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    f = request.files.get("image")
    if not f:
        return jsonify({"error": "no image"}), 400

    # ---- メモリで画像復元（Windowsのファイルロック回避）----
    data = f.read()
    np_arr = np.frombuffer(data, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return jsonify({"error": "invalid image"}), 400

    # ---- 検出を堅牢化：BGR→(失敗時)RGB→(最終)enforce_detection=False ----
    try:
        result = DeepFace.analyze(
            img_bgr, actions=["emotion"], detector_backend="opencv", enforce_detection=True
        )
    except Exception:
        try:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            result = DeepFace.analyze(
                img_rgb, actions=["emotion"], detector_backend="opencv", enforce_detection=True
            )
        except Exception:
            result = DeepFace.analyze(
                img_bgr, actions=["emotion"], detector_backend="opencv", enforce_detection=False
            )

    # 返り値の形ゆれを吸収
    if isinstance(result, list) and result:
        emo = result[0].get("dominant_emotion", "neutral")
    elif isinstance(result, dict):
        emo = result.get("dominant_emotion", "neutral")
    else:
        emo = "neutral"

    return jsonify({
        "emotion": emo,
        "watch_url": get_watch_url(emo)
    })

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

