import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2, numpy as np, math, threading, base64, csv, io, time, warnings
warnings.filterwarnings('ignore')
from datetime import datetime
from flask import Flask, Response, jsonify, send_from_directory, request
from flask_cors import CORS

print("Loading AI emotion model... (30 seconds first time)")
emotion_model = None
try:
    from deepface import DeepFace
    emotion_model = DeepFace
    print("✓ DeepFace emotion model ready")
except Exception as e:
    print(f"⚠ DeepFace not available: {e}")

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

os.makedirs("intruder_shots", exist_ok=True)
os.makedirs("whitelist",      exist_ok=True)

state = {
    "faces": [], "face_count": 0, "fps": 0,
    "night_mode": False, "privacy_mode": False,
    "attendance": [], "event_log": [],
    "liveness": {"score": 0, "label": "INACTIVE"},
    "emotion": {"happy":0,"neutral":0,"surprised":0,"angry":0,"sad":0,"disgust":0,"fear":0},
    "distance": {"label":"---","value":"---"},
    "alarm": False, "alarm_reason": "",
    "locked": False,
    "intrusion_count": 0,
    "whitelist": [],
    "whitelist_registering": False,
    "threat_level": "CLEAR",
    "session_last_face": time.time(),
    "session_timeout": 15,
}
state_lock = threading.Lock()

liveness_data = {
    "blink_count": 0,
    "eye_closed":  False,
    "last_blink":  time.time(),
}

latest_frame_for_emotion = None
latest_emotion = {"happy":0,"neutral":100,"surprised":0,"angry":0,"sad":0,"disgust":0,"fear":0}
emotion_lock = threading.Lock()

def emotion_worker():
    global latest_emotion
    while True:
        time.sleep(4.0)
        with emotion_lock:
            frame = latest_frame_for_emotion
        if frame is None or emotion_model is None:
            continue
        try:
            result = emotion_model.analyze(
                frame, actions=["emotion"],
                enforce_detection=False, silent=True
            )
            if result and isinstance(result, list):
                raw   = result[0].get("emotion", {})
                total = sum(raw.values()) or 1
                mapped = {
                    "happy":     round(raw.get("happy",    0) / total * 100),
                    "neutral":   round(raw.get("neutral",  0) / total * 100),
                    "surprised": round(raw.get("surprise", 0) / total * 100),
                    "angry":     round(raw.get("angry",    0) / total * 100),
                    "sad":       round(raw.get("sad",      0) / total * 100),
                    "disgust":   round(raw.get("disgust",  0) / total * 100),
                    "fear":      round(raw.get("fear",     0) / total * 100),
                }
                with emotion_lock:
                    latest_emotion = mapped
        except Exception:
            pass

threading.Thread(target=emotion_worker, daemon=True).start()

camera = None
camera_lock = threading.Lock()

def get_camera():
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            camera.set(cv2.CAP_PROP_FPS, 30)
            camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    return camera

def log_event(msg, kind="info"):
    entry = {"time": datetime.now().strftime("%H:%M:%S"), "msg": msg, "kind": kind}
    with state_lock:
        state["event_log"].insert(0, entry)
        if len(state["event_log"]) > 80:
            state["event_log"].pop()

def estimate_distance(face_w, frame_w):
    ratio = face_w / frame_w
    if ratio > 0.35: return "CLOSE", "~30 cm"
    if ratio > 0.18: return "NEAR",  "~60 cm"
    if ratio > 0.09: return "MID",   "~1.2 m"
    return "FAR", "~2.5 m+"

def check_liveness(face_gray):
    eyes        = eye_cascade.detectMultiScale(face_gray, 1.1, 5, minSize=(20,20))
    eyes_found  = len(eyes) >= 1
    t           = time.time()
    lap_var     = cv2.Laplacian(face_gray, cv2.CV_64F).var()
    texture     = min(100, int(lap_var / 3))

    if eyes_found:
        if liveness_data["eye_closed"]:
            liveness_data["blink_count"] += 1
            liveness_data["last_blink"]   = t
        liveness_data["eye_closed"] = False
    else:
        liveness_data["eye_closed"] = True

    if t - liveness_data["last_blink"] > 8:
        liveness_data["blink_count"] = max(0, liveness_data["blink_count"] - 1)

    score = min(100, min(60, liveness_data["blink_count"] * 20) + texture // 3)
    label = "LIVE" if score > 55 else "UNCERTAIN" if score > 30 else "SPOOF RISK"
    return score, label

def generate_frames():
    global latest_frame_for_emotion
    cam         = get_camera()
    prev_time   = time.time()
    frame_count = 0
    prev_count  = -1
    face_ids    = {}
    fid_counter = [0]
    COLORS      = [(0,220,255),(0,255,157),(255,180,0),(180,100,255),(255,80,80)]

    while True:
        ok, frame = cam.read()
        if not ok:
            time.sleep(0.03)
            continue

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        t     = time.time()

        frame_count += 1
        elapsed = t - prev_time
        fps = round(frame_count / elapsed) if elapsed > 0 else 0
        if elapsed > 1.0:
            prev_time = t
            frame_count = 0

        with state_lock:
            night       = state["night_mode"]
            privacy     = state["privacy_mode"]
            wl          = list(state["whitelist"])
            registering = state["whitelist_registering"]

        if night:
            frame = cv2.convertScaleAbs(frame, alpha=1.6, beta=40)

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray  = cv2.equalizeHist(gray)
        faces = face_cascade.detectMultiScale(gray, 1.15, 5, minSize=(60,60))
        face_count = len(faces)

        if face_count > 0:
            with state_lock:
                state["session_last_face"] = t
                if state["locked"]:
                    state["locked"]       = False
                    state["threat_level"] = "CLEAR"
                    log_event("SESSION UNLOCKED", "detect")

        with state_lock:
            idle    = t - state["session_last_face"]
            timeout = state["session_timeout"]
            if idle > timeout and not state["locked"]:
                state["locked"]       = True
                state["threat_level"] = "LOCKDOWN"
                log_event("SESSION TIMEOUT — LOCKED", "alert")

        liveness_score, liveness_label = 0, "INACTIVE"
        if face_count > 0:
            x0,y0,w0,h0 = faces[0]
            liveness_score, liveness_label = check_liveness(gray[y0:y0+h0, x0:x0+w0])

        if registering and face_count > 0:
            x,y,fw,fh = faces[0]
            fname = f"whitelist/face_{len(os.listdir('whitelist'))}.jpg"
            cv2.imwrite(fname, frame[y:y+fh, x:x+fw])
            with state_lock:
                state["whitelist"].append(fname)
                state["whitelist_registering"] = False
            log_event(f"FACE REGISTERED — {len(state['whitelist'])} in whitelist", "detect")

        detected_faces = []
        new_ids = {}

        for i,(x,y,fw,fh) in enumerate(faces):
            key = f"face_{i}"
            if key not in face_ids:
                fid_counter[0] += 1
                face_ids[key]   = f"F{fid_counter[0]}"
                fid             = face_ids[key]
                log_event(f"SUBJECT {fid} ENTERED FRAME", "detect")
                with state_lock:
                    state["attendance"].append({
                        "id":   fid,
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "date": datetime.now().strftime("%Y-%m-%d"),
                    })
                is_known = False
                for wp in wl:
                    try:
                        known_img = cv2.imread(wp)
                        known_img = cv2.cvtColor(known_img, cv2.COLOR_BGR2GRAY)
                        known_img = cv2.resize(known_img, (100, 100))
                        face_img  = cv2.cvtColor(frame[y:y+fh, x:x+fw], cv2.COLOR_BGR2GRAY)
                        face_img  = cv2.resize(face_img, (100, 100))
                        diff      = cv2.absdiff(known_img, face_img)
                        score     = np.mean(diff)
                        if score < 50:
                            is_known = True
                            break
                    except:
                        pass
                if len(wl) > 0 and not is_known:
                    shot = f"intruder_shots/{fid}_{datetime.now().strftime('%H%M%S')}.jpg"
                    cv2.imwrite(shot, frame[y:y+fh, x:x+fw])
                    with state_lock:
                        state["intrusion_count"] += 1
                        state["alarm"]            = True
                        state["alarm_reason"]     = f"UNKNOWN FACE {fid}"
                        state["threat_level"]     = "ALERT"
                    log_event(f"⚠ INTRUDER — {fid} NOT IN WHITELIST", "alert")

            fid = face_ids[key]
            new_ids[key] = fid
            color = COLORS[i % len(COLORS)]
            conf  = round(0.82 + math.sin(t*2+i)*0.08, 2)
            detected_faces.append({"id":fid,"x":int(x),"y":int(y),"w":int(fw),"h":int(fh),"conf":conf})

            if privacy:
                roi = frame[y:y+fh, x:x+fw]
                frame[y:y+fh, x:x+fw] = cv2.GaussianBlur(roi,(51,51),0)

            cs = 22
            cv2.line(frame,(x,y),(x+cs,y),color,2)
            cv2.line(frame,(x,y),(x,y+cs),color,2)
            cv2.line(frame,(x+fw-cs,y),(x+fw,y),color,2)
            cv2.line(frame,(x+fw,y),(x+fw,y+cs),color,2)
            cv2.line(frame,(x,y+fh-cs),(x,y+fh),color,2)
            cv2.line(frame,(x,y+fh),(x+cs,y+fh),color,2)
            cv2.line(frame,(x+fw-cs,y+fh),(x+fw,y+fh),color,2)
            cv2.line(frame,(x+fw,y+fh-cs),(x+fw,y+fh),color,2)
            cv2.putText(frame, f"{fid} {round(conf*100)}%  {liveness_label}",
                        (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

        face_ids = new_ids if face_count > 0 else {}

        if face_count != prev_count:
            log_event(f"{face_count} FACE{'S' if face_count>1 else ''} IN FRAME"
                      if face_count > 0 else "FRAME CLEARED")
            prev_count = face_count

        with state_lock:
            alarm  = state["alarm"]
            locked = state["locked"]

        if alarm:
            alpha   = 0.2 + 0.15 * abs(math.sin(t * 6))
            overlay = frame.copy()
            cv2.rectangle(overlay,(0,0),(w,h),(0,0,200),-1)
            cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)
            cv2.putText(frame,"INTRUSION ALERT",(w//2-180,h//2),
                        cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),3,cv2.LINE_AA)

        if locked:
            overlay = frame.copy()
            cv2.rectangle(overlay,(0,0),(w,h),(0,0,0),-1)
            cv2.addWeighted(overlay,0.78,frame,0.22,0,frame)
            cv2.putText(frame,"SESSION LOCKED",(w//2-200,h//2-20),
                        cv2.FONT_HERSHEY_SIMPLEX,1.4,(0,200,230),3,cv2.LINE_AA)
            cv2.putText(frame,"Show face to unlock",(w//2-160,h//2+30),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(80,160,180),2,cv2.LINE_AA)

        now_str = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame,f"IRIS v2.0 | {now_str} | FPS:{fps}",(8,22),
                    cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,200,220),1,cv2.LINE_AA)
        mode = ("NIGHT " if night else "")+("PRIVACY " if privacy else "") or "STANDARD"
        cv2.putText(frame,f"MODE:{mode} | FACES:{face_count} | {liveness_label}",(8,42),
                    cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,200,220),1,cv2.LINE_AA)

        if face_count > 0:
            with emotion_lock:
                latest_frame_for_emotion = cv2.resize(frame, (320, 240))

        with state_lock:
            state["faces"]      = detected_faces
            state["face_count"] = face_count
            state["fps"]        = fps
            state["liveness"]   = {"score": liveness_score, "label": liveness_label}
            if face_count > 0:
                biggest = max(detected_faces, key=lambda f: f["w"]*f["h"])
                dk, dv  = estimate_distance(biggest["w"], w)
                state["distance"] = {"label":dk,"value":dv}
            else:
                state["distance"] = {"label":"---","value":"---"}
                state["alarm"]    = False

        with emotion_lock:
            em = dict(latest_emotion)
        with state_lock:
            state["emotion"] = em

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/state")
def api_state():
    with state_lock:
        return jsonify({
            "face_count":      state["face_count"],
            "fps":             state["fps"],
            "night_mode":      state["night_mode"],
            "privacy_mode":    state["privacy_mode"],
            "emotion":         state["emotion"],
            "distance":        state["distance"],
            "liveness":        state["liveness"],
            "attendance":      state["attendance"][-30:],
            "event_log":       state["event_log"][:30],
            "alarm":           state["alarm"],
            "alarm_reason":    state["alarm_reason"],
            "locked":          state["locked"],
            "threat_level":    state["threat_level"],
            "intrusion_count": state["intrusion_count"],
            "whitelist_count": len(state["whitelist"]),
            "session_timeout": state["session_timeout"],
        })

@app.route("/api/toggle", methods=["POST"])
def api_toggle():
    data = request.json or {}
    key  = data.get("key")
    val  = None
    with state_lock:
        if key in ("night_mode","privacy_mode"):
            state[key] = not state[key]
            val = state[key]
    log_event(f"{'NIGHT' if key=='night_mode' else 'PRIVACY'} {'ON' if val else 'OFF'}")
    return jsonify({"ok": True, "value": val})

@app.route("/api/dismiss_alarm", methods=["POST"])
def dismiss_alarm():
    with state_lock:
        state["alarm"]        = False
        state["alarm_reason"] = ""
        state["threat_level"] = "CLEAR"
    log_event("ALARM DISMISSED", "info")
    return jsonify({"ok": True})

@app.route("/api/register_face", methods=["POST"])
def register_face():
    with state_lock:
        state["whitelist_registering"] = True
    log_event("REGISTERING FACE...", "detect")
    return jsonify({"ok": True})

@app.route("/api/clear_whitelist", methods=["POST"])
def clear_whitelist():
    import shutil
    with state_lock:
        state["whitelist"]    = []
        state["alarm"]        = False
        state["threat_level"] = "CLEAR"
    shutil.rmtree("whitelist", ignore_errors=True)
    os.makedirs("whitelist", exist_ok=True)
    log_event("WHITELIST CLEARED", "info")
    return jsonify({"ok": True})

@app.route("/api/set_timeout", methods=["POST"])
def set_timeout():
    data = request.json or {}
    secs = int(data.get("seconds", 15))
    with state_lock:
        state["session_timeout"] = secs
    log_event(f"TIMEOUT SET TO {secs}s", "info")
    return jsonify({"ok": True})

@app.route("/api/snapshot", methods=["POST"])
def api_snapshot():
    cam = get_camera()
    ret, frame = cam.read()
    if ret:
        frame = cv2.flip(frame, 1)
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        b64 = base64.b64encode(buf).decode()
        log_event("SNAPSHOT CAPTURED", "alert")
        return jsonify({"ok": True, "image": b64})
    return jsonify({"ok": False})

@app.route("/api/export_csv")
def export_csv():
    with state_lock:
        rows = list(state["attendance"])
    out = io.StringIO()
    wr  = csv.DictWriter(out, fieldnames=["id","time","date"])
    wr.writeheader()
    wr.writerows(rows)
    return Response(out.getvalue().encode(), mimetype="text/csv",
                    headers={"Content-Disposition":"attachment; filename=iris_attendance.csv"})

if __name__ == "__main__":
    print("\n╔══════════════════════════════════════╗")
    print("║   IRIS v2.0 — Cybersec Edition       ║")
    print("║   Open: http://localhost:5000        ║")
    print("╚══════════════════════════════════════╝\n")
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
