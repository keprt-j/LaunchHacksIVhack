from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
from openai import OpenAI
from deepface import DeepFace
import cv2
import numpy as np
import mediapipe as mp
import os
import requests
from werkzeug.utils import secure_filename
from scipy.spatial.distance import euclidean
from chatbot.chatbot_routes import chatbot_bp

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")
client = OpenAI(api_key="OPENAI_API_KEY")

UPLOAD_FOLDER = "uploads"
SURVEILLANCE_FOLDER = "surveillance"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SURVEILLANCE_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SURVEILLANCE_FOLDER'] = SURVEILLANCE_FOLDER
app.register_blueprint(chatbot_bp)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/live")
def live():
    """Display live emotion detection with embedded chatbot."""
    return render_template("live_result.html")


@app.route("/detect_emotion", methods=["POST"])
def detect_emotion():
    file = request.files['frame']
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    try:
        result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        if isinstance(result, list):
            result = result[0]
        
        emotion = result['dominant_emotion']
        if emotion in ["neutral", "disgust"]:
            emotion = "other"
        emotions = result['emotion']
    except Exception as e:
        emotion = None
        emotions = {}

    emotions = {k: float(v) for k, v in emotions.items()} if emotions else {}

    prompt = (
        f"Subject is feeling {emotion}. "
        f"Here are the emotion probabilities: {emotions}. "
        "Explain in 1-2 brief sentences, less than 15 words each, why the subject might be feeling this way, in context of an interrogation scene. Stick to facts, not stories or a dramatization."
    )

    chatbot_url = request.url_root.rstrip('/') + '/api/chat'
    try:
        chat_response = requests.post(
            chatbot_url,
            json={"message": prompt},
        )
        if chat_response.ok:
            explanation = chat_response.json().get("response", "")
        else:
            explanation = f"Chatbot error: {chat_response.text}"
    except Exception as e:
        explanation = f"Error: {e}"

    return jsonify({
        "emotion": emotion,
        "emotions": emotions,
        "prompt": prompt,
        "explanation": explanation
    })

@app.route("/analyze_video", methods=["GET", "POST"])
def analyze_video():
    if request.method == "POST":
        file = request.files['video']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        cap = cv2.VideoCapture(filepath)

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        interval = int(fps * 5)
        emotions_timeline = []
        frame_idx = 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        if fourcc == 0:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if fourcc == 0:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            
        processed_filename = "processed_" + filename
        processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                try:
                    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
                    if isinstance(result, list) and result:
                        emotion = result[0]['dominant_emotion']
                    else:
                        emotion = "Unknown"
                except Exception as e:
                    print(f"DeepFace analysis failed: {e}")
                    emotion = "Unknown"
                emotions_timeline.append({"time": round(frame_idx / fps, 2), "emotion": emotion})
            frame_idx += 1
        cap.release()

        LEFT_EYE_LANDMARKS = [33, 133, 160, 159, 158, 157, 173, 246]
        RIGHT_EYE_LANDMARKS = [362, 263, 387, 386, 385, 384, 398, 466]
        IRIS_LEFT = [468, 469, 470, 471, 472]
        IRIS_RIGHT = [473, 474, 475, 476, 477]
        BLINK_THRESHOLD = 0.20
        blink_times = []
        prev_eye_open = True
        frame_number = 0

        with mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as face_mesh:
            cap = cv2.VideoCapture(filepath)
            out = cv2.VideoWriter(processed_filepath, fourcc, fps, (width, height))
                
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        for idx in LEFT_EYE_LANDMARKS + RIGHT_EYE_LANDMARKS + IRIS_LEFT + IRIS_RIGHT:
                            lm = face_landmarks.landmark[idx]
                            x = int(lm.x * frame.shape[1])
                            y = int(lm.y * frame.shape[0])
                            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                        # Left eye EAR
                        left_top = face_landmarks.landmark[159]
                        left_bottom = face_landmarks.landmark[145]
                        left_left = face_landmarks.landmark[33]
                        left_right = face_landmarks.landmark[133]
                        left_vert_dist = euclidean((left_top.x, left_top.y), (left_bottom.x, left_bottom.y))
                        left_horiz_dist = euclidean((left_left.x, left_left.y), (left_right.x, left_right.y))
                        left_ear = left_vert_dist / left_horiz_dist if left_horiz_dist != 0 else 0

                        # Right eye EAR
                        right_top = face_landmarks.landmark[386]
                        right_bottom = face_landmarks.landmark[374]
                        right_left = face_landmarks.landmark[362]
                        right_right = face_landmarks.landmark[263]
                        right_vert_dist = euclidean((right_top.x, right_top.y), (right_bottom.x, right_bottom.y))
                        right_horiz_dist = euclidean((right_left.x, right_left.y), (right_right.x, right_right.y))
                        right_ear = right_vert_dist / right_horiz_dist if right_horiz_dist != 0 else 0

                        # Average EAR
                        avg_ear = (left_ear + right_ear) / 2

                        if avg_ear < BLINK_THRESHOLD and prev_eye_open:
                            blink_time = frame_number / fps
                            blink_times.append(blink_time)
                            prev_eye_open = False
                            cv2.putText(frame, "BLINK!", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)
                            emotions_timeline.append({"time": round(blink_time, 2), "emotion": "blink"})
                        elif avg_ear >= BLINK_THRESHOLD:
                            prev_eye_open = True   
                            out.write(frame)
                            frame_number += 1
                        else:
                            out.write(frame)
                            frame_number += 1
            cap.release()
            out.release()

        print(f"Processed video created: {processed_filepath}, size: {os.path.getsize(processed_filepath)} bytes")

        morse_code = ""
        morse_intervals = []
        if blink_times:
            # Calculate intervals between blinks
            intervals = [blink_times[i+1] - blink_times[i] for i in range(len(blink_times)-1)]
            morse_intervals = intervals  # Save for chatbot
            for interval in intervals:
                if interval <= 0.25:
                    morse_code += "."
                elif interval <= 1.2:
                    morse_code += "-"
                else:
                    morse_code += " "
        print(f"Morse code from blinks: {morse_code}")

        morse_chatbot_response = ""
        if morse_intervals:
            chatbot_url = request.url_root.rstrip('/') + '/api/chat'
            try:
                chat_prompt = (
                    f"Given these blink intervals (in seconds): {blink_times}, calculate potential Morse code. Provide 3 possible English translations or interpretations, explaining any ambiguity. Translation is done for prisoners of war, so make as many assumptions as necessary to an answer which makes sense (such as words like TORTURE or SOS). Be brief, limit of 20 words/explanation of a translation."
                )
                chat_response = requests.post(
                    chatbot_url,
                    json={"message": chat_prompt},
                )
                if chat_response.ok:
                    morse_chatbot_response = chat_response.json().get("response", "")
                else:
                    morse_chatbot_response = f"Chatbot error: {chat_response.text}"
            except Exception as e:
                morse_chatbot_response = f"Error: {e}"

        return render_template(
            "video_result.html",
            video_url=url_for('uploaded_file', filename=processed_filename),
            emotions=emotions_timeline,
            filename=filename,
            morse_code=morse_code,
            morse_chatbot_response=morse_chatbot_response
        )
    return render_template("video_upload.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve video files from uploads folder with proper MIME type and range support"""
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(file_path):
        return "File not found", 404

    file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    mime_types = {
        'mp4': 'video/mp4',
        'avi': 'video/x-msvideo',
        'mov': 'video/quicktime',
        'mkv': 'video/x-matroska',
        'flv': 'video/x-flv',
        'wmv': 'video/x-ms-wmv',
        'webm': 'video/webm'
    }
    
    mime_type = mime_types.get(file_ext, 'video/mp4')
    
    return send_from_directory(
        app.config['UPLOAD_FOLDER'], 
        filename, 
        as_attachment=False, 
        mimetype=mime_type,
        conditional=True  # Enable range requests for video streaming
    )

if __name__ == "__main__":
    app.run(debug=True, port=3000)
