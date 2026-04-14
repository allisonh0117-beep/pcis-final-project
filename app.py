
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.units import inch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import datetime
import tempfile
import os
import urllib.request

# Page config
st.set_page_config(
    page_title="NosillaRisk",
    page_icon="🐹",
    layout="centered"
)

# Download model once
MODEL_PATH = "pose_landmarker.task"
if not os.path.exists(MODEL_PATH):
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    urllib.request.urlretrieve(url, MODEL_PATH)

# Setup detector
@st.cache_resource
def load_detector():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(base_options=base_options)
    return vision.PoseLandmarker.create_from_options(options)

CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),(23,25),(24,26),
    (25,27),(26,28),(27,29),(28,30),(29,31),(30,32)
]

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine = np.clip(cosine, -1.0, 1.0)
    return round(np.degrees(np.arccos(cosine)), 1)

def get_midpoint(lm, idx1, idx2, w, h):
    return [(lm[idx1].x + lm[idx2].x) / 2 * w,
            (lm[idx1].y + lm[idx2].y) / 2 * h]

def get_rula_score(joint, angle):
    flexion = 180 - angle
    if joint == "spine":
        if flexion < 20: return 0, "LOW"
        elif flexion < 40: return 1, "MEDIUM"
        else: return 2, "HIGH"
    elif joint == "knee":
        if flexion < 30: return 0, "LOW"
        elif flexion < 60: return 1, "MEDIUM"
        else: return 2, "HIGH"
    elif joint == "hip":
        if flexion < 20: return 0, "LOW"
        elif flexion < 45: return 1, "MEDIUM"
        else: return 2, "HIGH"
    return 0, "LOW"

RISK_COLORS = {0: (0,255,0), 1: (0,255,255), 2: (0,0,255)}

def draw_risk_map(worst_scores):
    fig, ax = plt.subplots(1, 1, figsize=(5, 9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 18)
    ax.axis("off")
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")
    risk_colors = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}
    label_colors = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
    spine_col = risk_colors[worst_scores["spine"]]
    lknee_col = risk_colors[worst_scores["left_knee"]]
    rknee_col = risk_colors[worst_scores["right_knee"]]
    lhip_col  = risk_colors[worst_scores["left_hip"]]
    rhip_col  = risk_colors[worst_scores["right_hip"]]
    ax.add_patch(plt.Circle((5, 16), 1.2, color="#95a5a6", zorder=3))
    ax.add_patch(patches.FancyBboxPatch((4.6,14.5),0.8,1.2,boxstyle="round,pad=0.1",color="#95a5a6",zorder=3))
    ax.add_patch(patches.FancyBboxPatch((3.2,10.5),3.6,4.0,boxstyle="round,pad=0.1",color=spine_col,zorder=3))
    ax.add_patch(patches.FancyBboxPatch((3.2,8.8),1.6,1.8,boxstyle="round,pad=0.1",color=lhip_col,zorder=3))
    ax.add_patch(patches.FancyBboxPatch((5.2,8.8),1.6,1.8,boxstyle="round,pad=0.1",color=rhip_col,zorder=3))
    ax.add_patch(patches.FancyBboxPatch((3.3,6.0),1.4,2.8,boxstyle="round,pad=0.1",color=lhip_col,zorder=3))
    ax.add_patch(patches.FancyBboxPatch((5.3,6.0),1.4,2.8,boxstyle="round,pad=0.1",color=rhip_col,zorder=3))
    ax.add_patch(plt.Circle((4.0,5.7),0.8,color=lknee_col,zorder=4))
    ax.add_patch(plt.Circle((6.0,5.7),0.8,color=rknee_col,zorder=4))
    ax.add_patch(patches.FancyBboxPatch((3.3,3.0),1.4,2.7,boxstyle="round,pad=0.1",color=lknee_col,zorder=3))
    ax.add_patch(patches.FancyBboxPatch((5.3,3.0),1.4,2.7,boxstyle="round,pad=0.1",color=rknee_col,zorder=3))
    ax.add_patch(patches.FancyBboxPatch((3.0,2.2),2.0,0.8,boxstyle="round,pad=0.1",color="#95a5a6",zorder=3))
    ax.add_patch(patches.FancyBboxPatch((5.0,2.2),2.0,0.8,boxstyle="round,pad=0.1",color="#95a5a6",zorder=3))
    ax.text(5,17.5,"RISK MAP",ha="center",fontsize=13,fontweight="bold",color="white")
    for x,y,text,color in [
        (1.5,12.5,f"Spine\n{label_colors[worst_scores['spine']]}",spine_col),
        (1.0,9.5,f"L.Hip\n{label_colors[worst_scores['left_hip']]}",lhip_col),
        (8.5,9.5,f"R.Hip\n{label_colors[worst_scores['right_hip']]}",rhip_col),
        (1.0,5.7,f"L.Knee\n{label_colors[worst_scores['left_knee']]}",lknee_col),
        (8.5,5.7,f"R.Knee\n{label_colors[worst_scores['right_knee']]}",rknee_col),
    ]:
        ax.text(x,y,text,ha="center",fontsize=8,fontweight="bold",color=color)
    plt.tight_layout()
    path = "risk_map.png"
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    return path

def process_video(video_path, detector):
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = max(int(cap.get(cv2.CAP_PROP_FPS)), 1)
    out_path = "output_rula.mp4"
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    angle_data = {"frame":[],"left_knee":[],"right_knee":[],"left_hip":[],"right_hip":[],"spine":[]}
    worst_scores = {"spine":0,"left_knee":0,"right_knee":0,"left_hip":0,"right_hip":0}
    frame_count = 0
    progress = st.progress(0, text="Analyzing video...")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)
        if result.pose_landmarks:
            lm = result.pose_landmarks[0]
            h, w = frame.shape[:2]
            def gp(idx): return [lm[idx].x*w, lm[idx].y*h]
            sm = get_midpoint(lm,11,12,w,h)
            hm = get_midpoint(lm,23,24,w,h)
            km = get_midpoint(lm,25,26,w,h)
            lk = calculate_angle(gp(23),gp(25),gp(27))
            rk = calculate_angle(gp(24),gp(26),gp(28))
            lh = calculate_angle(gp(11),gp(23),gp(25))
            rh = calculate_angle(gp(12),gp(24),gp(26))
            sp = calculate_angle(sm,hm,km)
            angle_data["frame"].append(frame_count)
            angle_data["left_knee"].append(lk)
            angle_data["right_knee"].append(rk)
            angle_data["left_hip"].append(lh)
            angle_data["right_hip"].append(rh)
            angle_data["spine"].append(sp)
            ss,_ = get_rula_score("spine",sp)
            ls,_ = get_rula_score("knee",lk)
            rs,_ = get_rula_score("knee",rk)
            lhs,_ = get_rula_score("hip",lh)
            rhs,_ = get_rula_score("hip",rh)
            worst_scores["spine"]      = max(worst_scores["spine"],ss)
            worst_scores["left_knee"]  = max(worst_scores["left_knee"],ls)
            worst_scores["right_knee"] = max(worst_scores["right_knee"],rs)
            worst_scores["left_hip"]   = max(worst_scores["left_hip"],lhs)
            worst_scores["right_hip"]  = max(worst_scores["right_hip"],rhs)
            sc = RISK_COLORS[ss]
            for a,b in CONNECTIONS:
                ax2,ay2 = int(lm[a].x*w),int(lm[a].y*h)
                bx2,by2 = int(lm[b].x*w),int(lm[b].y*h)
                cv2.line(frame,(ax2,ay2),(bx2,by2),sc,3)
            for l in lm:
                cx,cy = int(l.x*w),int(l.y*h)
                cv2.circle(frame,(cx,cy),6,(255,255,255),-1)
            cv2.rectangle(frame,(0,0),(300,165),(0,0,0),-1)
            for i,(label,color) in enumerate([
                (f"Spine: {sp} [{['LOW','MEDIUM','HIGH'][ss]}]", sc),
                (f"L.Knee: {lk}", RISK_COLORS[ls]),
                (f"R.Knee: {rk}", RISK_COLORS[rs]),
                (f"L.Hip: {lh}", RISK_COLORS[lhs]),
                (f"R.Hip: {rh}", RISK_COLORS[rhs]),
            ]):
                cv2.putText(frame,label,(10,28+i*28),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
        out.write(frame)
        frame_count += 1
        if total_frames > 0:
            progress.progress(min(frame_count/total_frames, 1.0), text=f"Analyzing... {frame_count}/{total_frames} frames")
    cap.release()
    out.release()
    progress.progress(1.0, text="Analysis complete!")
    return angle_data, worst_scores, out_path

# ── UI ──────────────────────────────────────────────
st.title("🐹 NosillaRisk")
st.subheader("Online Biomechanics Risk Analyzer")
st.write("Upload a video of someone performing a physical task to get a full ergonomic risk report!")

uploaded_file = st.file_uploader("Upload video", type=["mp4","mov","avi"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.video(uploaded_file)
    
    if st.button("Analyze", type="primary"):
        detector = load_detector()
        angle_data, worst_scores, out_path = process_video(tmp_path, detector)

        st.success("Analysis complete!")

        # Risk summary
        st.subheader("Joint Risk Summary")
        risk_labels = {0:"LOW", 1:"MEDIUM", 2:"HIGH"}
        risk_emoji  = {0:"🟢", 1:"🟡", 2:"🔴"}
        cols = st.columns(5)
        joints = [("Spine","spine"),("L.Knee","left_knee"),
                  ("R.Knee","right_knee"),("L.Hip","left_hip"),("R.Hip","right_hip")]
        for col,(name,key) in zip(cols,joints):
            score = worst_scores[key]
            col.metric(name, f"{risk_emoji[score]} {risk_labels[score]}")

        # Side by side: video + risk map
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Analyzed Video")
            with open(out_path,"rb") as f:
                st.download_button("Download annotated video", f, "output_rula.mp4")

        with col2:
            st.subheader("Risk Map")
            map_path = draw_risk_map(worst_scores)
            st.image(map_path, width=220)

        # Angle details
        st.subheader("Peak Joint Angles")
        angle_table = {
            "Joint": ["Spine","Left Knee","Right Knee","Left Hip","Right Hip"],
            "Min Angle": [
                f"{min(angle_data['spine'])}°",
                f"{min(angle_data['left_knee'])}°",
                f"{min(angle_data['right_knee'])}°",
                f"{min(angle_data['left_hip'])}°",
                f"{min(angle_data['right_hip'])}°",
            ],
            "Risk": [
                f"{risk_emoji[worst_scores['spine']]} {risk_labels[worst_scores['spine']]}",
                f"{risk_emoji[worst_scores['left_knee']]} {risk_labels[worst_scores['left_knee']]}",
                f"{risk_emoji[worst_scores['right_knee']]} {risk_labels[worst_scores['right_knee']]}",
                f"{risk_emoji[worst_scores['left_hip']]} {risk_labels[worst_scores['left_hip']]}",
                f"{risk_emoji[worst_scores['right_hip']]} {risk_labels[worst_scores['right_hip']]}",
            ]
        }
        import pandas as pd
        st.table(pd.DataFrame(angle_table))

        # PDF download
        st.subheader("Download Report")
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        import datetime

        doc = SimpleDocTemplate("report.pdf", pagesize=letter,
                                rightMargin=50, leftMargin=50,
                                topMargin=50, bottomMargin=50)
        styles = getSampleStyleSheet()
        story = []
        story.append(Paragraph("ErgoVision — Biomechanics Risk Report", styles["Title"]))
        story.append(Paragraph(f"Generated: {datetime.datetime.now().strftime('%B %d, %Y')}", styles["Normal"]))
        story.append(Spacer(1, 0.2*inch))
        table_data = [["Joint","Peak Angle","Risk"]]
        for name, key, jtype in [("Spine","spine","spine"),("Left Knee","left_knee","knee"),
                                   ("Right Knee","right_knee","knee"),("Left Hip","left_hip","hip"),
                                   ("Right Hip","right_hip","hip")]:
            peak = min(angle_data[key])
            score, level = get_rula_score(jtype, peak)
            table_data.append([name, f"{peak}°", level])
        t = Table(table_data, colWidths=[2*inch,2*inch,2*inch])
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#2c3e50")),
            ("TEXTCOLOR",(0,0),(-1,0),colors.white),
            ("ALIGN",(0,0),(-1,-1),"CENTER"),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("GRID",(0,0),(-1,-1),0.5,colors.grey),
            ("TOPPADDING",(0,0),(-1,-1),8),
            ("BOTTOMPADDING",(0,0),(-1,-1),8),
        ]))
        story.append(t)
        story.append(Spacer(1,0.2*inch))
        story.append(Image(map_path, width=2*inch, height=3.5*inch))
        doc.build(story)
        with open("report.pdf","rb") as f:
            st.download_button("Download PDF Report", f, "ergonomic_report.pdf",
                               type="primary")
