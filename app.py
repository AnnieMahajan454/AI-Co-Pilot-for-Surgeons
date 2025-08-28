import os, glob, time, json
import numpy as np
import cv2
import streamlit as st

st.set_page_config(page_title="AI Co‑Pilot for Surgeons", layout="wide")

def load_frames_from_dir(path):
    files = sorted(glob.glob(os.path.join(path, "*.png")))
    frames = []
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            frames.append(img)
    return frames

def colorize(gray): return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
def banner(img, text, color=(60,60,60)):
    h,w = img.shape[:2]
    cv2.rectangle(img,(0,0),(w,36),color,-1)
    cv2.putText(img,text,(10,24),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2,cv2.LINE_AA)

def segment_heuristics(gray):
    return cv2.inRange(gray,0,85), cv2.inRange(gray,200,255), cv2.inRange(gray,105,130)

def detect_anomaly(gray, prev, sens):
    blur = cv2.medianBlur(gray,5)
    diff = cv2.subtract(gray, blur)
    _, th = cv2.threshold(diff, int(40+60*(1-sens)),255,cv2.THRESH_BINARY)
    th = cv2.morphologyEx(th,cv2.MORPH_OPEN,np.ones((3,3),np.uint8),iterations=1)
    if prev is not None:
        d = cv2.absdiff(gray,prev)
        _, motion = cv2.threshold(d, int(15+40*(1-sens)),255,cv2.THRESH_BINARY)
        motion = cv2.morphologyEx(motion,cv2.MORPH_OPEN,np.ones((3,3),np.uint8),iterations=1)
        th = cv2.bitwise_or(th,motion)
    contours,_=cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    box=None
    if contours:
        c=max(contours,key=cv2.contourArea)
        if cv2.contourArea(c)>60: box=cv2.boundingRect(c)
    return box,th

def overlay_masks(bgr,lung,bone,liver,a=0.35):
    overlay=bgr.copy()
    overlay[lung>0]=(180,180,255); overlay[bone>0]=(220,180,220); overlay[liver>0]=(180,240,180)
    return cv2.addWeighted(overlay,a,bgr,1-a,0)

def hough_tool(gray):
    edges=cv2.Canny(gray,80,150)
    lines=cv2.HoughLinesP(edges,1,np.pi/180,60,minLineLength=60,maxLineGap=10)
    if lines is None: return None
    best=None;L=0
    for l in lines[:,0]:
        x1,y1,x2,y2=l;len_=np.hypot(x2-x1,y2-y1)
        if len_>L: L=len_;best=(x1,y1,x2,y2)
    return best

def point_line_distance(a,b,p):
    a,b,p=np.array(a,float),np.array(b,float),np.array(p,float)
    if np.allclose(a,b): return np.linalg.norm(p-a)
    return abs(np.cross(b-a,p-a))/np.linalg.norm(b-a)

def metrics_badge(bgr,fps,lat):
    cv2.putText(bgr,f"FPS:{fps:.1f} Lat:{lat:.0f}ms",(10,bgr.shape[0]-12),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2,cv2.LINE_AA)

st.sidebar.title("Surgical Co‑Pilot POC")
mode=st.sidebar.radio("Mode",["Smart Highlighter","Predictive Co‑Pilot"])
vid=st.sidebar.selectbox("Sample Video",["Video 1 — Anatomy + Tumor","Video 2 — Tool Drift + Shift"])
sens=st.sidebar.slider("Sensitivity",0.1,0.9,0.6,0.05)
thresh=st.sidebar.slider("Drift Threshold(px)",5,60,20,1)
fps_play=st.sidebar.slider("Playback FPS",6,24,12,1)
record=st.sidebar.checkbox("Record Demo")
export=st.sidebar.button("Export Report")

base=os.path.dirname(__file__)
vdir="video1_frames" if "Video 1" in vid else "video2_frames"
frames=load_frames_from_dir(os.path.join(base,"sample_data",vdir))
if not frames: st.stop()

st.markdown("# AI Co‑Pilot for Surgeons")
explain=st.empty()
if mode=="Smart Highlighter": explain.info("Highlights anatomy and flags anomalies in real-time.")
else: explain.info("Monitors tool path and nudges when drift detected.")

left,right=st.columns(2)
raw_pl=left.empty(); ai_pl=right.empty()

writer=None
if record:
    fourcc=cv2.VideoWriter_fourcc(*'mp4v')
    h,w=frames[0].shape;writer=cv2.VideoWriter(os.path.join(base,"sample_data","out.mp4"),fourcc,fps_play,(w*2,h))

events={"anomaly":0,"drift":0}; prev=None;lat_hist=[];ideal=((100,100),(540,380))
for gray in frames:
    t0=time.time();raw=colorize(gray);out=raw.copy()
    if mode=="Smart Highlighter":
        lung,bone,liver=segment_heuristics(gray)
        out=overlay_masks(out,lung,bone,liver)
        box,_=detect_anomaly(gray,prev,sens)
        if box: x,y,w,h=box;cv2.rectangle(out,(x,y),(x+w,y+h),(0,0,255),2);banner(out,"Anomaly!",(0,0,200));events["anomaly"]+=1
        else: banner(out,"Monitoring...",(60,60,60))
    else:
        a,b=ideal;cv2.line(out,a,b,(50,180,255),2);line=hough_tool(gray)
        if line: x1,y1,x2,y2=line;cv2.line(out,(x1,y1),(x2,y2),(0,255,255),3);mid=((x1+x2)//2,(y1+y2)//2);d=int(point_line_distance(a,b,mid))
        if line and d>thresh: banner(out,f"Drift {d}px",(0,165,255));cv2.arrowedLine(out,mid,(mid[0]-15,mid[1]-15),(0,165,255),3,tipLength=0.4);events["drift"]+=1
        elif line: banner(out,"On track",(60,60,60))
    prev=gray
    lat=(time.time()-t0)*1000;lat_hist.append(lat);fps=1000/np.mean(lat_hist[-min(10,len(lat_hist)):])
    metrics_badge(out,fps,lat)
    raw_pl.image(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB),channels="RGB",use_container_width=True)
    ai_pl.image(cv2.cvtColor(out,cv2.COLOR_BGR2RGB),channels="RGB",use_container_width=True)
    if writer is not None: writer.write(np.concatenate([raw,out],1))
    time.sleep(1.0/fps_play)
if writer is not None: writer.release(); st.sidebar.success("Saved output mp4")
if export:
    cv2.imwrite(os.path.join(base,"reports","snapshot.png"),out)
    with open(os.path.join(base,"reports","summary.json"),"w") as f: json.dump(events,f,indent=2)
    st.sidebar.success("Exported report")
