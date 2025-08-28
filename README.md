# AI Co‑Pilot for Surgeons — **Polished Round‑1 POC**

This upgraded project is **judge‑ready**:
- Two **sample videos** (choose in sidebar).
- **Split‑screen**: Original vs AI overlay.
- **Smart Highlighter** and **Predictive Co‑Pilot** modes.
- **Live metrics**: FPS and per‑frame latency estimate.
- **Record Demo** to MP4 and **Export Report** (screenshots + JSON summary).

## Quick Start
```bash
python -m venv .venv && .venv\Scripts\activate   # Windows
# or: source .venv/bin/activate                     # mac/linux
pip install -r requirements.txt
streamlit run app.py
```

## Round‑2 Path
Swap the heuristics for real models (UNet/YOLO) but keep this exact UI.
