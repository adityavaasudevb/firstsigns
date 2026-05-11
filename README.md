# FirstSigns
### AI-Based Behavioral Monitoring System

FirstSigns is an AI-assisted behavioral monitoring prototype that analyzes uploaded child behavior videos, extracts session-level behavioral metrics, and visualizes changes across sessions. It is designed as a screening support tool, not a diagnostic system, and won second place at ISB HackFest 2026.

## How It Works

- Video upload
- OpenCV + MediaPipe extraction
- 5 behavioral metrics
- Z-score deviation analysis
- Isolation Forest anomaly detection
- Longitudinal trend visualization

## Tech Stack

Python, Streamlit, OpenCV, MediaPipe, scikit-learn, SQLite

## Run Locally

```powershell
.\venv\Scripts\Activate.ps1
streamlit run app.py
```

## Metrics

- Engagement: Composite score summarizing behavioral consistency, face presence, gaze, gesture activity, and movement stability.
- Face Presence: Proportion of processed frames where a face is detected.
- Gaze Score: Estimate of visual attention based on face landmarks and eye openness.
- Gesture Score: Estimate of repeated hand movement activity detected across frames.
- Spike Density: Frequency of notable movement spikes in face position over time.

## Disclaimer

This system is not a diagnostic tool. It is designed as a behavioral screening support prototype.
