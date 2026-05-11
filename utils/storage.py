import os
import sqlite3
import uuid
from datetime import datetime


DB_PATH = os.path.join("data", "firstsigns.db")


def init_storage():
    os.makedirs("data", exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                child_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                engagement REAL NOT NULL,
                gaze_score REAL NOT NULL,
                gesture_score REAL NOT NULL,
                spike_density REAL NOT NULL,
                face_presence REAL NOT NULL
            )
            """
        )


def add_session(child_id, session_data):
    init_storage()

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO sessions (
                child_id,
                session_id,
                timestamp,
                engagement,
                gaze_score,
                gesture_score,
                spike_density,
                face_presence
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                child_id,
                str(uuid.uuid4()),
                str(datetime.now()),
                float(session_data["engagement"]),
                float(session_data["gaze_score"]),
                float(session_data["gesture_score"]),
                float(session_data["spike_density"]),
                float(session_data["face_presence"]),
            ),
        )


def get_sessions(child_id):
    init_storage()

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT
                session_id,
                timestamp,
                engagement,
                gaze_score,
                gesture_score,
                spike_density,
                face_presence
            FROM sessions
            WHERE child_id = ?
            ORDER BY timestamp, rowid
            """,
            (child_id,),
        ).fetchall()

    sessions = []

    for row in rows:
        feature_vector = [
            float(row["engagement"]),
            float(row["gaze_score"]),
            float(row["gesture_score"]),
            float(row["spike_density"]),
            float(row["face_presence"]),
        ]

        sessions.append(
            {
                "session_id": row["session_id"],
                "time": row["timestamp"],
                "engagement": feature_vector[0],
                "gaze_score": feature_vector[1],
                "gesture_score": feature_vector[2],
                "spike_density": feature_vector[3],
                "face_presence": feature_vector[4],
                "features": feature_vector,
            }
        )

    return sessions
