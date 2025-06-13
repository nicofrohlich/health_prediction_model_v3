import sqlite3
from app.model.predictor import calculate_news_w_row

DB_PATH = "data.db"


def init_db():
    import sqlite3
    conn = sqlite3.connect("data.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            timestamp TEXT,
            heart_rate REAL,
            spo2 REAL,
            systolic_bp REAL,
            bmi REAL,
            age INTEGER,
            ews_w INTEGER,
            media REAL,
            mediana REAL,
            desvio REAL,
            trend REAL,
            escore_max INTEGER,
            escore_min INTEGER
        )
    """)
    conn.commit()
    conn.close()


def insert_signal(data):

    # Calcular o score do novo registro
    current_score = calculate_news_w_row(
        data.heart_rate,
        data.spo2,
        data.systolic_bp,
        data.bmi,
        data.age
    )

    # Conectar ao banco e buscar os 6 registros anteriores do paciente
    conn = sqlite3.connect("data.db")
    c = conn.cursor()
    c.execute("""
        SELECT ews_w
        FROM signals
        WHERE patient_id = ?
        ORDER BY timestamp DESC
        LIMIT 6
    """, (data.patient_id,))
    rows = c.fetchall()
    conn.close()

    # Reconstruir janela de 7 escores
    previous_scores = [r[0] for r in rows[::-1]]
    full_scores = previous_scores + [current_score]
    while len(full_scores) < 7:
        full_scores.insert(0, full_scores[0])

    # Calcular features
    import numpy as np
    media = float(np.mean(full_scores))
    mediana = float(np.median(full_scores))
    desvio = float(np.std(full_scores))
    trend = float(full_scores[-1] - full_scores[0])
    escore_max = int(max(full_scores))
    escore_min = int(min(full_scores))

    # Inserir tudo na tabela
    conn = sqlite3.connect("data.db")
    c = conn.cursor()
    c.execute("""
        INSERT INTO signals (
            patient_id, timestamp, heart_rate, spo2, systolic_bp, bmi, age,
            ews_w, media, mediana, desvio, trend,
            escore_max, escore_min
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data.patient_id, data.timestamp.isoformat(),
        data.heart_rate, data.spo2, data.systolic_bp, data.bmi, data.age,
        current_score, media, mediana, desvio, trend,
        escore_max, escore_min
    ))
    conn.commit()
    conn.close()


def get_last_7_features(patient_id):

    conn = sqlite3.connect("data.db")
    c = conn.cursor()
    c.execute("""
        SELECT
            ews_w,
            media, mediana, desvio, trend,
            escore_max, escore_min
        FROM signals
        WHERE patient_id = ?
        ORDER BY timestamp DESC
        LIMIT 7
    """, (patient_id,))
    rows = c.fetchall()
    conn.close()

    if not rows:
        return None

    scores = [r[0] for r in rows[::-1]]  # ews_w
    while len(scores) < 7:
        scores.insert(0, scores[0])

    # Pega os valores das features derivadas mais recentes (último registro)
    latest = rows[0]
    derived = list(latest[1:])  # ignora ews_w, pega as features derivadas

    return scores + derived

