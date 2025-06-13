import pandas as pd
import numpy as np

import os

def preprocess(dataset_folder: str, output_csv_path: str):
    obs_path = os.path.join(dataset_folder, "observations.csv")
    patients_path = os.path.join(dataset_folder, "patients.csv")

    df_obs = pd.read_csv(obs_path)
    df_patients = pd.read_csv(patients_path)

    codes_of_interest = ['39156-5', '8867-4', '8480-6', '2708-6']
    df_vitals = df_obs[df_obs['CODE'].isin(codes_of_interest)].copy()
    df_vitals['DATE'] = pd.to_datetime(df_vitals['DATE'])
    df_vitals.rename(columns={'PATIENT': 'patient_id', 'DATE': 'timestamp'}, inplace=True)

    df_vitals_pivot = df_vitals.pivot_table(
        index=['patient_id', 'timestamp'],
        columns='CODE',
        values='VALUE',
        aggfunc='first'
    ).reset_index()

    df_vitals_pivot.rename(columns={
        '39156-5': 'bmi',
        '8867-4': 'heart_rate',
        '8480-6': 'systolic_bp',
        '2708-6': 'spo2'
    }, inplace=True)

    df_patients = df_patients[['Id', 'BIRTHDATE']].copy()
    df_patients.rename(columns={'Id': 'patient_id'}, inplace=True)
    df_patients['BIRTHDATE'] = pd.to_datetime(df_patients['BIRTHDATE'])

    df_merged = pd.merge(df_vitals_pivot, df_patients, on='patient_id', how='left')
    df_merged['timestamp'] = df_merged['timestamp'].dt.tz_localize(None)
    df_merged['age'] = df_merged.apply(lambda row: (row['timestamp'] - row['BIRTHDATE']).days // 365, axis=1)

    # df_merged['heart_rate'] = df_merged['heart_rate'].fillna(75).astype(float)
    # df_merged['spo2'] = df_merged['spo2'].fillna(98).astype(float)
    # df_merged['systolic_bp'] = df_merged['systolic_bp'].fillna(120).astype(float)
    # df_merged['bmi'] = df_merged['bmi'].fillna(23).astype(float)
    df_merged['age'] = df_merged['age'].astype(int)

    # Define faixas fisiológicas realistas
    df_merged['heart_rate'] = df_merged['heart_rate'].apply(
        lambda x: np.random.randint(60, 90) if pd.isna(x) else x).astype(float)

    df_merged['spo2'] = df_merged['spo2'].apply(
        lambda x: np.random.randint(95, 100) if pd.isna(x) else x).astype(float)

    df_merged['systolic_bp'] = df_merged['systolic_bp'].apply(
        lambda x: np.random.randint(110, 130) if pd.isna(x) else x).astype(float)

    df_merged['bmi'] = df_merged['bmi'].apply(
        lambda x: np.random.uniform(21, 26) if pd.isna(x) else x).astype(float)

    df_merged.to_csv("processed_dataset.csv", index=False)

    def score_heart_rate(hr): return 3 if hr <= 40 or hr > 130 else 1 if hr <= 50 or hr > 110 else 0
    def score_spo2(spo2): return 3 if spo2 <= 91 else 2 if spo2 <= 93 else 1 if spo2 <= 95 else 0
    def score_systolic_bp(bp): return 3 if bp <= 90 or bp >= 220 else 2 if bp <= 100 else 1 if bp <= 110 else 0
    def score_age(age): return 1 if age >= 65 else 0
    def score_bmi(bmi): return 1 if bmi < 18.5 or bmi >= 35 else 0

    df_merged['score_hr'] = df_merged['heart_rate'].apply(score_heart_rate)
    df_merged['score_spo2'] = df_merged['spo2'].apply(score_spo2)
    df_merged['score_bp'] = df_merged['systolic_bp'].apply(score_systolic_bp)
    df_merged['score_age'] = df_merged['age'].apply(score_age)
    df_merged['score_bmi'] = df_merged['bmi'].apply(score_bmi)

    df_merged['ews_w'] = df_merged[
        ['score_hr', 'score_spo2', 'score_bp', 'score_age', 'score_bmi']
    ].sum(axis=1)
    df_merged['is_critical'] = df_merged[['score_hr', 'score_spo2', 'score_bp']].max(axis=1) == 3
    df_merged['ews_w'] = df_merged.apply(
        lambda row: max(row['ews_w'], 5) if row['is_critical'] else row['ews_w'], axis=1
    )

    df_result = df_merged[['patient_id', 'timestamp', 'ews_w']].sort_values(by=['patient_id', 'timestamp'])

    window_size = 7
    forecast_horizon = 3
    threshold = 3

    samples = []
    for patient_id, group in df_result.groupby('patient_id'):
        group = group.reset_index(drop=True)
        for i in range(len(group) - window_size - forecast_horizon + 1):
            window = group.iloc[i:i+window_size]
            future = group.iloc[i+window_size:i+window_size+forecast_horizon]
            label = int((future['ews_w'] >= threshold).any())
            sample = {
                'patient_id': patient_id,
                **{f'ews_w_day_{j+1}': window.iloc[j]['ews_w'] for j in range(window_size)},
                'risk_next_3_days': label
            }
            samples.append(sample)

    df_final = pd.DataFrame(samples)
    df_final.to_csv(output_csv_path, index=False)
    print(f"Arquivo salvo em: {output_csv_path}")
