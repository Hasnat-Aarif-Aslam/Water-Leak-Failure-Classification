# main.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, re
from collections import defaultdict, Counter

# ──────────────────────────────────────────────────────────────────────────────
# 1) APP CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Water Leak Classifier", layout="wide")
st.title("💧 Water Leak Failure Classification")

# ──────────────────────────────────────────────────────────────────────────────
# 2) LOAD MODEL & THRESHOLDS
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    clf = joblib.load("/mnt/d/TREASURY/ORDERS UNDER WORKING/icecarnage/Model/leak_classifier_pipeline.joblib")
    le  = joblib.load("/mnt/d/TREASURY/ORDERS UNDER WORKING/icecarnage/Model/label_encoder.joblib")
    df_thresh = pd.read_csv("/mnt/d/TREASURY/ORDERS UNDER WORKING/icecarnage/Interpretation/feature_thresholds.csv")
    return clf, le, df_thresh

clf, le, df_thresh = load_artifacts()

# Sidebar: human‑readable rules
st.sidebar.header("Decision Boundaries")
for _, row in df_thresh.iterrows():
    feat = row["feature"]
    if row["type"] == "continuous":
        low, high = row["25% quantile"], row["75% quantile"]
        st.sidebar.markdown(f"**{feat}**: low ≤{low}, med {low}–{high}, high >{high}")
    else:
        split = row["top split 1"]
        st.sidebar.markdown(f"**{feat}**: split == {split}")

# ──────────────────────────────────────────────────────────────────────────────
# 3) PREPROCESSING FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

numeric_feats = [
    'flow_rate_lps','total_flow_volume_m3','nighttime_flow_rate_lps',
    'flow_rate_change_per_min','pressure_psi','min_pressure_24h_psi',
    'leak_noise_score','acoustic_amplitude_db','vibration_frequency_hz',
    'zero_flow_duration_min','pipe_diameter_mm','pipe_age_years',
    'valve_position_percent','pump_flow_rate_lps'
]
binary_feats = ['unusual_usage_flag']
categorical_feats = ['pipe_material','valve_status','pump_state']
time_feats = ['weekday','month','day','hour_sin','hour_cos']
location_feats = ['coord_lat','coord_lon','inc_lat','inc_lon']


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # parse dates
    df['timestamp']   = pd.to_datetime(df['timestamp'])
    df['repair_date'] = pd.to_datetime(df.get('repair_date', None), errors='coerce')

    # split coords
    df[['coord_lat','coord_lon']] = (
        df['coordinates_latlon'].str.split(',', expand=True).astype(float)
    )
    df[['inc_lat','inc_lon']] = (
        df['incident_latlon'].str.split(',', expand=True).astype(float)
    )

    # cap numeric outliers (IQR) using numeric_feats
    for col in numeric_feats:
        Q1, Q3 = df[col].quantile([.25, .75])
        IQR    = Q3 - Q1
        df[col] = df[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)

    # engineer time features
    df['weekday']  = df['timestamp'].dt.weekday
    df['month']    = df['timestamp'].dt.month
    df['day']      = df['timestamp'].dt.day
    df['hour']     = df['timestamp'].dt.hour
    df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
    df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)

    # drop unused
    drop_cols = [
        'meter_id','pipe_id','customer_id',
        'repair_date','repair_duration_hr',
        'bill_amount_usd','monthly_usage_gal',
        'coordinates_latlon','incident_latlon',
        'timestamp','hour',
        'usage_baseline_gal','consumption_gal','failure_type'
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # cast categoricals
    for c in categorical_feats:
        if c in df.columns:
            df[c] = df[c].astype('category')

    # IMPUTE
    # median for numeric, time, location
    for col in numeric_feats + time_feats + location_feats + binary_feats:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)

    # mode for categorical
    for col in categorical_feats:
        df[col].fillna(df[col].mode()[0], inplace=True)

    return df

# ──────────────────────────────────────────────────────────────────────────────
# 4) FILE UPLOAD & PROCESSING
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("## Upload New Data")
uploaded = st.file_uploader("CSV or Excel (minus target column)", type=["csv","xlsx"])
if uploaded:
    try:
        df_new = pd.read_csv(uploaded)
    except:
        df_new = pd.read_excel(uploaded)

    st.write(f"Loaded **{len(df_new)}** samples.")
    st.dataframe(df_new.head(5), use_container_width=True, height=200)

    # preprocess
    df_proc = preprocess(df_new)
    st.markdown("### Preprocessed Sample")
    st.dataframe(df_proc.head(5), use_container_width=True, height=200)

    # predict
    preds_num = clf.predict(df_proc)
    preds_str = le.inverse_transform(preds_num)
    df_proc['predicted_failure_type'] = preds_str

    st.markdown("## 2. Predictions")
    st.dataframe(df_proc.head(10), use_container_width=True, height=300)

    csv_out = df_proc.to_csv(index=False).encode('utf-8')
    st.download_button("Download All Predictions", csv_out, "predictions.csv", "text/csv")

else:
    st.info("Please upload your data to get predictions.")

# ──────────────────────────────────────────────────────────────────────────────
# 5) FOOTER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("Built with XGBoost")



