import streamlit as st
import pandas as pd
import joblib
import requests
import numpy as np
from datetime import datetime

# ==========================
#  PRÉDICTION EN TEMPS RÉEL (API AviationStack)
# ==========================

API_KEY = "3057f8d1b6d9283ee6a287082a3388af"
JFK_IATA = "JFK"  # Code IATA pour filtrer les vols au départ de JFK

# Charger le modèle entraîné
MODEL_PATH = "best_flight_delay_model2.pkl"

try:
    pipeline = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error(f"🚨 Fichier modèle introuvable : {MODEL_PATH}. Vérifie son emplacement !")
    st.stop()

st.title("✈️ Prédiction des Retards de Vols 🚀")

# 🛫 **1. Récupération des vols en temps réel**
def fetch_live_flights():
    url = f"http://api.aviationstack.com/v1/flights?access_key={API_KEY}&dep_iata={JFK_IATA}&flight_status=active"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get("data", [])
    else:
        st.error(f"⚠️ Erreur API : {response.status_code}")
        return []

# 🔄 **2. Transformation des données en format utilisable**
def prepare_live_data(flight_data):
    if not flight_data:
        return None

    df_live = pd.DataFrame(flight_data)

    # Extraction des informations utiles
    df_live['FL_DATE'] = pd.to_datetime(df_live['flight_date'])
    df_live['YEAR'] = df_live['FL_DATE'].dt.year
    df_live['MONTH'] = df_live['FL_DATE'].dt.month
    df_live['DAY'] = df_live['FL_DATE'].dt.day
    df_live['DAY_OF_WEEK'] = df_live['FL_DATE'].dt.dayofweek
    df_live['IS_WEEKEND'] = df_live['DAY_OF_WEEK'].apply(lambda x: 1 if x in [5, 6] else 0)

    df_live['SEASON'] = df_live['MONTH'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
    })

    df_live['AIRLINE_CODE'] = df_live['airline'].apply(lambda x: x.get('iata', 'UNKNOWN'))
    df_live['DEST'] = df_live['arrival'].apply(lambda x: x.get('iata', 'UNKNOWN'))
    df_live['DEP_DELAY'] = df_live['departure'].apply(lambda x: x.get('delay', 0))
    df_live['HOUR'] = pd.to_datetime(df_live['departure'].apply(lambda x: x.get('scheduled', ''))).dt.hour

    def convert_to_hhmm(timestamp):
        if pd.isna(timestamp) or timestamp == "":
            return np.nan
        dt = pd.to_datetime(timestamp, errors='coerce')
        return dt.hour * 100 + dt.minute if not pd.isna(dt) else np.nan

    df_live['CRS_DEP_TIME'] = df_live['departure'].apply(lambda x: convert_to_hhmm(x.get('scheduled', '')))
    df_live['DEP_TIME'] = df_live['departure'].apply(lambda x: convert_to_hhmm(x.get('actual', x.get('estimated', x.get('scheduled', 0)))))
    df_live['WHEELS_OFF'] = df_live['departure'].apply(lambda x: convert_to_hhmm(x.get('actual_runway', x.get('estimated_runway', x.get('estimated', x.get('scheduled', 0))))))
    df_live['WHEELS_ON'] = df_live['arrival'].apply(lambda x: convert_to_hhmm(x.get('actual_runway', x.get('estimated_runway', x.get('estimated', x.get('scheduled', 0))))))

    def calculate_duration(start, end):
        if pd.isna(start) or pd.isna(end) or start == "" or end == "":
            return np.nan
        return (pd.to_datetime(end) - pd.to_datetime(start)).seconds // 60

    df_live['CRS_ELAPSED_TIME'] = df_live.apply(lambda x: calculate_duration(x['departure'].get('scheduled', ''), x['arrival'].get('scheduled', '')), axis=1)
    df_live['ELAPSED_TIME'] = df_live.apply(lambda x: calculate_duration(x['departure'].get('actual', x['departure'].get('estimated', x['departure'].get('scheduled', ''))),
                                                                          x['arrival'].get('actual', x['arrival'].get('estimated', x['arrival'].get('scheduled', '')))), axis=1)

    delay_keys = ['DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER', 'DELAY_DUE_NAS', 'DELAY_DUE_SECURITY', 'DELAY_DUE_LATE_AIRCRAFT']
    for key in delay_keys:
        df_live[key] = 0

    # ✅ Vérifier colonnes attendues
    expected_columns = list(pipeline.named_steps['preprocessor'].feature_names_in_)
    missing_columns = [col for col in expected_columns if col not in df_live.columns]

    if missing_columns:
        st.error(f"🚨 Colonnes manquantes : {missing_columns}")
        st.stop()

    df_live.fillna(0, inplace=True)

    return df_live[expected_columns]  

# 🚀 **3. Interface Streamlit**
if st.button("🔄 Récupérer les vols en temps réel"):
    live_flights = fetch_live_flights()
    
    if live_flights:
        live_data = prepare_live_data(live_flights)
        
        if live_data is not None:
            predictions = pipeline.predict(live_data)

            for i, flight in enumerate(live_flights):
                statut = "🟢 À l'heure" if predictions[i] == 0 else "🔴 En retard" if predictions[i] > 0 else "🟡 En avance"

                st.subheader(f"✈️ {flight['flight']['iata']} vers {flight['arrival']['iata']}")
                st.write(f"🏢 Aéroport d'arrivée : {flight['arrival']['airport']}")
                st.write(f"🕒 Départ prévu : {flight['departure']['scheduled']}")
                st.write(f"⏳ Prédiction de retard : **{predictions[i]:.2f} min**")
                st.write(f"📌 **Statut : {statut}**")

        else:
            st.warning("⚠️ Données invalides après transformation.")
    else:
        st.warning("⚠️ Aucun vol récupéré, vérifiez l'API.")
