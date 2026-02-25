import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import numpy as np

# 1. Configuration de la page
st.set_page_config(page_title="Moodify - Recommandateur", page_icon="🎧", layout="wide")

with st.sidebar:
    st.header("À propos")
    st.write("J'ai développé cette application en tant que projet personnel en **MAM3 à Polytech Nice Sophia**.")
    st.info("💡 **Concept :** L'IA traduit votre humeur en variables mathématiques pour trouver la musique parfaite.")
    st.write("---")
    st.write("👤 **Créé par :** Firas Ghedir")
    st.write("📧 **Contact :** firasghedir3@gmail.com")
    st.write("🔗 **LinkedIn :** [Mon Profil](https://www.linkedin.com/in/firas-ghedir-421510213)")

st.title("🎧 Moodify : La musique selon ton humeur")
st.markdown("Dis-moi ce que tu ressens et ce que tu aimes, je m'occupe du reste !")

# ---------------------------------------------------------
# 2. Chargement des données (C'EST ICI QUE J'AI MIS TON CODE)
# ---------------------------------------------------------
@st.cache_data
def load_data():
    # On charge le dataset téléchargé depuis Kaggle
    df = pd.read_csv("dataset.csv") # Renomme bien le fichier téléchargé en "dataset.csv"
    
    # 1. On renomme la colonne genre pour la rendre plus simple
    if 'track_genre' in df.columns:
        df = df.rename(columns={'track_genre': 'genre'})
    
    # 2. On filtre STRICTEMENT les 4 classes de ton cahier des charges
    genres_autorises = ['rap', 'jazz', 'house', 'pop']
    df = df[df['genre'].isin(genres_autorises)]
    
    # 3. Nettoyage des données manquantes et des doublons
    df = df.dropna(subset=['track_name', 'artists'])
    df = df.drop_duplicates(subset=['track_name', 'artists'])
    
    # 4. ÉQUILIBRAGE DU DATASET (La consigne du cahier des charges !)
    n_samples = 500 
    
    # On groupe par genre, on prend 500 chansons au hasard pour chaque, et on rassemble le tout
    # Attention: si un genre a moins de 500 chansons, ça plantera. On utilise replace=True au cas où.
    df_balanced = df.groupby('genre').sample(n=n_samples, random_state=42, replace=True)
    
    # 5. On réinitialise l'index proprement
    df_balanced = df_balanced.reset_index(drop=True)
    
    return df_balanced

try:
    df = load_data()
except FileNotFoundError:
    st.error("🚨 Fichier 'dataset.csv' introuvable. N'oublie pas de le mettre dans le même dossier que ce script !")
    st.stop()

has_genres = True # Puisqu'on filtre par genre dans la fonction, on sait que la colonne existe

# 3. Les Caractéristiques (Features) tirées de ton cahier des charges
features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'speechiness', 'valence', 'loudness']

missing_features = [f for f in features if f not in df.columns]
if missing_features:
    st.error(f"Il manque ces colonnes dans ton dataset : {missing_features}")
    st.stop()

# Normaliser la loudness pour être entre 0 et 1
# Loudness typical range: -60 to 0 dB
df['loudness_normalized'] = (df['loudness'] + 60) / 60
df['loudness_normalized'] = df['loudness_normalized'].clip(0, 1)

# 4. Interface Utilisateur : Choix de l'humeur et des goûts
col1, col2 = st.columns(2)

with col1:
    st.subheader("1️⃣ Tes goûts musicaux")
    available_genres = df['genre'].unique()
    selected_genres = st.multiselect("Choisis tes genres :", available_genres, default=available_genres)

with col2:
    st.subheader("2️⃣ Ton humeur actuelle")
    
    # Tabs for mood selection
    tab1, tab2 = st.tabs(["🎯 Quick Mood", "🎛️ Custom DJ Mode"])
    
    with tab1:
        mood_choice = st.radio(
            "Comment te sens-tu ?",
            ["🔥 Sport / Motivation", "🧠 Concentration / Étude", "🌧️ Triste / Calme", "🎉 Fête / Joie"],
            key="mood_radio"
        )
        use_custom_mood = False
    
    with tab2:
        enable_custom_dj = st.checkbox('Activer le mode Custom DJ personnalisé', value=False)
        
        if enable_custom_dj:
            st.write("Ajuste les curseurs pour personnaliser ton profil musical")
            custom_mood_values = {}
            for feature in features:
                if feature == 'loudness_normalized':
                    custom_mood_values[feature] = st.slider(
                        f"🔊 {feature.replace('_', ' ').title()}",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.5,
                        step=0.05,
                        help="Normalisé entre 0 (très silencieux) et 1 (très fort)"
                    )
                else:
                    custom_mood_values[feature] = st.slider(
                        f"🎵 {feature.title()}",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.5,
                        step=0.05
                    )
        else:
            st.info("Cochez la case ci-dessus pour activer les curseurs personnalisés")
            custom_mood_values = None

st.write("---")

# 5. Création du profil mathématique de l'humeur
# Ordre des features : ['acousticness', 'danceability', 'energy', 'instrumentalness', 'speechiness', 'valence', 'loudness_normalized']
mood_profiles = {
    "🔥 Sport / Motivation":    [0.05, 0.80, 0.90, 0.00, 0.10, 0.85, 0.85],
    "🧠 Concentration / Étude": [0.60, 0.30, 0.20, 0.85, 0.05, 0.45, 0.35],
    "🌧️ Triste / Calme":        [0.85, 0.40, 0.15, 0.10, 0.05, 0.20, 0.25],
    "🎉 Fête / Joie":           [0.10, 0.90, 0.85, 0.00, 0.15, 0.90, 0.80] 
}

# Determine which mood profile to use - this will be calculated later after the button click
# since we need to know the state of the checkbox at that time

if st.button("Trouver ma musique !"):
    # Determine which mood profile to use based on checkbox state
    if enable_custom_dj and custom_mood_values is not None:
        ideal_vector = np.array([custom_mood_values[f] for f in features]).reshape(1, -1)
    else:
        ideal_vector = np.array(mood_profiles[mood_choice]).reshape(1, -1)
    
    # 6. Filtrage du Dataset
    if selected_genres:
        df_filtered = df[df['genre'].isin(selected_genres)].copy()
    else:
        st.error("Sélectionne au moins un genre musical !")
        st.stop()
    
    if len(df_filtered) == 0:
        st.error("Aucune chanson trouvée avec les critères sélectionnés !")
        st.stop()
        
    df_filtered = df_filtered.reset_index(drop=True)
    # Use loudness_normalized instead of loudness for features
    features_for_model = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'speechiness', 'valence', 'loudness_normalized']
    X = df_filtered[features_for_model]

    # 7. Normalisation et KNN
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ideal_scaled = scaler.transform(ideal_vector)

    # Increase n_neighbors to 20 to find more candidates, then filter for duplicates
    n_neighbors = min(20, len(df_filtered))
    model = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree')
    model.fit(X_scaled)

    distances, indices = model.kneighbors(ideal_scaled)

    st.success(f"### 🎧 Voici les meilleures recommandations pour ton humeur !")
    
    # 8. Affichage des résultats avec lecteur Spotify - filter out duplicates
    seen_tracks = set()
    displayed_count = 0
    first_unique_idx = None
    
    for i in range(n_neighbors):
        if displayed_count >= 5:
            break
            
        idx = indices[0][i]
        song = df_filtered.iloc[idx]
        track_key = (song['track_name'], song['artists'])
        
        # Skip if we've already displayed this track
        if track_key in seen_tracks:
            continue
        
        seen_tracks.add(track_key)
        displayed_count += 1
        
        # Store the first unique track index for the radar chart
        if first_unique_idx is None:
            first_unique_idx = idx
        
        st.write(f"**{displayed_count}. {song['track_name']}** - {song['artists']} ({song['genre']})")
        
        # Display Spotify player if track_id exists
        if 'track_id' in df_filtered.columns and pd.notna(song['track_id']):
            spotify_url = f"https://open.spotify.com/embed/track/{song['track_id']}"
            st.components.v1.iframe(src=spotify_url, height=80, scrolling=False)

    # 9. Visualisation Radar Chart
    st.write("---")
    st.subheader("📊 Pourquoi ces choix ? (Analyse mathématique)")
    
    fig = go.Figure()
    
    # L'humeur idéale demandée
    fig.add_trace(go.Scatterpolar(
        r=ideal_vector[0],
        theta=features_for_model,
        fill='toself',
        name='Ton Humeur Idéale',
        line_color='magenta'
    ))
    
    # La top recommandation n°1 (première unique)
    if first_unique_idx is not None:
        fig.add_trace(go.Scatterpolar(
            r=df_filtered.loc[first_unique_idx, features_for_model].values,
            theta=features_for_model,
            fill='toself',
            name='Recommandation N°1',
            line_color='cyan'
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Comparaison : Ton Humeur vs La Chanson Trouvée"
    )

    st.plotly_chart(fig)