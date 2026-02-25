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
# 2. Chargement des données 
# ---------------------------------------------------------
@st.cache_data
def load_data():
    # Chargement du dataset depuis le CSV.
    # Remarque pédagogique : on utilise ici l'ensemble des données disponibles
    # après filtrage par genre (on ne resample plus pour équilibrer), afin
    # de conserver la richesse et la variabilité du dataset réel.
    df = pd.read_csv("dataset.csv")

    # Renommer la colonne 'track_genre' en 'genre' si nécessaire
    if 'track_genre' in df.columns:
        df = df.rename(columns={'track_genre': 'genre'})

    # Filtrer pour ne garder que les genres demandés par le projet
    genres_autorises = ['rap', 'jazz', 'house', 'pop']
    df = df[df['genre'].isin(genres_autorises)]

    # Nettoyage : retirer les lignes sans titre ou sans artiste
    df = df.dropna(subset=['track_name', 'artists'])

    # Supprimer les doublons exacts (même titre + mêmes artistes)
    df = df.drop_duplicates(subset=['track_name', 'artists'])

    # Nous utilisons l'ensemble filtré sans rééchantillonnage ici. Cela signifie
    # que la fréquence des genres dans le dataset reflète la réalité des données
    # d'origine (ce qui peut être souhaitable pour certaines recommandations).

    # Réinitialiser l'index pour plus de propreté
    df = df.reset_index(drop=True)

    return df

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
    
    # Onglets pour la sélection de l'humeur (Quick Mood vs Custom DJ Mode)
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
    # Pour le modèle on utilisera la loudness normalisée (0..1) afin que
    # toutes les features aient des échelles comparables.
    features_for_model = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'speechiness', 'valence', 'loudness_normalized']
    X = df_filtered[features_for_model]

    # 7. Normalisation et KNN
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ideal_scaled = scaler.transform(ideal_vector)

    # Nous cherchons jusqu'à 20 voisins afin d'avoir un pool plus large de candidats
    # puis nous filtrons les doublons pour n'afficher que 5 morceaux uniques.
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
        
        # Si ce morceau a déjà été affiché (même titre + même artiste), on l'ignore
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
