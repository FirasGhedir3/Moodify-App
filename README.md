# 🎧 Moodify : Le Recommandateur de Musique par IA

🌐 **[Testez l'application en direct ici !](https://moodify-app-ntw8uwxaob3tdu6cvxz8ex.streamlit.app/)**

## 📖 À propos du projet
**Moodify** est une application web interactive développée dans le cadre d'un projet personnel en **MAM3 à Polytech Nice Sophia**. 
L'objectif de ce projet est d'appliquer des concepts de Data Science et de Machine Learning pour recommander des musiques en fonction de l'humeur de l'utilisateur et de ses goûts musicaux, plutôt que par de simples similarités d'artistes.

## ✨ Fonctionnalités
* **Recherche par Humeur :** Choix entre 4 ambiances prédéfinies (Sport, Concentration, Triste/Calme, Fête) qui génèrent un profil mathématique idéal.
* **Filtre par Genres :** Limite les recommandations aux genres sélectionnés (Rap, Jazz, House, Pop).
* **Mode "Custom DJ" :** Personnalisation totale du profil de recherche grâce à 7 curseurs interactifs (Énergie, Valence, Volume, Acoustique, etc.).
* **Lecteur Spotify Intégré :** Écoute des extraits de 30 secondes directement depuis l'application.
* **Explicabilité de l'IA (Radar Chart) :** Visualisation graphique avec Plotly comparant l'humeur demandée avec les caractéristiques réelles de la chanson recommandée.

## 🧠 Fonctionnement Technique (L'IA)
L'application repose sur un algorithme de **K-Plus Proches Voisins (KNN)**. 
1. **Extraction des caractéristiques :** Utilisation de 7 variables audio issues du [Spotify Tracks Dataset de Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset).
2. **Traitement des données :** Correction des biais sur l'instrumentalité pour les genres acoustiques (Jazz/House) et utilisation de l'intégralité du dataset pour une précision accrue.
3. **Standardisation :** Les données sont mises à la même échelle via `StandardScaler` de Scikit-Learn pour assurer un calcul de distance équitable.
4. **Modélisation :** L'algorithme `NearestNeighbors` (ball_tree) calcule la distance euclidienne entre le "Vecteur Idéal" et les milliers de chansons du dataset.

## 🛠️ Technologies Utilisées
* **Langage :** Python
* **Interface Web :** Streamlit
* **Machine Learning :** Scikit-Learn (KNN, StandardScaler)
* **Manipulation de Données :** Pandas, NumPy
* **Visualisation :** Plotly (Radar Charts)

## 💻 Installation en local
Pour faire tourner ce projet sur votre machine :

1. **Clonez ce dépôt :**
   ```bash
   git clone [https://github.com/FirasGhedir3/Moodify-App.git](https://github.com/FirasGhedir3/Moodify-App.git)
   cd Moodify-App
Préparez les données :
Téléchargez le fichier dataset.csv depuis Kaggle et placez-le à la racine du dossier.

Installez les bibliothèques :

Bash
pip install -r requirements.txt
Lancez l'application :

Bash
streamlit run app.py
👤 Contact
Firas Ghedir

🎓 Élève Ingénieur en Mathématiques Appliquées (MAM3) - Polytech Nice Sophia

📧 Email : firasghedir3@gmail.com

🔗 LinkedIn : Mon Profil
