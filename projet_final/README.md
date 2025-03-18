# Prédiction des Retards de Vols

## Description du Projet

Ce projet a été développé pour prédire les retards des vols aériens en utilisant des techniques de Machine Learning. Le modèle est entraîné sur des données comprenant l'historique des vols, les compagnies aériennes, les aéroports ainsi que d'autres paramètres pertinents. 

L'objectif est d'aider les compagnies aériennes et les passagers à anticiper les retards afin d'optimiser la gestion du trafic aérien et améliorer l'expérience des voyageurs.

---

## Structure du Projet

Le projet est organisé en plusieurs fichiers et notebooks :

- **`entrainement.ipynb`** : Ce notebook contient le processus d'entraînement du modèle, incluant la préparation des données, la sélection des features et l'optimisation des hyperparamètres avec Optuna.
- **`prédiction.ipynb`** : Notebook permettant de tester le modèle sur des données en temps réel et d'afficher les prédictions. Il utilise l'API **AviationStack** via l'URL :
  ```
  https://api.aviationstack.com/v1/flights?access_key=3057f8d1b6d9283ee6a287082a3388af&dep_iata=JFK&flight_status=active
  ```
- **`app.py`** : Interface web développée avec Streamlit pour visualiser les prédictions des retards de vols en temps réel. Il récupère les données de l'API **AviationStack** avec la même URL.
- **`best_flight_delay_model2.pkl`** : Modèle de Machine Learning sauvegardé après entraînement.
- **`flights_sample_3m.csv`** : Dataset utilisé pour l'entraînement, issu de Kaggle : [Flight Delay and Cancellation Dataset 2019-2023](https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023/data?select=flights_sample_3m.csv).

---

## Sources des Données

### Jeu de données pour l'entraînement
Le fichier `flights_sample_3m.csv` provient de Kaggle et contient des informations détaillées sur des vols de 2019 à 2023, incluant :
- L'heure de départ et d'arrivée des vols
- Les délais de vol enregistrés
- Les compagnies aériennes et aéroports concernés
- Les retards dus à différents facteurs (météo, contrôle aérien, retard d’un avion précédent, etc.)

### API de récupération des vols en temps réel
L'API utilisée pour obtenir les vols en temps réel est **AviationStack**. Elle permet de récupérer des informations sur les vols en cours.

⚠️ **Limitations de l'API** :
- L'API gratuite permet uniquement **100 requêtes par mois**.
- Chaque requête retourne au maximum **100 vols**.
- Les données incluent le statut du vol, l'heure de départ et d'arrivée, ainsi que les retards détectés.

L'API est appelée dans les fichiers `prédiction.ipynb` et `app.py` via l'URL :
```
https://api.aviationstack.com/v1/flights?access_key=3057f8d1b6d9283ee6a287082a3388af&dep_iata=JFK&flight_status=active
```

---

## Installation et Exécution

### Prérequis

Avant d'exécuter le projet, assurez-vous d'avoir les outils suivants installés sur votre machine :

- Python 3.7+
- Poetry (gestionnaire de dépendances)
- Jupyter Notebook
- Streamlit

### Installation des dépendances avec Poetry

Le projet utilise **Poetry** pour gérer les dépendances. Pour installer Poetry, exécutez :
```bash
pip install poetry
```

Ensuite, pour installer toutes les dépendances du projet, exécutez :
```bash
poetry install
```

### Entraînement du Modèle

Pour entraîner le modèle, ouvrez Jupyter Notebook et exécutez `entrainement.ipynb` :
```bash
jupyter notebook
```
Ce notebook effectuera les étapes suivantes :
1. Chargement et prétraitement des données
2. Sélection des variables pertinentes
3. Construction et optimisation du modèle (Random Forest)
4. Entraînement et évaluation du modèle
5. Sauvegarde du modèle final sous `best_flight_delay_model2.pkl`

   
## Résultats de l'Entraînement du Modèle

Après entraînement du modèle Random Forest avec optimisation des hyperparamètres via Optuna, les performances obtenues sont :

- **MAE (Erreur absolue moyenne)** : 1.85 minutes
- **MSE (Erreur quadratique moyenne)** : 19.34
- **RMSE (Racine de l'erreur quadratique moyenne)** : 4.40 minutes
- **R² Score** : 0.9937

Ces résultats montrent que le modèle est très précis pour prédire les retards de vols.
   

### Prédiction en Temps Réel

Pour tester la prédiction sur des vols en temps réel, ouvrez et exécutez `prédiction.ipynb`.

### Lancer l'Application Web

Pour afficher les prédictions dans une interface utilisateur, lancez l'application Streamlit avec :
```bash
poetry run streamlit run app.py
```
L'interface sera disponible sur `http://localhost:8505`.

---

## Fonctionnalités du Projet

- **Traitement des données** : Nettoyage et transformation des données des vols aériens.
- **Modélisation Machine Learning** : Utilisation d'un modèle Random Forest optimisé avec Optuna.
- **Prédiction en temps réel** : Récupération des vols via l'API AviationStack et estimation des retards.
- **Interface Web** : Application Streamlit pour une visualisation intuitive des prédictions.


