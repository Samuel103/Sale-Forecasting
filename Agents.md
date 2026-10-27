Architecture du projet — Sale Forecasting
Objectif
Développer un projet Python de Machine Learning qui regroupe l'entraînement d'un modèle LightGBM, l'inférence via une API FastAPI et le suivi des expériences avec MLflow.

Le projet doit être simple à développer localement avec uv, testable et prêt à être conteneurisé avec Docker, puis éventuellement déployé sur Kubernetes.
Principes d'architecture
- Utiliser un seul dépôt Git, un seul pyproject.toml et un seul package Python (sale_forecasting).
- Adopter le src layout : le code métier réside dans src/sale_forecasting/.
- Séparer la logique de l'API, du modèle, des données et des pipelines.
- L'API fait de l'inférence uniquement : elle n'entraîne pas un modèle à chaque requête.
- Le pipeline entraîne et évalue le modèle, puis sauvegarde une version exploitable par l'API.
- Mutualiser les transformations de features nécessaires à l'entraînement et à l'inférence, en évitant toute fuite de données temporelle.
- Configurer les chemins, hyperparamètres et connexions par configuration; ne pas les coder en dur.
- Les notebooks servent à explorer et expérimenter; le code réutilisable doit vivre dans src/.
Structure des dossiers
sale-forecasting/
├── pyproject.toml               # Dépendances, packaging et configuration des outils
├── uv.lock                      # Verrouillage des dépendances
├── README.md
├── .gitignore
├── .env.example                # Exemple des variables d'environnement (sans secrets)
├── config.yaml                 # Paramètres d'entraînement et de forecasting
├── Dockerfile                  # Image Docker (API et/ou entraînement)
├── docker-compose.yml          # Services locaux (API, MLflow, etc.)
│
├── data/                       # Données locales, normalement ignorées par Git
│   ├── raw/                    # Données sources
│   ├── processed/              # Données nettoyées et features préparées
│   └── predictions/            # Exports de prévisions
│
├── models/                     # Artefacts de modèles locaux (ignorés par Git)
│
├── notebooks/
│   ├── exploration.ipynb       # Analyse exploratoire
│   └── training.ipynb          # Tests et expérimentations
│
├── src/
│   └── sale_forecasting/
│       ├── __init__.py
│       │
│       ├── api/
│       │   ├── __init__.py
│       │   ├── main.py         # Création de l'application FastAPI, lifespan
│       │   ├── routes.py       # Endpoints HTTP (ex. POST /predict)
│       │   └── schemas.py      # Contrats Pydantic request/response
│       │
│       ├── model/
│       │   ├── __init__.py
│       │   ├── training.py     # Entraînement LightGBM
│       │   ├── predict.py      # Chargement du modèle et inférence
│       │   └── evaluation.py   # MAE, RMSE, WAPE et autres métriques
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loader.py       # Chargement des données (Polars)
│       │   ├── preprocessing.py # Nettoyage et préparation
│       │   ├── features.py     # LAG, rolling mean, variables calendaires
|       |   └── postprocessing.py # Transformation avec les données du test
│       │
│       ├── pipelines/
│       │   ├── __init__.py
│       │   └── train_pipeline.py # Orchestration du training
│       │
│       └── core/
│           ├── __init__.py
│           ├── config.py       # Lecture et validation de la configuration
│           └── mlflow.py       # Fonctions d'intégration MLflow
│
└── tests/
    ├── test_features.py
    ├── test_training.py
    ├── test_predict.py
    └── test_api.py
Attention : data/ à la racine contient les fichiers de données; src/sale_forecasting/data/ contient le code Python de traitement des données. De même, models/ contient les artefacts sauvegardés, tandis que src/sale_forecasting/model/ contient le code du modèle.
Responsabilités des modules
Module	Responsabilité	Ne doit pas faire
api/	Valider les requêtes, appeler le service de prédiction, renvoyer des réponses HTTP	Entraîner le modèle, implémenter le preprocessing métier
model/	Entraîner, évaluer, charger et exécuter LightGBM	Dépendre de FastAPI ou des routes HTTP
data/ (package)	Charger, nettoyer et transformer les données; construire les features	Exposer des endpoints HTTP
pipelines/	Orchestrer les étapes de training, validation, métriques et publication	Contenir tout le code métier en doublon
core/	Configurer l'application et les intégrations transversales	Contenir la logique métier du forecasting
notebooks/	Explorer et comparer les expériences	Devenir la seule source du code de production


Flux d'entraînement
Données historiques
       ↓
data/loader.py
       ↓
data/preprocessing.py
       ↓
data/features.py
       ↓
model/training.py
       ↓
model/evaluation.py
       ↓
MLflow (métriques, paramètres, artefact du modèle)
       ↓
Modèle versionné utilisable par l'API
Le fichier pipelines/train_pipeline.py orchestre ce flux. Une validation chronologique doit être utilisée pour un problème de forecasting : ne jamais entraîner sur des données futures par rapport à la période validée.
Flux d'inférence
Client HTTP
    ↓
FastAPI — api/routes.py
    ↓
Validation — api/schemas.py
    ↓
Construction des features à partir des données disponibles
    ↓
model/predict.py
    ↓
Modèle LightGBM déjà chargé
    ↓
Réponse JSON
Le modèle est chargé au démarrage de l'API (par exemple via le lifespan FastAPI) et réutilisé entre les requêtes. En production, choisir explicitement une version du modèle, plutôt que charger automatiquement une version aléatoire ou le dernier fichier trouvé.

Important pour le forecasting : les features de type LAG ou rolling mean nécessitent un historique disponible au moment de la prédiction. L'API doit disposer de cet historique ou recevoir des features correctement calculées; elle ne peut pas les déduire d'une seule valeur isolée. Il faut conserver l'ordre des colonnes et l'encodage des catégories attendus par le modèle.
Commandes de développement
Une fois le package installé dans l'environnement uv :
# Installer les dépendances verrouillées
uv sync

# Démarrer FastAPI (développement)
uv run uvicorn sale_forecasting.api.main:app --reload

# Exécuter le pipeline d'entraînement
uv run python -m sale_forecasting.pipelines.train_pipeline

# Exécuter les tests (pytest doit être dans les dépendances de développement)
uv run pytest
train_pipeline.py doit exposer un point d'entrée exécutable (if __name__ == "__main__":). Le pyproject.toml doit être configuré pour installer le package depuis src/.
Docker, MLflow et Kubernetes
- Local : docker-compose.yml permet de démarrer l'API et le serveur MLflow avec leur stockage persistant.
- Entraînement : un processus distinct (commande, conteneur ponctuel ou Kubernetes Job) entraîne et enregistre un nouveau modèle dans MLflow.
- Inférence : l'API est un service à longue durée de vie qui charge une version sélectionnée du modèle et reçoit les requêtes de prédiction.
- Docker : au départ, la même image peut servir à l'API et au training avec des commandes différentes. Les séparer en deux images reste possible par la suite.
- Kubernetes : une API peut fonctionner dans un Deployment, les entraînements dans des Jobs ou CronJobs, et MLflow dans un service avec une base et un stockage d'artefacts persistants.
- CI/CD : à chaque push sur main, exécuter les tests, construire l'image et la publier dans un registre. Un déploiement peut ensuite récupérer cette image sans devoir réentraîner systématiquement le modèle.
Règles pour l'agent de développement
1. Respecter la structure des dossiers et les responsabilités décrites dans ce document.
2. Ne pas créer de deuxième projet Python pour l'API sans besoin architectural explicite.
3. Centraliser le code réutilisable sous src/sale_forecasting/.
4. Ne pas dupliquer les transformations de features entre le training et l'inférence.
5. Ne jamais importer api/ depuis model/, data/ ou pipelines/.
6. Utiliser des imports absolus, par exemple from sale_forecasting.model.predict import predict.
7. Garder les endpoints HTTP légers; placer la logique de prédiction dans model/predict.py.
8. Ne pas entraîner le modèle lors des appels HTTP de prédiction.
9. Stocker les modèles et données générés hors du code source; ne pas les committer par défaut.
10. Utiliser uv pour la gestion des dépendances et l'exécution.
11. Ajouter ou mettre à jour les tests pour les nouvelles fonctions importantes.
12. Utiliser MLflow pour suivre paramètres, métriques et versions des modèles.
13. Préserver une séparation temporelle stricte entre entraînement et validation.
14. Éviter de complexifier prématurément l'architecture (microservices, abstractions inutiles, plusieurs packages).
Hors périmètre initial
- Architecture microservices.
- Entraînement directement dans les endpoints FastAPI.
- Orchestrateur de workflows complexe tant qu'un pipeline exécutable suffit.
- Kubernetes obligatoire pour le développement local.