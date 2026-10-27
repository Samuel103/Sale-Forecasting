# Sale Forecasting

Projet Python avec un seul package, `sale_forecasting`, installé depuis `src/`.

## Développement

```sh
uv sync --locked
uv run pytest
uv run ruff check .
```

Les imports internes sont absolus (`from sale_forecasting.core.config import ...`).
Ruff trie les imports en trois groupes : bibliothèque standard, dépendances
externes et package du projet. Pour appliquer ce tri :

```sh
uv run ruff check --select I --fix .
```

Exécuter les modules depuis l'environnement installé avec `python -m`, plutôt
que lancer directement leurs fichiers :

```sh
uv run python -m sale_forecasting.pipelines.train_pipeline
```

L'entraînement nécessite les données `data/train.csv` et un serveur MLflow
accessible à l'adresse définie dans `config/config.yml`.
Les chemins relatifs de cette configuration sont résolus depuis la racine
du projet, indépendamment du répertoire courant.

Les transformations communes sont regroupées dans `data/preprocessing.py`.
`model/training.py` apprend les encodages des catégories, sépare la cible et
entraîne le modèle. `pipelines/train_pipeline.py` charge les données configurées
et lance cet entraînement.

`model/predict.py` charge le modèle sélectionné et réutilise ses encodages.
Il applique ensuite `data/postprocessing.py` pour retourner les colonnes
`id`, `customer_id`, `product`, `date`, `sales`, dans l'ordre des lignes retenues
par l'horizon configuré. Les libellés des produits viennent des données d'origine.
Pour réutiliser un modèle entre plusieurs appels :

```python
from sale_forecasting.model.predict import load_model, predict

model = load_model()
predictions = predict(test_df, model=model)
```

## API FastAPI

```sh
uv run uvicorn sale_forecasting.api.main:app --reload
```

La documentation interactive est disponible sur http://127.0.0.1:8000/docs.

- `POST /train` sans corps : exécute le pipeline existant avec les données et
  les paramètres de `config/config.yml`, puis recharge le modèle en mémoire.
  La requête attend la fin du training et retourne `{"status":"completed"}`.
  Les données configurées et le serveur MLflow doivent être disponibles.
- `POST /predict` : reçoit les lignes brutes et renvoie les prévisions.
  Le modèle est chargé au démarrage et réutilisé entre les requêtes.

```sh
curl -X POST http://127.0.0.1:8000/train

curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"rows":[{"id":10,"store_nbr":1,"family":"GROCERY","onpromotion":0,"date":"2026-01-01"}]}'
```

La réponse de prédiction contient `predictions`, une liste de lignes avec
`id`, `customer_id`, `product`, `date` et `sales`. L'horizon configuré
(`preprocessing.inference_days`) s'applique aux dates reçues. Les identifiants
doivent être uniques dans chaque requête.

Sans modèle exporté, `/predict` répond 503 et `/train` reste disponible.
Un deuxième entraînement simultané répond 409. En cas d'échec du training,
la réponse est 500, les détails sont journalisés et le modèle précédent reste
utilisable.

Cette API simple est destinée au développement local avec un seul processus
Uvicorn : le verrou d'entraînement et le modèle en mémoire sont propres à ce
processus. Le training lancé par HTTP n'est pas une tâche persistante ; arrêter
le serveur l'interrompt. Le pipeline existant journalise des métriques
d'entraînement, sans validation chronologique séparée à ce stade.

## Tests avec Postman

1. Importer `postman/sale_forecasting.postman_collection.json` et
   `postman/local.postman_environment.json` dans Postman via **Import**.
2. Sélectionner l'environnement **Sale Forecasting — Local**. La variable
   `base_url` vaut `http://127.0.0.1:8000` et peut être modifiée.
3. Démarrer l'API et MLflow, puis lancer les dossiers dans l'ordre avec le
   Collection Runner, ou envoyer les requêtes individuellement.

La collection contient 10 requêtes avec assertions : disponibilité OpenAPI,
entraînement réel, prédictions unitaires et multiples, et six cas de validation
qui doivent répondre 422. Le dossier d'entraînement relance le training à chaque
exécution ; il peut être désélectionné si un modèle est déjà chargé.
Pour un training long, augmenter le délai de requête dans les paramètres Postman
ou le régler à `0` pour attendre sans limite.
