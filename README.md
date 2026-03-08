# TP4 — Customer Churn Detection (MLOps)

Ce TP entraîne un modèle de classification du churn client avec 3 stratégies de gestion du déséquilibre des classes :
- sans correction du déséquilibre,
- avec `class_weight`,
- avec sur-échantillonnage `SMOTE`.

## Exécution

```bash
python -m pip install -r requirements.txt
python script.py
```

## Résultats générés

- `metrics.txt` : F1-score train/validation pour les 3 approches.
- `conf_matrix.png` : matrice de confusion combinée.
- `models/*.pkl` : modèles entraînés et sauvegardés.

## Structure

- `data/dataset.csv` : dataset source.
- `script.py` : pipeline data prep + entraînement + évaluation.
- `requirements.txt` : dépendances Python.

## DVC (optionnel)

```bash
python -m pip install dvc[s3]
dvc init
```

## GitHub Actions (CML + DVC)

Le workflow est prêt dans `.github/workflows/cml-churn.yaml`.

Ce qu'il fait automatiquement:
- installe Python et les dépendances,
- exécute `dvc repro` (stage `train` dans `dvc.yaml`),
- génère un rapport CML avec `metrics.txt` et `conf_matrix.png`,
- poste le rapport sur la PR,
- archive les artefacts (`metrics.txt`, `conf_matrix.png`, `models/`).

Pour l'utiliser:
1. push sur `main` ou ouvre une Pull Request,
2. lance manuellement via **workflow_dispatch** si nécessaire.

Note: `GITHUB_TOKEN` est fourni automatiquement par GitHub Actions pour publier le commentaire CML.

### Secrets GitHub pour remote DVC (optionnel)

Le workflow supporte 2 modes de remote DVC en CI:

- **S3** (prioritaire si configuré):
	- `DVC_S3_BUCKET` (ex: `my-dvc-bucket/path`)
	- `DVC_S3_REGION` (optionnel)
	- `DVC_S3_ENDPOINTURL` (optionnel, pour MinIO/S3 compatible)
	- `AWS_ACCESS_KEY_ID`
	- `AWS_SECRET_ACCESS_KEY`
	- `AWS_SESSION_TOKEN` (optionnel)

- **GDrive** (utilisé si S3 non défini):
	- `DVC_GDRIVE_FOLDER_ID`
	- `DVC_GDRIVE_SERVICE_ACCOUNT_JSON` (optionnel mais recommandé en CI)

Si aucun secret remote n'est fourni, le workflow exécute simplement le pipeline avec les données locales du repo.
