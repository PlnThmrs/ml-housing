# ML - Housing Project
Construction d'une page HTML avec prédiction par régression linéaire du prix d'une maison californienne

## Prérequis
- Python 3.10
- Outil de lecture de fichier .ipynb (VS Code, Jupyter, etc)

## Etapes
- Ouvrir le dossier dans Powershell et créer l'environnement virtuel: `py -3.10 -m venv .venv`
- Activer l'environnement virtuel : `.venv\Scripts\activate`
- Mettre à jour pip : `python -m pip install --upgrade pip`
- Installer les dépendences : `python -m pip install -r requirements.txt`
- Ouvrir le notebook et choisir le kernel associé au venv
- Lancer le notebook pour créer le modèle, le script de preprocessing et les métriques
- Lancer le backend : `uvicorn backend.app:app --reload`
- Dans une autre fenêtre Powershell, lancer le frontend : `streamlit run frontend/streamlit_app.py`
- L'application ouvre une page web en local, le modèle de régression linéaire peut être utilisé

## Exemple d'utilisation
- Revenu médian : 5.64
- Population du quartier : 558
- Age médian des maisons : 52
- Occupation moyenne : 2.55
- Nombre moyen de pièces : 5.82
- Latitude : 37.85
- Nombre moyen de chambres : 1.07
- Longitude : -122.25
> **Prix estimé** : 354,399$
