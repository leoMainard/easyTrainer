creer un programme qui permet de sélectionner une préparation de données
de sélectionner des modèles et de les entrainer facilement
pouvoir sauvegarder les résultats
les visualiser

possiblement dans une interface streamlit
## TODO
`tests/`
- Ecrire des tests avant de publier sur Pypi
`data/`
- text.py:
    - encodage de texte : transformation en vecteurs (tfidf, word2vec, etc.)
    - stemming
    - retirer les mots très fréquents
    - retirer les mots rares
    - retirer les mots en majuscules pour filtrer les acronymes par exemple
    - Remplacement d’expressions régulières personnalisées : e.g., remplacer les emails, URLs hashtags, @mentions.
- numeric.py : transformation de données numériques
- categorial.py
- image.py
- video.py
- audio.py

`models/` : modèles sklearn dans un premier temps
- classification
- regression
- clustering
- dimensionality_reduction


## Installation
```
# Ajout de librairie au fichier pyproject.toml
poetry add library
```

```
pip install easyTrainer
```


## Structure
```
mon_projet/
├── easyTrainer/
│   ├── __init__.py
│   ├── data/
│   │   ├── base_preparator.py
│   │   ├── text.py.py # Contient la classe TextualPreparator  
│   │   └── utils.py 
│   ├── models/
│   │   ├── base_model.py
│   │   └── sklearn_classifier_model.py
│   ├── config/
│   │   └── enums.py
│   └── resources/
│       └── stopwords_fr.txt
├── .gitignore
├── poetry.lock
├── pyproject.toml
└── README.md

```