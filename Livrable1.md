# Livrable 1 — Facial Emotion Detection (Computer Vision)

## 1) Titre du projet
**Détection des émotions faciales en temps réel (Real-Time Facial Emotion Detection)**

---

## 2) Contexte
La reconnaissance des émotions à partir d’un visage est un domaine important en **Computer Vision** et en **Deep Learning**.  
Elle est utilisée dans plusieurs applications comme :
- l’analyse du comportement humain
- les systèmes interactifs (IA, assistants)
- la surveillance intelligente
- l’amélioration de l’expérience utilisateur

---

## 3) Objectif du projet
L’objectif principal de ce projet est de développer un système capable de :
- détecter un visage dans une image ou une vidéo
- reconnaître automatiquement l’émotion du visage
- afficher l’émotion prédite en **temps réel** via une webcam

---

## 4) Problématique
**Comment construire un modèle de classification capable de reconnaître correctement les émotions faciales, malgré :**
- la variation de lumière
- les différentes poses du visage
- les différences entre les personnes (âge, genre, expressions)
- la qualité variable des images
- le besoin d’une exécution rapide en temps réel

---

## 5) Dataset utilisé
Le dataset utilisé est un dataset d’images de visages classées selon différentes émotions.

📌 **Lien du dataset / référence :**  
https://www.kaggle.com/code/ujjwalkar/transfer-learnig-realtime-facial-emotion-detection

📌 **Classes (émotions) présentes :**
- Angry
- Disgusted
- Fearful
- Happy
- Neutral
- Sad
- Surprised

📌 **Structure du dataset :**
Le dataset est organisé en deux parties :
- `train/` : données d'entraînement
- `test/` : données de test



## 6) Méthodologie (Approche proposée)
Le projet sera réalisé en 2 phases principales :

### Phase 1 — Entraînement du modèle
- Prétraitement des images (resize, normalisation)
- Utilisation du **Transfer Learning**
- Entraînement sur le dossier `train/`
- Validation sur une partie des données (`validation split`)
- Évaluation finale sur `test/`

### Phase 2 — Détection en temps réel
- Utilisation de la webcam via OpenCV
- Détection du visage (face detection)
- Prédiction de l’émotion par le modèle entraîné
- Affichage du résultat en direct (label + probabilité)

---

## 7) Outils et technologies
- **Python**
- **Kaggle**
- **OpenCV**
- **Google Colab** (entraînement + GPU)
- **VS Code** (exécution en local)
- **GitHub** (versioning + livrables)

---

## 8) Résultats attendus
À la fin du projet, on doit obtenir :
- un modèle entraîné sauvegardé (`.h5` ou SavedModel)
- une précision correcte sur le test set
- une application en temps réel affichant l’émotion détectée
- un dépôt GitHub propre et structuré
