# ML-SNCF-Transilien
Projet ML M2 IEF Dauphine 2025/2026 - Prévision en temps réel du temps d'attente au quai sur le réseau SNCF-Transilien à partir de données capteurs | Challenge Data ENS #166
# Prévision du temps d'attente au quai — SNCF Transilien

Projet Machine Learning réalisé dans le cadre du Master 2 IEF 2025/2026 à l'Université Paris Dauphine PSL.

**Auteurs :** Gopi VENOU & Rémi Schmitt  
**Enseignant :** Sylvain Benoît  
**Date de rendu :** 17 avril 2026  

---

## Contexte

Ce projet s'appuie sur le **Challenge Data ENS #166** proposé par SNCF-Transilien : *Real-time forecast of platform waiting time*. SNCF-Transilien opère 6 200 trains par jour pour 3,4 millions de voyageurs sur 84 gares d'Île-de-France. Des capteurs de pression installés sur les quais mesurent en temps réel les temps d'attente. L'objectif est de prédire le temps d'attente au quai 0 (`p0q0`) à partir de l'historique immédiat du train courant et des trains précédents à la même gare, sur une période de validation strictement postérieure à l'entraînement (novembre-décembre 2023).

La métrique imposée par le challenge est le **MAPE** (Mean Absolute Percentage Error), calculé en excluant les observations nulles (42,7% des valeurs de la cible).

---

## Structure du projet

```
ML-SNCF-Transilien/
│
├── Venou(Schmitt)_SNCF.ipynb    # Notebook principal complet
├── y_pred_RF_final.csv          # Prédictions soumises au challenge
└── README.md
```

---

## Démarche

Le notebook suit une progression du plus simple au plus complexe :

**1. Exploration et préparation des données**  
Analyse statistique des 667 264 observations, matrice de corrélation de Pearson, tests de Kruskal-Wallis pour détecter les associations non-linéaires entre features et cible.

**2. Feature engineering**  
Construction de 26 features organisées en six familles : temporalité (jour, mois, weekend), position dans le trajet (arrêt normalisé, phase), agrégats des gares précédentes (moyenne, max, écart-type, somme), agrégats des trains précédents (gradient de congestion), interaction croisée gare-train, et frequency encoding sans data leakage.

**3. Modèles de référence (Baseline)**  
DummyRegressor (MAPE = 1.000), Ridge (MAPE = 0.814), Lasso (MAPE = 0.814) avec validation croisée temporelle via TimeSeriesSplit.

**4. Modèle non supervisé — K-Means**  
Clustering des 84 gares sur leurs profils agrégés (21 statistiques par gare). Sélection du nombre optimal de clusters par méthode Elbow et score de silhouette. Le label de cluster est ensuite injecté comme feature dans les modèles supervisés.

**5. Modèles supervisés avancés**  
Random Forest avec GridSearchCV (MAPE = 0.777) et XGBoost avec RandomizedSearchCV, tous deux avec TimeSeriesSplit et vérification du ratio surapprentissage.

**6. Interprétabilité**  
Analyse SHAP (TreeExplainer sur XGBoost) et LIME sur Random Forest pour 3 observations aux profils contrastés. Les deux méthodes convergent vers les mêmes drivers : `p0q2`, `p0q3` et `cluster_gare` dominent les contributions.

**7. Deep Learning — MLP**  
Réseau dense 128→64→32→1 avec Batch Normalization, Dropout 20%, optimiseur Adam et Early Stopping (patience = 15). Entraîné sur 100 epochs sur GPU T4 (Google Colab).

---

## Résultats

| Rang | Modèle | MAPE validation |
|------|--------|----------------|
| 🥇 | MLP Dense (Deep Learning) | **0.7439** |
| 🥈 | Random Forest | 0.7768 |
| 🥉 | Ridge + KMeans | 0.8143 |
| 4 | Ridge / Lasso | 0.8145 |
| 5 | Naïf (médiane) | 1.0000 |
| 6 | XGBoost | 1.0750 |

Le MLP atteint les meilleures performances grâce à une convergence complète permise par le GPU. Le Random Forest est retenu pour la soumission finale en raison de sa stabilité et reproductibilité sur données tabulaires.

---

## Reproductibilité

Le notebook est exécutable sur **Google Colab** avec GPU T4. Les fichiers de données (`x_train_final.csv`, `x_test_final.csv`, `y_train_final_j5KGWWK.csv`, `y_sample_final.csv`) doivent être téléchargés depuis le [Challenge Data ENS #166](https://challengedata.ens.fr/participants/challenges/166/) et uploadés dans l'environnement Colab avant exécution.

Lien Google Colab : https://colab.research.google.com/drive/1KWbQjq2oIILy-BjCciSbhAibovQ93g8I?usp=sharing
