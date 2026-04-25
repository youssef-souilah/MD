# 🧠 Guide Complet de Révision — Deep Learning & Réseaux de Neurones
> **Niveau Master · Science des Données**  
> Document structuré pour une révision méthodique et efficace

---

## 📋 Table des Matières

1. [Le Neurone Artificiel & le Perceptron](#1-le-neurone-artificiel--le-perceptron)
2. [Fonctions d'Activation](#2-fonctions-dactivation)
3. [La Fonction de Coût (Log Loss)](#3-la-fonction-de-coût-log-loss)
4. [La Descente de Gradient](#4-la-descente-de-gradient)
5. [La Rétropropagation (Back-Propagation)](#5-la-rétropropagation-back-propagation)
6. [Les Réseaux Multicouches (MLP)](#6-les-réseaux-multicouches-mlp)
7. [Hyperparamètres & Stratégies d'Entraînement](#7-hyperparamètres--stratégies-dentraînement)
8. [Régularisation & Généralisation](#8-régularisation--généralisation)
9. [Architectures Spécialisées : CNN](#9-architectures-spécialisées--cnn)
10. [Architectures Séquentielles : RNN, LSTM, GRU](#10-architectures-séquentielles--rnn-lstm-gru)
11. [NLP & Word Embeddings](#11-nlp--word-embeddings)
12. [OCR & CTC](#12-ocr--ctc)
13. [Optimiseurs Avancés](#13-optimiseurs-avancés)
14. [Initialisation des Poids](#14-initialisation-des-poids)
15. [Batch Normalization & Dropout](#15-batch-normalization--dropout)
16. [Transfer Learning](#16-transfer-learning)
17. [Fiches Mémo : Formules Clés](#17-fiches-mémo--formules-clés)
18. [Checklist de Révision](#18-checklist-de-révision)

---

## 1. Le Neurone Artificiel & le Perceptron

### 🔬 Neurone Biologique vs Artificiel

| Composant biologique | Équivalent artificiel | Rôle |
|---|---|---|
| Dendrites | Entrées $x_1, x_2, \ldots, x_n$ | Réception des signaux |
| Poids synaptiques | $w_1, w_2, \ldots, w_n$ | Importance de chaque signal |
| Corps cellulaire (soma) | Agrégation $Z = \sum w_i x_i + b$ | Sommation pondérée |
| Seuil d'activation | Fonction d'activation $g(Z)$ | Déclenchement ou non |
| Axone | Sortie $\hat{y} = g(Z)$ | Transmission du résultat |

### ⚡ Le Perceptron (Rosenblatt, 1957)

**Définition :** unité de calcul élémentaire qui effectue une classification binaire linéaire.

**Équations fondamentales :**
```
Agrégation  :  Z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b  =  W·X + b
Activation  :  ŷ = g(Z)    (fonction échelon : 1 si Z ≥ 0, 0 sinon)
```

**Règle d'apprentissage de Hebb :**
```
W ← W + α(y - ŷ)X
b ← b + α(y - ŷ)
```
> **α (learning rate)** = taille du "pas" de correction  
> **(y - ŷ)** = erreur de prédiction (0 si correct, ±1 si erroné)

### 🚧 Limite fondamentale : le problème XOR

| $x_1$ | $x_2$ | XOR | Séparable ? |
|---|---|---|---|
| 0 | 0 | 0 | ❌ Impossible avec une seule droite |
| 0 | 1 | 1 | |
| 1 | 0 | 1 | |
| 1 | 1 | 0 | |

**Pourquoi ?** Le Perceptron trace une frontière **linéaire** (droite/hyperplan). Les données XOR ne sont pas linéairement séparables.

**Solution :** ajouter des couches cachées → MLP (Multi-Layer Perceptron)

### 📜 Historique clé à connaître

```
1943 → McCulloch & Pitts : premier neurone artificiel (binaire)
1957 → Rosenblatt : Perceptron avec apprentissage
1969 → Minsky & Papert : limites du Perceptron (XOR) → 1er hiver de l'IA
1986 → Rumelhart, Hinton, Williams : Back-Propagation → renaissance
1998 → LeCun : LeNet-5, premier CNN appliqué (reconnaissance chiffres)
2006 → Hinton : pré-entraînement non supervisé → résurgence Deep Learning
2012 → AlexNet (GPU + Big Data) → percée sur ImageNet
2017 → Transformers (Vaswani) → révolution NLP
```

---

## 2. Fonctions d'Activation

### 🔑 Rôle de la fonction d'activation

1. **Introduit la non-linéarité** : sans elle, un réseau profond reste linéaire
   - Sans activation : $W_2(W_1 X) = (W_2 W_1)X$ → équivalent à une seule couche
2. **Normalise la sortie** : transforme $Z \in (-\infty, +\infty)$ en valeur exploitable

### 📊 Tableau comparatif des fonctions d'activation

| Fonction | Formule | Sortie | Dérivée | Avantages | Inconvénients |
|---|---|---|---|---|---|
| **Sigmoïde** | $\frac{1}{1+e^{-z}}$ | $(0, 1)$ | $g(1-g)$ | Probabilité en sortie | Vanishing gradient pour $|z|$ grand |
| **Tanh** | $\frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $(-1, 1)$ | $1-g^2$ | Centré en 0 | Vanishing gradient |
| **ReLU** | $\max(0, z)$ | $[0, +\infty)$ | $0$ ou $1$ | Pas de vanishing, rapide | Neurones "morts" (z<0) |
| **Leaky ReLU** | $\max(0.01z, z)$ | $\mathbb{R}$ | $0.01$ ou $1$ | Évite neurones morts | Hyperparamètre supplémentaire |
| **Softmax** | $\frac{e^{z_k}}{\sum_j e^{z_j}}$ | $(0,1)$ somme=1 | Matrice jacobienne | Classification multiclasse | Coûteux en calcul |
| **Linéaire** | $z$ | $\mathbb{R}$ | $1$ | Régression | Aucune non-linéarité |

### ⭐ Dérivée de la Sigmoïde (à maîtriser absolument)

**Démonstration complète :**
```
g(z) = (1 + e^{-z})^{-1}

g'(z) = -(1 + e^{-z})^{-2} × (-e^{-z})
       = e^{-z} / (1 + e^{-z})²
       = (1 + e^{-z} - 1) / (1 + e^{-z})²
       = 1/(1+e^{-z}) - 1/(1+e^{-z})²
       = g(z) - g(z)²
       = g(z)(1 - g(z))    ✅
```

**Valeur maximale :** $g'(0) = 0.5 × 0.5 = 0.25$

### 🎯 Quelle activation choisir en sortie ?

| Tâche | Activation de sortie | Fonction de coût |
|---|---|---|
| Régression (valeur continue) | Linéaire ou ReLU | MSE |
| Classification binaire | Sigmoïde | Log Loss (BCE) |
| Classification multiclasse | Softmax | Entropie croisée |

---

## 3. La Fonction de Coût (Log Loss)

### 📐 Formule de la Log Loss (entropie croisée binaire)

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1-y^{(i)}) \log(1-\hat{y}^{(i)}) \right]$$

### 🔍 Analyse terme par terme

- Si **y = 1** → seul $\log(\hat{y})$ compte. Pénalisé si $\hat{y}$ proche de 0
- Si **y = 0** → seul $\log(1-\hat{y})$ compte. Pénalisé si $\hat{y}$ proche de 1
- **Résultat :** forte pénalité quand on est certain... mais faux !

### 🧮 Origine mathématique : Maximum de Vraisemblance

**Étape 1 — Modèle de Bernoulli :**
```
P(y | x; θ) = ŷ^y × (1-ŷ)^(1-y)
```

**Étape 2 — Vraisemblance conjointe (m exemples i.i.d.) :**
```
L(θ) = ∏ᵢ ŷ^(yᵢ) × (1-ŷ)^(1-yᵢ)
```

**Étape 3 — Log-vraisemblance (produit → somme) :**
```
log L(θ) = Σᵢ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]
```

**Étape 4 — Minimisation (changer le signe, normaliser) :**
```
J(θ) = -1/m × log L(θ)   ← descente de gradient minimise J
```

### ❓ Pourquoi le logarithme, le signe négatif et 1/m ?

| Élément | Justification |
|---|---|
| **log** | Transforme le produit en somme. Évite l'underflow numérique |
| **signe −** | Transforme la maximisation (vraisemblance) en minimisation (pour gradient) |
| **1/m** | Normalise par la taille du dataset. Rend α indépendant de m |

---

## 4. La Descente de Gradient

### 🎯 Principe fondamental

Trouver les paramètres $\theta$ qui **minimisent** la fonction de coût $J(\theta)$ en se déplaçant dans la direction opposée au gradient.

```
θ ← θ - α × ∂J/∂θ
```

### 📊 Comparaison des variantes

| Méthode | Données par update | Vitesse | Stabilité | Cas d'usage |
|---|---|---|---|---|
| **Batch GD** | Tout le dataset (m) | Lent | Très stable | Petits datasets |
| **SGD** | 1 exemple | Très rapide | Oscillant | Grands datasets, régularisation |
| **Mini-batch GD** | B exemples (32–256) | Rapide (GPU) | Bon compromis | ✅ **Standard en DL** |

### ⚠️ Impact du Learning Rate α

```
α trop grand  →  oscille autour du minimum, peut diverger
                 Courbe de perte : chaotique ou croissante

α optimal     →  convergence smooth et rapide
                 Courbe de perte : décroissante régulière

α trop petit  →  convergence très lente
                 Risque de blocage dans un minimum local
```

### 🔢 Formules de mise à jour complètes

**Poids et biais sur le dataset entier (forme matricielle) :**
```
∂J/∂θ = (1/m) × Xᵀ × (g(Z) - Y)     ← dimension : (n × 1)
∂J/∂b = (1/m) × Σ (g(Z) - Y)          ← scalaire

θ ← θ - α × ∂J/∂θ
b ← b - α × ∂J/∂b
```

---

## 5. La Rétropropagation (Back-Propagation)

### 🔗 La Règle de la Chaîne (Chain Rule)

**Principe :** si $J$ dépend de $g$, qui dépend de $Z$, qui dépend de $\theta$ :

$$\frac{dJ}{d\theta} = \frac{dJ}{dg} \times \frac{dg}{dZ} \times \frac{dZ}{d\theta}$$

### 📐 Décomposition des 3 dérivées intermédiaires

```
1. dJ/dg  =  -y/ŷ + (1-y)/(1-ŷ)            (dérivée de la Log Loss)

2. dg/dZ  =  g(Z)(1 - g(Z))                  (dérivée de la Sigmoïde)

3. dZ/dθ  =  X                                (dérivée de Z = Xθ + b)
```

**Résultat simplifié (remarquable !) :**
```
dJ/dθ = (1/m) × Xᵀ × (g(Z) - Y)
```

### 🔄 Algorithme de Back-Propagation pour un MLP à 2 couches

```
=== FORWARD PROPAGATION ===
Z[1] = W[1] × X + b[1]          → agrégation couche 1
A[1] = g(Z[1])                   → activation couche 1
Z[2] = W[2] × A[1] + b[2]       → agrégation couche 2
A[2] = g(Z[2])                   → prédiction finale (sortie)

=== CALCUL DE LA PERTE ===
J = -1/m × Σ [y log(A[2]) + (1-y) log(1-A[2])]

=== BACK PROPAGATION ===
dZ[2] = A[2] - Y                              ← erreur de sortie
dW[2] = (1/m) × dZ[2] × A[1]ᵀ
db[2] = (1/m) × Σ dZ[2]

dZ[1] = W[2]ᵀ × dZ[2] ⊙ g'(Z[1])           ← ⊙ = Hadamard (élément par élément)
dW[1] = (1/m) × dZ[1] × Xᵀ
db[1] = (1/m) × Σ dZ[1]

=== MISE À JOUR ===
W[l] ← W[l] - α × dW[l]
b[l] ← b[l] - α × db[l]
```

### 💡 Intuition de la Back-Propagation

> "On observe l'erreur en sortie, puis on remonte la chaîne couche par couche, en attribuant une 'part de responsabilité' à chaque poids, en utilisant la Chain Rule."

**Analogie :** enquêter sur un bug en production → identifier quelle étape du pipeline a le plus contribué à l'erreur, pour la corriger.

---

## 6. Les Réseaux Multicouches (MLP)

### 🏗️ Architecture générale

```
Entrée X     Couche cachée 1    Couche cachée 2    Sortie
(n features) → [n₁ neurones] → [n₂ neurones] → [1 ou K neurones]
              W[1]∈ℝ(n×n₁)   W[2]∈ℝ(n₁×n₂)   W[3]∈ℝ(n₂×K)
```

### 📐 Dimensions des matrices de poids

**Règle générale :** $W^{[l]} \in \mathbb{R}^{n_{l-1} \times n_l}$

| Couche | Dimension | Signification |
|---|---|---|
| $W^{[1]}$ | $(n_{\text{entrée}} \times n_1)$ | $n$ features → $n_1$ neurones cachés |
| $W^{[2]}$ | $(n_1 \times n_2)$ | $n_1$ → $n_2$ neurones |
| $W^{[L]}$ | $(n_{L-1} \times K)$ | Dernière couche → $K$ sorties |

**Exemple :** réseau (2 features → 4 cachés → 3 cachés → 1 sortie)
```
W[1] : (2 × 4)
W[2] : (4 × 3)
W[3] : (3 × 1)
```

### ⚙️ Les 4 étapes d'implémentation

```
1. INITIALISATION    → W aléatoires (petites valeurs), b = 0
2. FORWARD PROP     → Calcul des prédictions couche par couche
3. BACK PROP        → Calcul des gradients (Chain Rule)
4. MISE À JOUR      → θ ← θ - α × ∂J/∂θ
```

### 🔁 Théorème d'approximation universelle

> Un MLP avec **une seule couche cachée** suffisamment large et une activation **non-linéaire** peut approximer n'importe quelle fonction continue sur un ensemble compact.

---

## 7. Hyperparamètres & Stratégies d'Entraînement

### 📌 Définition

> Les **hyperparamètres** ne sont **pas appris** par le modèle. Ils sont fixés avant l'entraînement et contrôlent le processus d'apprentissage.

| Hyperparamètre | Valeurs typiques | Impact |
|---|---|---|
| Learning Rate α | 1e-4 à 1e-1 | Vitesse et stabilité de convergence |
| Batch Size | 32, 64, 128, 256 | Mémoire, vitesse, variance du gradient |
| Nombre d'epochs | 10 à 1000 | Durée d'entraînement |
| Nb de couches cachées | 1 à des dizaines | Expressivité du modèle |
| Nb de neurones/couche | 64 à 4096 | Capacité d'apprentissage |
| Dropout rate | 0.2 à 0.5 | Régularisation |

### 🔄 Epoch vs Batch Size vs Itération

```
Dataset : m exemples   →   Batch Size = B
Itérations par epoch   =   ⌈m / B⌉
Epoch                  =   1 passage complet du dataset
```

**Exemple :** dataset de 1000 exemples, batch size 32
```
Itérations par epoch = ⌈1000/32⌉ = 32 itérations
```

### 🛑 Early Stopping

**Algorithme :**
1. Évaluer $J_{\text{val}}$ à chaque epoch
2. Si $J_{\text{val}}$ n'améliore pas pendant `patience` epochs → STOP
3. Restaurer les poids de la meilleure epoch

**Avantage :** régularisation sans modifier l'architecture.

---

## 8. Régularisation & Généralisation

### 🎯 Overfitting vs Underfitting

```
UNDERFITTING (sous-apprentissage)        OVERFITTING (surapprentissage)
─────────────────────────────────        ─────────────────────────────────
Train Loss : élevée                      Train Loss : très faible (≈ 0)
Val Loss   : élevée                      Val Loss   : élevée (grand écart)
Cause      : modèle trop simple          Cause      : modèle trop complexe ou peu de données
Solution   : + de neurones, + d'epochs  Solution   : régularisation (voir ci-dessous)
```

### 🛡️ Techniques de régularisation

#### 1. L2 Regularization (Ridge / Weight Decay)
```
J_L2 = J + λ × Σ wⱼ²

Effet sur la mise à jour :
wⱼ ← wⱼ(1 - 2λα) - α × ∂J/∂wⱼ   ← "décroissance" des poids
```
→ Pousse les poids vers 0, mais jamais exactement à 0. Modèles lisses.

#### 2. L1 Regularization (Lasso)
```
J_L1 = J + λ × Σ |wⱼ|
```
→ Peut pousser certains poids **exactement à 0** → parcimonie (sparsity) → sélection de features

#### 3. Dropout
- Désactiver aléatoirement une fraction `p` des neurones à chaque forward pass
- À l'inférence : désactivé, mais sorties multipliées par `(1-p)` (inverted dropout)
- **Effet :** force le réseau à apprendre des représentations redondantes

#### 4. Data Augmentation
Générer de nouveaux exemples par transformations :
- Flip horizontal/vertical
- Rotation (±30°)
- Recadrage aléatoire (Random Crop)
- Modifications colorimétriques (luminosité, contraste)
- Mixup, CutMix (avancé)

#### 5. Early Stopping
→ Voir section 7

---

## 9. Architectures Spécialisées : CNN

### 🖼️ Pourquoi les CNN ?

**Problème des MLP pour les images :**
- Image 224×224×3 = 150 528 pixels en entrée
- MLP fully-connected → des millions de paramètres → overfitting, coûteux

**Solution CNN :** partage de poids (weight sharing) via des filtres de convolution

### 🔧 Composants d'un CNN

#### Couche de Convolution
```
Filtre (kernel) : matrice k×k de poids appris
Opération       : glissement du filtre sur l'image
Sortie          : feature map (carte de caractéristiques)
Formule         : (I ★ F)[i,j] = Σ Σ I[i+m, j+n] × F[m,n]
```

**Paramètres importants :**
- `kernel_size` : taille du filtre (3×3, 5×5)
- `stride` : pas de déplacement
- `padding` : rembourrage des bords (`same` ou `valid`)
- `nb_filtres` : nombre de feature maps produites

#### Couche de Pooling (Max Pooling)
```
Objectif : réduire la résolution spatiale
Max Pooling 2×2 : prend le max dans chaque bloc 2×2 → divise par 2 la taille
```

**3 rôles fondamentaux :**
1. Réduction dimensionnelle (moins de paramètres, plus rapide)
2. Invariance spatiale (légère translation → même sortie)
3. Prévention de l'overfitting

#### Architecture type CNN
```
IMAGE (H × W × C)
    ↓ Conv(32 filtres, 3×3) + ReLU
Feature maps (H × W × 32)
    ↓ MaxPool(2×2)
Feature maps (H/2 × W/2 × 32)
    ↓ Conv(64 filtres, 3×3) + ReLU
Feature maps (H/2 × W/2 × 64)
    ↓ MaxPool(2×2)
Feature maps (H/4 × W/4 × 64)
    ↓ Flatten
Vecteur 1D
    ↓ Dense(128) + ReLU + Dropout
    ↓ Dense(K) + Softmax
Prédiction (K classes)
```

### 📈 Vision Classique vs Deep Learning

| Aspect | Vision Classique | CNN (Deep Learning) |
|---|---|---|
| Features | Conçues manuellement (HOG, SIFT, Sobel) | Apprises automatiquement depuis les pixels |
| Adaptabilité | Difficile à transférer | Transfer Learning facile |
| Performance | Limitée | État de l'art |
| Expertise requise | Ingénierie manuelle extensive | Définir l'architecture + données |

**Hiérarchie de représentation des CNN :**
```
pixels → bords → formes simples → parties d'objets → objets complets
[conv1]   [conv2]     [conv3]           [conv4]          [FC]
```

---

## 10. Architectures Séquentielles : RNN, LSTM, GRU

### 🔁 RNN (Recurrent Neural Network)

**Motivation :** traiter des données **séquentielles** où l'ordre compte (texte, audio, séries temporelles).

**Équation de récurrence :**
```
hₜ = g(Wₕ × hₜ₋₁ + Wₓ × xₜ + b)
ŷₜ = softmax(Wᵧ × hₜ)
```

- `hₜ` : état caché à l'instant t (résumé du passé)
- `hₜ₋₁` : état caché précédent (mémoire)
- `xₜ` : entrée à l'instant t

**Différence fondamentale avec le MLP :**

| MLP (Feedforward) | RNN |
|---|---|
| Entrées indépendantes | Entrées séquentielles liées |
| Pas de mémoire | État caché = mémoire |
| $f(x_1), f(x_2)$ séparés | $h_t = f(h_{t-1}, x_t)$ |

### ⚠️ Problème du Vanishing Gradient

**Cause :** lors de la BPTT (rétropropagation à travers le temps), les gradients sont multipliés par la même matrice $W_h$ à chaque pas. Si ses valeurs propres < 1 :
```
∂J/∂h₁ ≈ ∏ₜ Wₕᵀ × g'(Zₜ) → 0   (exponentiellement)
```
Les premières positions de la séquence sont "oubliées".

### 🔐 LSTM (Long Short-Term Memory)

**Innovation :** un **état de cellule** $C_t$ (mémoire à long terme) protégé par des **portes apprenables**.

#### Les 3 portes du LSTM

```
1. Porte d'oubli  fₜ = σ(Wf × [hₜ₋₁, xₜ] + bf)
   → Que garder de Cₜ₋₁ ? (0 = oublier, 1 = garder)

2. Porte d'entrée iₜ = σ(Wi × [hₜ₋₁, xₜ] + bi)
   → Quelles nouvelles infos ajouter ?
   Candidat        C̃ₜ = tanh(Wc × [hₜ₋₁, xₜ] + bc)

3. Porte de sortie oₜ = σ(Wo × [hₜ₋₁, xₜ] + bo)
   → Que lire dans Cₜ pour former hₜ ?
```

**Mise à jour de la cellule :**
```
Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ
hₜ = oₜ ⊙ tanh(Cₜ)
```

**Pourquoi ça marche ?** Le gradient traverse $C_t$ de façon **additive** → pas de multiplication répétée → pas de vanishing.

### 🔄 GRU (Gated Recurrent Unit)

Version simplifiée du LSTM (2 portes au lieu de 3), plus rapide à entraîner :
```
Porte de reset   rₜ = σ(Wr × [hₜ₋₁, xₜ])
Porte de mise à jour zₜ = σ(Wz × [hₜ₋₁, xₜ])
Candidat         h̃ₜ = tanh(W × [rₜ ⊙ hₜ₋₁, xₜ])
hₜ = (1 - zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ
```

---

## 11. NLP & Word Embeddings

### 📝 Évolution des représentations textuelles

#### Approche classique : Bag of Words / One-Hot

```
Vocabulaire : ["chat", "chien", "maison"]
"chat"  → [1, 0, 0]
"chien" → [0, 1, 0]
"maison"→ [0, 0, 1]
```

**Problèmes :**
- Dimension = |vocabulaire| → très sparse (creux)
- Aucune sémantique : "chat" et "chien" sont orthogonaux
- Ignore l'ordre des mots

#### Révolution : Word Embeddings

```
"chat"    → [0.23, -0.15, 0.87, ...]   (vecteur dense de dim 100-300)
"chien"   → [0.21, -0.13, 0.85, ...]   (proche de "chat" !)
"voiture" → [-0.54, 0.72, -0.11, ...]  (éloigné de "chat")
```

**Propriété remarquable (Word2Vec) :**
```
Roi⃗ - Homme⃗ + Femme⃗ ≈ Reine⃗
Paris⃗ - France⃗ + Italie⃗ ≈ Rome⃗
```

#### Méthodes d'embeddings

| Méthode | Année | Principe |
|---|---|---|
| Word2Vec (Skip-gram/CBOW) | 2013 | Prédire mot → contexte ou contexte → mot |
| GloVe | 2014 | Factorisation de la matrice de co-occurrence |
| FastText | 2016 | Embeddings de sous-mots (gère les mots inconnus) |
| BERT | 2018 | Embeddings contextuels bidirectionnels |

### 🤖 Révolution Deep Learning en NLP

| Aspect | Méthodes classiques | Deep Learning |
|---|---|---|
| Features | Règles grammaticales manuelles | Apprises automatiquement |
| Contexte | Local (n-grammes) | Global (Attention / Transformers) |
| Polysémie | Non géré | BERT : représentations contextuelles |
| Transfert | Difficile | BERT, GPT : pré-entraînement masqué |

---

## 12. OCR & CTC

### 📄 Évolution de l'OCR

#### OCR Classique (pipeline manuel)

```
Image
  ↓ Prétraitement (binarisation, débruitage)
  ↓ Segmentation (découper chaque caractère)  ← GOULOT D'ÉTRANGLEMENT
  ↓ Extraction de features (HOG, projections)
  ↓ Classification (SVM, k-NN)
Texte
```

**Limites :**
- Requiert une segmentation parfaite
- Sensible aux polices inconnues, à l'inclinaison, aux caractères collés

#### OCR Moderne (Deep Learning)

```
Image de ligne de texte
  ↓ CNN     → feature maps (caractéristiques visuelles)
  ↓ BLSTM   → séquence de vecteurs contextuels (bidirectionnel)
  ↓ CTC     → transcription sans segmentation manuelle
Texte
```

### 🔗 CTC (Connectionist Temporal Classification)

**Problème résolu :** entrée de longueur $T$ (frames CNN), sortie de longueur $L < T$ (caractères). Pas d'alignement manuel.

**Solution CTC :**
1. Introduit un symbole **blank** (ε) pour les frames sans caractère
2. Marginalise sur **tous les alignements possibles** via programmation dynamique (forward-backward)

**Exemple d'alignement :**
```
Transcription : "CAT"
Alignements valides sur 8 frames :
  C ε C A ε A T ε T   → collapse → "CAT"
  C C ε A A ε T T ε   → collapse → "CAT"
  ...
```

**Règles de décodage :**
- Supprimer les blancs
- Supprimer les répétitions consécutives (sauf si séparées par un blanc)

---

## 13. Optimiseurs Avancés

### 📊 Hiérarchie des optimiseurs

#### SGD + Momentum
```
vₜ = β × vₜ₋₁ + (1-β) × gₜ     (β = 0.9 typiquement)
θ ← θ - α × vₜ
```
→ Accumule une "vitesse" dans les directions persistantes. Passe les plateaux.

#### RMSProp
```
sₜ = β × sₜ₋₁ + (1-β) × gₜ²    (moyenne des carrés)
θᵢ ← θᵢ - α/√(sₜ + ε) × gₜ
```
→ Adapte le LR **par dimension**. Ideal pour les RNN.

#### Adam (Adaptive Moment Estimation)
```
mₜ = β₁ × mₜ₋₁ + (1-β₁) × gₜ      (1er moment = momentum)
vₜ = β₂ × vₜ₋₁ + (1-β₂) × gₜ²     (2ème moment = RMSProp)

m̂ₜ = mₜ/(1-β₁ᵗ)                    (correction du biais)
v̂ₜ = vₜ/(1-β₂ᵗ)

θ ← θ - α × m̂ₜ/√(v̂ₜ + ε)
```

**Paramètres par défaut :** α=0.001, β₁=0.9, β₂=0.999, ε=1e-8

| Optimiseur | Avantage | Inconvénient |
|---|---|---|
| SGD | Simple, peut mieux généraliser | Lent, sensible au LR |
| SGD + Momentum | Accélère la convergence | Hyperparamètre β |
| RMSProp | Bon pour RNN | Pas de correction de biais |
| **Adam** | Rapide, robuste, standard | Peut sur-adapter (fine-tuning) |

**Recommandation :** démarrer avec **Adam (α=1e-3)** pour la plupart des tâches.

---

## 14. Initialisation des Poids

### ⚠️ Pourquoi pas initialiser à zéro ?

Si $W = 0$ : tous les neurones d'une même couche calculent exactement la même chose → reçoivent les mêmes gradients → restent identiques. **Symétrie non brisée.**

### 📐 Méthodes d'initialisation

| Méthode | Formule | Pour quelle activation |
|---|---|---|
| Xavier/Glorot | $\mathcal{N}(0, \sqrt{2/(n_{in}+n_{out})})$ | Tanh, Sigmoïde |
| He | $\mathcal{N}(0, \sqrt{2/n_{in}})$ | ReLU |
| Random normal | $\mathcal{N}(0, 0.01)$ | Générique (petits réseaux) |

**Biais :** toujours initialisé à **0** (pas de problème de symétrie pour le biais).

---

## 15. Batch Normalization & Dropout

### 📊 Batch Normalization (BN)

**Formule :**
```
μ_B = (1/m) × Σ xᵢ                (moyenne du batch)
σ²_B = (1/m) × Σ (xᵢ - μ_B)²     (variance du batch)
x̂ᵢ = (xᵢ - μ_B) / √(σ²_B + ε)   (normalisation)
yᵢ = γ × x̂ᵢ + β                   (ré-scaling appris)
```

**Avantages :**
- Permet des LR plus élevés → convergence plus rapide
- Réduit la sensibilité à l'initialisation
- Légère régularisation
- Combat le "covariate shift" interne

**Placement :** généralement entre la couche linéaire et l'activation : `Dense → BN → ReLU`

### 🎲 Dropout

```
Entraînement : désactiver chaque neurone avec probabilité p
               ỹᵢ = yᵢ × Bernoulli(1-p) / (1-p)   (inverted dropout)

Inférence    : désactivé (toutes les connexions actives)
               Sorties multipliées par (1-p) pour conserver l'espérance
```

**Valeurs typiques :** p = 0.2 à 0.5

**Intuition :** équivalent à entraîner $2^n$ sous-réseaux différents et faire leur moyenne.

---

## 16. Transfer Learning

### 🔄 Principe

Réutiliser un modèle pré-entraîné sur une tâche source (ex: ImageNet) comme point de départ pour une tâche cible avec peu de données.

**Pourquoi ça marche ?**  
Les premières couches d'un CNN apprennent des features **universelles** (bords, textures, formes) → réutilisables pour toute tâche de vision.

### 🎯 Stratégies

#### Feature Extraction (geler les couches)
```
Modèle pré-entraîné (ex: ResNet50)
  → Geler TOUTES les couches de base
  → Ajouter et entraîner seulement la couche de classification finale
  Quand utiliser : peu de données, tâche similaire à la source
```

#### Fine-tuning (débloquer progressivement)
```
Modèle pré-entraîné
  → Geler les premières couches
  → Fine-tuner les dernières couches avec un faible LR (ex: 1e-5)
  Quand utiliser : données suffisantes, plus de flexibilité requise
```

### 📋 Quand utiliser le Transfer Learning ?

| Données disponibles | Similarité avec source | Stratégie recommandée |
|---|---|---|
| Peu | Haute | Feature extraction |
| Peu | Faible | Fine-tuning des couches finales |
| Beaucoup | Haute | Fine-tuning global (LR faible) |
| Beaucoup | Faible | Entraîner from scratch |

---

## 17. Fiches Mémo : Formules Clés

### 🔷 Forward Propagation

```
Pour chaque couche l = 1, ..., L :
  Z[l] = W[l] × A[l-1] + b[l]
  A[l] = g[l](Z[l])

avec A[0] = X  (l'entrée)
```

### 🔷 Log Loss

```
J(θ) = -1/m × Σᵢ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]
```

### 🔷 Sigmoïde et dérivée

```
g(z) = 1 / (1 + e^{-z})
g'(z) = g(z)(1 - g(z))
```

### 🔷 Softmax

```
Softmax(Zₖ) = e^{Zₖ} / Σⱼ e^{Zⱼ}     ∀k = 1,...,K
```

### 🔷 Back-Propagation (réseau à 2 couches)

```
dZ[2] = A[2] - Y
dW[2] = 1/m × dZ[2] × A[1]ᵀ
db[2] = 1/m × Σ dZ[2]

dZ[1] = W[2]ᵀ × dZ[2] ⊙ g'(Z[1])
dW[1] = 1/m × dZ[1] × Xᵀ
db[1] = 1/m × Σ dZ[1]
```

### 🔷 Mise à jour des paramètres

```
W[l] ← W[l] - α × dW[l]
b[l] ← b[l] - α × db[l]
```

### 🔷 Perceptron (règle de Hebb)

```
W ← W + α(y - ŷ)X
b ← b + α(y - ŷ)
```

### 🔷 Gradients finaux (Logistic Regression)

```
∂J/∂θ = 1/m × Xᵀ(g(Z) - Y)      ∈ ℝⁿˣ¹
∂J/∂b = 1/m × Σ(g(Z) - Y)        ∈ ℝ
```

### 🔷 LSTM — Portes

```
fₜ = σ(Wf[hₜ₋₁, xₜ] + bf)      ← oubli
iₜ = σ(Wi[hₜ₋₁, xₜ] + bi)      ← entrée
C̃ₜ = tanh(Wc[hₜ₋₁, xₜ] + bc)  ← candidat cellule
oₜ = σ(Wo[hₜ₋₁, xₜ] + bo)      ← sortie

Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ
hₜ = oₜ ⊙ tanh(Cₜ)
```

### 🔷 Adam

```
mₜ = β₁mₜ₋₁ + (1-β₁)gₜ     m̂ₜ = mₜ/(1-β₁ᵗ)
vₜ = β₂vₜ₋₁ + (1-β₂)gₜ²    v̂ₜ = vₜ/(1-β₂ᵗ)
θ ← θ - α × m̂ₜ/√(v̂ₜ + ε)
```

---

## 18. Checklist de Révision

Utilisez cette liste pour valider votre préparation avant l'examen.

### ✅ Niveau 1 — Fondamentaux (OBLIGATOIRE)

- [ ] Je peux expliquer la différence entre neurone biologique et artificiel
- [ ] Je connais les 4 étapes d'un réseau de neurones (Init, Forward, Back, Update)
- [ ] Je sais écrire la formule de mise à jour du Perceptron et donner un exemple numérique
- [ ] Je peux expliquer pourquoi le XOR est impossible pour un Perceptron
- [ ] Je connais la formule de la Log Loss et son origine (MLE + Bernoulli)
- [ ] Je sais justifier le logarithme, le signe − et le 1/m dans la Log Loss
- [ ] Je connais la différence entre Epoch, Batch Size et Itération
- [ ] Je sais expliquer l'Overfitting et 3 techniques pour le contrer
- [ ] Je sais ce que fait l'Early Stopping

### ✅ Niveau 2 — Mathématiques (ESSENTIEL pour Master)

- [ ] Je peux démontrer que $g'(z) = g(z)(1-g(z))$ (dérivée Sigmoïde)
- [ ] Je peux dériver la Log Loss depuis la vraisemblance de Bernoulli
- [ ] Je sais appliquer la Chain Rule pour $dJ/d\theta$
- [ ] Je peux donner les formules de back-propagation pour 2 couches ($dZ^{[1]}$, $dZ^{[2]}$)
- [ ] Je sais calculer les dimensions des matrices $W^{[l]}$ pour une architecture donnée
- [ ] Je comprends pourquoi $dZ/d\theta = X$
- [ ] Je sais pourquoi initialiser les poids à 0 est une erreur

### ✅ Niveau 3 — Architectures (CULTURE GÉNÉRALE)

- [ ] Je sais à quoi sert un CNN et comment fonctionne la convolution
- [ ] Je comprends le rôle du Max Pooling (3 raisons)
- [ ] Je peux expliquer la différence MLP vs RNN (mémoire / séquence)
- [ ] Je comprends le problème du Vanishing Gradient et comment le LSTM le résout
- [ ] Je connais les 3 portes du LSTM et leur rôle
- [ ] Je sais ce qu'est un Word Embedding vs Bag of Words
- [ ] Je comprends pourquoi la CTC est nécessaire en OCR/ASR
- [ ] Je peux décrire le pipeline CNN + RNN + CTC pour l'OCR

### ✅ Niveau 4 — Implémentation Python (BONUS)

- [ ] Je sais implémenter la Sigmoïde et la Log Loss en NumPy
- [ ] Je peux écrire la boucle de descente de gradient complète
- [ ] Je sais utiliser `MLPClassifier` vs `SGDClassifier` de Scikit-Learn
- [ ] Je comprends la différence `MLPClassifier` vs `MLPRegressor`
- [ ] Je peux identifier une erreur d'initialisation dans un code de réseau de neurones

---

## 💡 Conseils Stratégiques pour l'Examen

### 🎯 Priorités absolues (fort coefficient)

1. **La démonstration de $g'(z) = g(z)(1-g(z))$** — faites-la 3 fois à la main
2. **Les formules de back-propagation** ($dZ^{[1]}$, $dZ^{[2]}$, $dW$, $db$) — savoir les réécrire sans document
3. **La dérivation de la Log Loss** depuis la vraisemblance — comprendre chaque étape
4. **L'impact du learning rate** — savoir décrire visuellement ce qui se passe
5. **La Chain Rule** — décomposer $dJ/d\theta$ en 3 dérivées intermédiaires

### 📝 Technique d'auto-test

> Prenez une feuille blanche et réécrivez de mémoire :
> 1. L'algorithme de back-propagation (forward + backward + update)
> 2. Les 3 dérivées intermédiaires de la Chain Rule
> 3. La dérivée de la Sigmoïde (démonstration complète)

### ⚡ Points souvent confondus

| Confusion fréquente | Clarification |
|---|---|
| Epoch vs Itération | Epoch = 1 passage complet / Itération = 1 mise à jour |
| Overfitting vs Underfitting | Trop complexe vs trop simple |
| Sigmoïde vs Softmax | Binaire (1 sortie) vs Multiclasse (K sorties) |
| $dZ^{[2]}$ vs $dZ^{[1]}$ | Couche sortie : $A-Y$ / Couche cachée : $W^{T} \cdot dZ \odot g'$ |
| Batch Size vs Epoch | Nombre d'exemples par update vs passage complet |

---

*Document généré à des fins pédagogiques — Module Deep Learning, Master Science des Données*
