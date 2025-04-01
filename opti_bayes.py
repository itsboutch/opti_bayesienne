# Mustapha Hacene DJEROUA GP3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, DotProduct, RationalQuadratic
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from skopt import BayesSearchCV
from skopt.plots import plot_convergence, plot_objective
from scipy.optimize import fmin_l_bfgs_b
from skopt import gp_minimize

# ---------------------------------------------------------------
# Partie 1 : Optimisation Bayésienne
# ---------------------------------------------------------------

data = pd.read_csv('tp2_atdn_donnees.csv')

# --------------------------
# Question 4 : Maximisation de la production agricole
# --------------------------
# Préparation des données
X = data[['Humidité (%)', 'Température (°C)']].values
y = data['Rendement agricole (t/ha)'].values

# Définition de la fonction objectif
def objective(params):
    """Fonction à maximiser (rendement agricole)"""
    humidity, temperature = params
    # Simulation d'un modèle 
    return -np.abs(humidity-70) - np.abs(temperature-25) + np.random.normal(0, 0.5)


space = [
    (20.0, 90.0),   # Plage humidité
    (10.0, 35.0)     # Plage température
]

result = gp_minimize(
    lambda x: -objective(x),  # On minimise l'opposé pour maximiser
    space,
    n_calls=30,
    random_state=42,
    n_initial_points=10
)

# Visualisation
plt.figure(figsize=(10, 6))
plt.plot(result.func_vals, 'ro-')
plt.title('Convergence de l\'optimisation')
plt.xlabel('Itération')
plt.ylabel('Rendement')
plt.show()

# --------------------------
# Question 5 : Optimisation des hyperparamètres
# --------------------------

# Préparation des données complètes
X = data.drop(['Rendement agricole (t/ha)', 'Type de sol'], axis=1)
y = data['Rendement agricole (t/ha)']
le = LabelEncoder()
X['Type de sol'] = le.fit_transform(data['Type de sol'])

# Split des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Recherche bayésienne
opt = BayesSearchCV(
    RandomForestRegressor(),
    {
        'n_estimators': (50, 200),
        'max_depth': (3, 20),
        'min_samples_split': (2, 10)
    },
    n_iter=30,
    cv=3,
    n_jobs=-1
)

opt.fit(X_train, y_train)

# Comparaison avec Grid Search et Random Search
# Paramètres pour les recherches
params = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [3, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10]
}

# Grid Search
grid_search = GridSearchCV(
    RandomForestRegressor(),
    params,
    cv=3,
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

# Random Search
random_search = RandomizedSearchCV(
    RandomForestRegressor(),
    params,
    n_iter=30,
    cv=3,
    n_jobs=-1,
    verbose=1
)
random_search.fit(X_train, y_train)

# Comparaison des résultats
print("\nComparaison des méthodes d'optimisation :")
print(f"Bayesian Search - Meilleur score : {opt.best_score_:.4f}")
print(f"Grid Search - Meilleur score : {grid_search.best_score_:.4f}")
print(f"Random Search - Meilleur score : {random_search.best_score_:.4f}")

# Affichage des meilleurs paramètres
print("\nMeilleurs paramètres Bayesian :", opt.best_params_)
print("Meilleurs paramètres Grid :", grid_search.best_params_)
print("Meilleurs paramètres Random :", random_search.best_params_)
# --------------------------
# Visualisation des résultats (Q6)
# --------------------------
plot_convergence(opt.optimizer_results_)
plt.title('Courbe de convergence')
plt.show()

# ---------------------------------------------------------------
# Partie 2 : Modèles Bayésiens à Noyau
# ---------------------------------------------------------------

# --------------------------
# Question 11 : Régression bayésienne
# --------------------------

# Sélection des caractéristiques
X = data[['Humidité (%)', 'Température (°C)']].values
y = data['Rendement agricole (t/ha)'].values

# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Définition du noyau
kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)

# Modèle
gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.1)
gpr.fit(X_scaled, y)

# Prédictions
X_test = np.linspace(X.min(axis=0), X.max(axis=0), 100)
y_mean, y_std = gpr.predict(X_test, return_std=True)

# Visualisation
plt.figure(figsize=(12, 6))
plt.scatter(X[:, 0], y, c='k', label='Données')
plt.plot(X_test[:, 0], y_mean, 'r-', label='Prédiction')
plt.fill_between(X_test[:, 0], y_mean-1.96*y_std, y_mean+1.96*y_std, alpha=0.2)
plt.xlabel('Humidité')
plt.ylabel('Rendement')
plt.legend()
plt.show()

# --------------------------
# Question 12 : Classification bayésienne
# --------------------------

# Préparation des données
X = data[['Humidité (%)', 'Température (°C)']].values
y = le.fit_transform(data['Type de sol'])

# Modèle
gpc = GaussianProcessClassifier(kernel=RBF())
gpc.fit(X, y)

# Comparaison avec SVM
svm = SVC()
svm.fit(X, y)

print(f"Précision GPC: {gpc.score(X, y):.2f}")
print(f"Précision SVM: {svm.score(X, y):.2f}")

# --------------------------
# Test de différents noyaux (Q14)
# --------------------------
# Optimiseur personnalisé avec maxiter augmenté
def custom_optimizer(obj_func, initial_theta, bounds):
    opt_res = fmin_l_bfgs_b(
        obj_func,
        initial_theta,
        bounds=bounds,
        maxiter=5000  # Augmenter le nombre d'itérations
    )
    return opt_res[0], opt_res[1]  

# Dictionnaire des noyaux
kernels = {
    'RBF': RBF(),
    'Matern': Matern(),
    'Linéaire': DotProduct()
}

# Test des noyaux avec l'optimiseur modifié
for name, kernel in kernels.items():
    model = GaussianProcessRegressor(kernel=kernel, optimizer=custom_optimizer, n_restarts_optimizer=10)
    model.fit(X_scaled, y)
    print(f"Score {name}: {model.score(X_scaled, y):.2f}")

# Précision GPC: 0.35
# Précision SVM: 0.35
# Score RBF: 1.00
# Score Matern: 1.00
# Score Linéaire: 0.00