# -*- coding: utf-8 -*-
"""
Script para criar um dataset simulado, treinar um modelo de Machine Learning
e salvá-lo para o projeto Global Solution - Monitor de Enchentes.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Diretório atual do script
folder_path = os.path.dirname(os.path.abspath(__file__))

# 1. Criar Dataset Simulado
# Simulando leituras de um sensor ultrassônico (distância em cm)
# Menor distância = maior nível de água = maior risco
np.random.seed(42)  # Para reprodutibilidade

# Níveis normais (distância grande)
distancia_normal = np.random.uniform(50, 100, 300)  # 300 amostras, 50-100 cm
risco_normal = np.zeros(300)  # Risco 0: Normal

# Níveis de alerta (distância média)
distancia_alerta = np.random.uniform(20, 50, 150)  # 150 amostras, 20-50 cm
risco_alerta = np.ones(150)  # Risco 1: Alerta

# Níveis de inundação (distância pequena)
distancia_inundacao = np.random.uniform(5, 20, 50)  # 50 amostras, 5-20 cm
risco_inundacao = np.full(50, 2)  # Risco 2: Inundação

# Combinar os dados
distancias = np.concatenate([distancia_normal, distancia_alerta, distancia_inundacao])
riscos = np.concatenate([risco_normal, risco_alerta, risco_inundacao])

# Criar DataFrame
data = pd.DataFrame({
    'distancia_cm': distancias,
    'nivel_risco': riscos.astype(int)  # 0: Normal, 1: Alerta, 2: Inundação
})

# Embaralhar os dados
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Salvar dataset (opcional)
dataset_path = os.path.join(folder_path, 'flood_risk_dataset.csv')
data.to_csv(dataset_path, index=False)
print(f"Dataset simulado criado e salvo em {dataset_path}")

# 2. Treinar Modelo de Classificação (Árvore de Decisão)
X = data[['distancia_cm']]  # Features
y = data['nivel_risco']     # Target

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Criar e treinar o modelo
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Avaliar o modelo (opcional)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo no conjunto de teste: {accuracy:.2f}")

# 3. Salvar o Modelo Treinado
model_filename = os.path.join(folder_path, 'flood_risk_model.joblib')
joblib.dump(model, model_filename)
print(f"Modelo treinado salvo em {model_filename}")
