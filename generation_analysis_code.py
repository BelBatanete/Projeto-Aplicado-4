
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Carregar os dados
model_performance = pd.read_csv('model_performance_final.csv')
climatic_correlation = pd.read_csv('climatic_correlation_final.csv')
seasonal_production = pd.read_csv('seasonal_production_final.csv')
error_by_region = pd.read_csv('error_by_region_final.csv')

# 2. Figura 1: Comparação entre os Modelos Preditivos
plt.figure(figsize=(10, 6))
x = np.arange(len(model_performance["Modelo"]))
width = 0.35
plt.bar(x - width/2, model_performance["Energia Solar (MAE)"], width, label="Energia Solar")
plt.bar(x + width/2, model_performance["Energia Eólica (MAE)"], width, label="Energia Eólica")
plt.xticks(x, model_performance["Modelo"])
plt.ylabel("Erro Médio Absoluto (MAE)")
plt.title("Comparação entre os Modelos Preditivos")
plt.legend()
plt.tight_layout()
plt.savefig('fig1_model_comparison.png')

# 3. Figura 2: Heatmap de Correlação
plt.figure(figsize=(8, 6))
sns.heatmap(climatic_correlation.set_index("Variável Climática"), annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
plt.title("Heatmap de Correlação entre Variáveis Climáticas e Produção de Energia")
plt.tight_layout()
plt.savefig('fig2_climatic_correlation.png')

# 4. Figura 3: Padrões Sazonais na Produção de Energia Solar
plt.figure(figsize=(10, 6))
months_numeric = range(1, 13)
plt.plot(months_numeric, seasonal_production["Produção Solar (MW)"], marker='o', label="Produção Solar")
plt.title("Padrões Sazonais na Produção de Energia Solar")
plt.xlabel("Mês")
plt.ylabel("Produção Solar (MW)")
plt.xticks(ticks=months_numeric, labels=seasonal_production["Mês"])
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('fig3_seasonal_patterns.png')

# 5. Figura 4: Erro de Previsão (MAE) por Região e Modelo
plt.figure(figsize=(12, 6))
sns.barplot(data=error_by_region, x="Modelo", y="Erro (MAE)", hue="Região", palette="Set2")
plt.title("Erro de Previsão (MAE) por Região e Modelo")
plt.ylabel("Erro Médio Absoluto (MAE)")
plt.xlabel("Modelo")
plt.legend(title="Região")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('fig4_error_by_region.png')
