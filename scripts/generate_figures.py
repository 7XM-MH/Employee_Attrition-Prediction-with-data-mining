import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / 'docs' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR = ROOT / 'results'

# 1. ROC Curves
with open(RES_DIR / 'roc_data.json', 'r') as f:
    roc_data = json.load(f)

plt.figure(figsize=(9, 6))
for name, d in roc_data.items():
    auc_val = d.get('auc', 0.5)
    plt.plot(d['fpr'], d['tpr'], label=f"{name} (AUC = {auc_val:.3f})", lw=2)
plt.plot([0, 1], [0, 1], 'k--', label='Aléatoire (AUC = 0.500)')
plt.xlabel('Taux de Faux Positifs (FPR)')
plt.ylabel('Taux de Vrais Positifs (TPR / Recall)')
plt.title('Courbes ROC Comparatives — Benchmark 6 Modèles')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(FIG_DIR / 'roc_curves.png', dpi=200)
plt.close()

# 2. Confusion Matrices
with open(RES_DIR / 'confusion_matrices.json', 'r') as f:
    cms = json.load(f)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for i, (name, cm) in enumerate(cms.items()):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[i],
                xticklabels=['Reste (0)', 'Quitte (1)'], yticklabels=['Reste (0)', 'Quitte (1)'])
    axes[i].set_title(f'Matrice : {name}')
    axes[i].set_xlabel('Prédiction')
    axes[i].set_ylabel('Vérité Terrain')
plt.tight_layout()
plt.savefig(FIG_DIR / 'confusion_matrices.png', dpi=200)
plt.close()

# 3. Feature Importance
with open(RES_DIR / 'feature_importances.json', 'r') as f:
    fi = json.load(f)

rf_fi = pd.Series(fi['RandomForest']).sort_values(ascending=True)
plt.figure(figsize=(10, 7))
rf_fi.plot(kind='barh', color='#4F46E5')
plt.title('Importance des Variables — Forêt Aléatoire (Gini)')
plt.xlabel('Importance Relative')
plt.tight_layout()
plt.savefig(FIG_DIR / 'feature_importance.png', dpi=200)
plt.close()

print('Static figures generated in docs/figures/')
