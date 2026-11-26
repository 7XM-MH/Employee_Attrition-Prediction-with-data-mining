# Convention de contribution

Ce document s'applique à tous les lots du mini-projet *Employee Attrition*. Le référentiel d'organisation est [`docs/PLAN_PROJET.md`](docs/PLAN_PROJET.md).

## 1. Principe général

- Une branche `feature/lotN-*` par lot, créée depuis `upstream/main`.
- **Un seul commit et une seule PR par lot.** Les corrections demandées en revue sont amendées dans ce même commit avant tout nouveau push.
- La PR cible toujours `Anwar77-b:main`.
- La revue de `Anwar77-b` est obligatoire : demander le reviewer et ajouter un commentaire de demande de revue.
- L'auteur de la PR ne la merge pas. Le merge est effectué par le reviewer après validation.
- Aucun identifiant TASK (préfixe de tâche externe) n'est utilisé dans les branches, commits, fichiers ou messages de PR. Seuls les numéros de lot et de PR sont employés.

## 2. Nommage

| Élément | Convention | Exemple |
|---|---|---|
| Branche | `feature/lotN-<sujet-court>` | `feature/lot0-organisation` |
| Commit | `<type>: <description impérative courte>` | `docs: add project plan, contribution rules and fix anonymity in LICENSE` |
| Types autorisés | `docs:`, `feat:`, `test:`, `chore:` | `test: cover preprocessing contracts` |

La description du commit reste en minuscule après le deux-points, sans point final.

## 3. Anonymat

- Seuls des codes étudiants au format `IASDxx` sont autorisés dans les livrables.
- Aucun nom, prénom, matricule ni adresse personnelle n'est enregistré dans les fichiers suivis ni dans les comptes rendus.
- Les identifiants GitHub servent uniquement à la traçabilité des commits et à la demande de revue.

## 4. Périmètre protégé

Ne pas modifier, sauf lot explicitement dédié et accord du reviewer :

- `app/**`
- `.streamlit/**`
- `src/app.py`
- le code ML (`src/**`, `scripts/**`), les tests (`tests/**`) et les artefacts générés (`*.joblib`, `results/*.json`, `data/processed/*.csv`) pour le LOT 0.

## 5. Commandes de vérification

À exécuter depuis la racine du dépôt avant de demander la revue :

```bash
python -m pytest tests -q
```

Sous Windows avec l'environnement du projet :

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
git diff --check
git status --short --branch
```

Contrôles attendus :

- `pytest tests -q` : tous les tests passent (ou écart explicitement expliqué dans la PR) ;
- aucun marqueur temporaire ni ancien placeholder de spécialité dans les fichiers touchés ;
- tous les liens Markdown internes valides ;
- aucun fichier du périmètre protégé ni artefact généré dans `git status` ;
- `git diff --check` sans erreur.

## 6. Étapes d'une PR

1. `git fetch upstream` puis création de la branche depuis `upstream/main`.
2. Réalisation du lot, puis vérifications de la section 5.
3. Commit unique avec un message conventionnel.
4. Push de la branche et ouverture d'une seule PR vers `Anwar77-b:main`.
5. Demande de revue à `Anwar77-b` + commentaire de demande de revue.
6. Réponse aux remarques par amend du commit, sans merge de la part de l'auteur.
