# Guide de configuration : Claude Code avec le Proxy

Ce guide explique comment configurer l'outil CLI officiel **Claude Code** pour qu'il utilise ce proxy au lieu de l'API Anthropic officielle.

## Prérequis

1. Avoir ce proxy lancé et accessible (ex: `http://localhost:8000`).
2. Avoir installé `claude-code` (`npm install -g @anthropic-ai/claude-code`).

## Étape 1 : Lancer le Proxy

Assurez-vous que le proxy tourne :
```bash
python src/proxy_app/main.py
```
Notez l'URL affichée (par défaut `http://0.0.0.0:8000` ou `http://localhost:8000`).

## Étape 2 : Configurer Claude Code

Ouvrez un nouveau terminal et exécutez la commande suivante pour dire à Claude Code d'utiliser votre proxy :

```bash
claude config set base_url http://localhost:8000/v1
```

**⚠️ IMPORTANT :** N'oubliez pas le `/v1` à la fin de l'URL.

## Étape 3 : Authentification

Lors de la première utilisation (ex: `claude login`), Claude Code va vous demander de vous authentifier.

### Cas A : Authentification Activée (Recommandé)
Si votre proxy a une `PROXY_API_KEY` définie dans le fichier `.env` :
- Quand Claude Code demande une clé, entrez votre `PROXY_API_KEY`.

### Cas B : Pas d'Authentification
Si vous n'avez pas défini de `PROXY_API_KEY` :
- Entrez n'importe quelle chaîne de caractères (ex: `sk-dummy-key`) quand Claude Code vous le demande. Le proxy l'acceptera.

## Étape 4 : Utilisation

Vous pouvez maintenant utiliser Claude Code normalement !

```bash
claude "Crée un fichier hello.py qui affiche bonjour"
```

Le proxy interceptera les requêtes, les convertira au format OpenAI, et les routera vers vos fournisseurs configurés (OpenAI, Gemini, Groq, etc.) tout en renvoyant les réponses au format attendu par Claude Code.

## Dépannage

- **Erreur 404** : Vérifiez que vous avez bien mis `/v1` à la fin de l'URL (`http://localhost:8000/v1`).
- **Erreur 401** : Vérifiez que la clé que vous donnez à Claude correspond à votre `PROXY_API_KEY`.
- **Rien ne se passe** : Regardez les logs du terminal où tourne le proxy. Vous devriez voir "🤖 Received request from Claude Code CLI".
