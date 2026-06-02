# Code Auditor — Plugin VS Code Frontend Plan

> Stack: React 18 + TypeScript + Vite + Tailwind + shadcn/ui + Zustand + TanStack Query  
> Architecture: Extension VS Code (src/) + React WebView (webview-ui/)  
> Backend: `http://localhost:8765` — voir `BACKEND_API_REFERENCE.md`

---

## 1. Structure des Dossiers

```
plugin_code_auditor/
├── src/                          # Extension VS Code (Node.js)
│   ├── extension.ts              # Point d'entrée, activation
│   ├── api/
│   │   ├── backendClient.ts      # Client HTTP base (fetch + auth header)
│   │   ├── chatClient.ts         # /api/chat, /api/chat/stream, /api/chat/apply
│   │   ├── gitClient.ts          # /api/git/*
│   │   ├── ciClient.ts           # /api/ci/*, /api/cd/*
│   │   ├── watchClient.ts        # /watch/*, /ws WebSocket
│   │   └── authClient.ts         # Stockage SecretStorage + token
│   ├── providers/
│   │   ├── InlineCompletionProvider.ts   # /api/chat/complete/inline
│   │   ├── DiagnosticsProvider.ts        # Issues → Problems panel
│   │   ├── CodeLensProvider.ts           # Proactive suggestions → CodeLens
│   │   └── StatusBarProvider.ts          # Backend status bar item
│   ├── commands/
│   │   ├── chatCommands.ts       # explain, complete, generate
│   │   ├── applyCommands.ts      # apply patch, workspace edit
│   │   ├── gitCommands.ts        # commit-msg, branch readiness
│   │   ├── ciCommands.ts         # ci analyze, cd score
│   │   └── testCommands.ts       # generate tests
│   ├── webview/
│   │   ├── WebviewHost.ts        # Crée et gère le WebviewPanel
│   │   └── messageProtocol.ts    # Types WebView ↔ Extension
│   └── utils/
│       ├── activeEditorContext.ts # Lire cursor_line, active_function, selected_text
│       ├── workspaceEdit.ts       # Convertir workspace_edit → vscode.WorkspaceEdit
│       └── markdown.ts            # Sanitize markdown pour notifications
│
├── webview-ui/                   # React App (chargée dans WebviewPanel)
│   ├── src/
│   │   ├── App.tsx               # Router principal + Layout 3 colonnes
│   │   ├── main.tsx
│   │   ├── api/
│   │   │   └── backendApi.ts     # Calls directs HTTP depuis React
│   │   ├── components/           # Composants UI réutilisables
│   │   │   ├── Layout/
│   │   │   │   ├── MainLayout.tsx
│   │   │   │   ├── TopBar.tsx
│   │   │   │   ├── Sidebar.tsx
│   │   │   │   └── RightPanel.tsx
│   │   │   ├── ui/               # shadcn/ui + custom
│   │   │   │   ├── Badge.tsx
│   │   │   │   ├── Card.tsx
│   │   │   │   ├── Button.tsx
│   │   │   │   ├── StatusDot.tsx
│   │   │   │   └── MarkdownRenderer.tsx
│   │   │   └── shared/
│   │   │       ├── BackendStatusBanner.tsx
│   │   │       ├── ErrorBoundary.tsx
│   │   │       └── LoadingSpinner.tsx
│   │   ├── features/
│   │   │   ├── auth/             # Login / Connect Backend
│   │   │   ├── chat/             # Chat Assistant (module principal)
│   │   │   ├── watch/            # Watch Mode + Realtime Feed
│   │   │   ├── git/              # Smart Git
│   │   │   ├── cicd/             # CI/CD Intelligence
│   │   │   ├── tests/            # Test Generation
│   │   │   ├── proactive/        # Proactive Suggestions Panel
│   │   │   └── settings/         # Settings Page
│   │   ├── store/
│   │   │   ├── authStore.ts
│   │   │   ├── settingsStore.ts
│   │   │   ├── chatStore.ts
│   │   │   ├── watchStore.ts
│   │   │   ├── gitStore.ts
│   │   │   ├── ciStore.ts
│   │   │   └── uiStore.ts
│   │   ├── hooks/
│   │   │   ├── useHealth.ts
│   │   │   ├── useWebSocket.ts
│   │   │   ├── useStreamingChat.ts
│   │   │   ├── useProactive.ts
│   │   │   └── useEditorContext.ts
│   │   └── types/
│   │       ├── chat.ts
│   │       ├── git.ts
│   │       ├── ci.ts
│   │       └── watch.ts
│   ├── package.json
│   └── vite.config.ts
│
├── package.json                  # Extension manifest + contributes
├── tsconfig.json
└── webpack.config.js
```

---

## 2. Layout Principal (3 colonnes)

```
┌─────────────────────────────────────────────────────────────────┐
│  TopBar: Code Auditor AI  •  🟢 Backend Ready  •  ⚙️  [User]   │
├───────────────┬─────────────────────────────────┬───────────────┤
│  Left Sidebar │  Main Workspace                 │  Right Panel  │
│               │                                 │               │
│  🤖 Chat      │  ← Vue active selon navigation  │  Context      │
│  👁 Watch     │                                 │  - Target file│
│  🌿 Smart Git │                                 │  - RAG docs   │
│  🔄 CI/CD     │                                 │  - Metrics    │
│  🧪 Tests     │                                 │  - Actions    │
│  💡 Proactive │                                 │               │
│  📋 History   │                                 │               │
│  ⚙️  Settings  │                                 │               │
└───────────────┴─────────────────────────────────┴───────────────┘
```

---

## 3. Communication Extension ↔ WebView

```typescript
// messageProtocol.ts

// React WebView → Extension VS Code
type WebviewToExtension =
  | { type: "chat.applyPatch";    payload: { workspace_edit: any } }
  | { type: "editor.getContext";  payload: {} }
  | { type: "settings.save";      payload: Settings }
  | { type: "auth.storeToken";    payload: { token: string; url: string } }
  | { type: "openFile";           payload: { path: string; line?: number } }
  | { type: "notification";       payload: { level: string; message: string } };

// Extension VS Code → React WebView
type ExtensionToWebview =
  | { type: "editor.context";     payload: EditorContext }
  | { type: "auth.state";         payload: AuthState }
  | { type: "watch.event";        payload: WatchEvent }
  | { type: "settings.state";     payload: Settings }
  | { type: "health.state";       payload: HealthState };

// EditorContext envoyé automatiquement à chaque changement de fichier/curseur
interface EditorContext {
  file_path:       string;
  cursor_line:     number;
  active_function: string;
  selected_text:   string;
  visible_range:   [number, number];
  language:        string;
}
```

---

## 4. Module Auth / Connexion Backend

### Deux modes
- **Local Dev** : Pas de compte. URL `http://127.0.0.1:8765`, token optionnel.
- **Remote Team** : Bearer token requis (future JWT).

### Stockage sécurisé (extension.ts)
```typescript
// JAMAIS dans localStorage WebView
await context.secrets.store("codeAuditor.backendUrl", url);
await context.secrets.store("codeAuditor.authToken", token);
```

### Écran de connexion
```
┌─ Connect to Backend ──────────────────────────┐
│                                               │
│  Backend URL:  [http://localhost:8765       ] │
│  Auth Token:   [Optional ────────────────── ] │
│                                               │
│  [Test Connection]   → 🟢 Backend v8.0.0     │
│                                               │
│  [Connect]                                    │
└───────────────────────────────────────────────┘
```

---

## 5. Module Chat Assistant

### Endpoints utilisés
| Action | Endpoint |
|--------|----------|
| Q&A / Explication | `POST /api/chat/stream` (SSE) |
| Complétion fonction | `POST /api/chat/complete` |
| Génération classe | `POST /api/chat/generate` |
| Historique | `GET /api/chat/history/{session_id}` |
| Mémoire sémantique | `POST /api/chat/memory/semantic` |

### Parser SSE (hook `useStreamingChat`)
```typescript
// Événements reçus dans l'ordre
// data: {"type":"status","content":"Planification..."}
// data: {"type":"plan","intent":"explain","context_level":"fast"}
// data: {"type":"token","content":"Cette "}
// data: {"type":"code","content":"def foo():\n ..."}
// data: {"type":"done","session_id":"...","elapsed_seconds":2.1}
// data: {"type":"error","content":"..."}
```

### Composants React
```
features/chat/
├── ChatView.tsx           # Layout + orchestration
├── ChatMessageList.tsx    # Liste des messages
├── MessageBubble.tsx      # Bulle user/assistant avec markdown
├── StreamingMessage.tsx   # Tokens affichés au fil du streaming
├── Composer.tsx           # Input + quick actions
├── CodeBlockActions.tsx   # Copy | Apply | Regenerate sur chaque code block
├── ApplyPatchModal.tsx    # Preview diff avant d'appliquer
└── SessionHistory.tsx     # Sidebar sessions
```

### Payload à envoyer (injecter le contexte éditeur)
```typescript
const payload = {
  message:         userInput,
  project_path:    settings.projectPath,
  session_id:      currentSessionId,
  target_file:     editorContext.file_path,
  cursor_line:     editorContext.cursor_line,
  active_function: editorContext.active_function,
  selected_text:   editorContext.selected_text,
  visible_range:   editorContext.visible_range,
};
```

---

## 6. Module Apply Patch

### Flux
```
Code généré dans le chat
→ Bouton "Apply" sur le code block
→ POST /api/chat/apply  { write_mode: "dry_run" }
→ Réponse: { diff, workspace_edit, valid, errors }
→ DiffPreviewModal.tsx  (affiche le diff Monaco)
→ Bouton "Confirm Apply"
→ postMessage({ type: "chat.applyPatch", payload: { workspace_edit } })
→ Extension.ts reçoit → convertit → vscode.workspace.applyEdit(...)
→ Toast "Applied ✓"
```

### Code extension pour appliquer
```typescript
// workspaceEdit.ts
export async function applyWorkspaceEdit(payload: any) {
  const edit = new vscode.WorkspaceEdit();
  for (const fe of payload.edits) {
    const uri = vscode.Uri.parse(fe.uri);
    for (const e of fe.edits) {
      const range = new vscode.Range(
        e.range.start.line, e.range.start.character,
        e.range.end.line,   e.range.end.character
      );
      edit.replace(uri, range, e.newText);
    }
  }
  await vscode.workspace.applyEdit(edit);
}
```

---

## 7. Module Watch Mode

### Endpoints utilisés
| Action | Endpoint |
|--------|----------|
| Démarrer | `POST /watch/start` |
| Arrêter | `POST /watch/stop` |
| Statut | `GET /watch/status` |
| Événements | `WS /ws` |
| Analyse fichier | `POST /analyze/file` |

### WebSocket (`useWebSocket` hook)
```typescript
// Événements reçus
// { type: "watch_event",    file, strategy, language, raw_analysis }
// { type: "file_deleted",   file }
// { type: "analysis_result",file, strategy, skipped }
// { type: "error",          detail }
// { type: "connected",      version, clients }

// Mettre à jour watchStore + afficher dans realtime feed
```

### Composants React
```
features/watch/
├── WatchView.tsx          # Dashboard principal
├── WatchControls.tsx      # Start/Stop + statut
├── RealtimeEventFeed.tsx  # Liste scrollable des événements
├── IssueTable.tsx         # Issues détectées
└── AnalysisDrawer.tsx     # Affiche raw_analysis en Markdown
```

---

## 8. Module Smart Git

### Endpoints utilisés
| Action | Endpoint | Réponse clé |
|--------|----------|-------------|
| Statut session | `POST /api/git/status` | `session_snapshot` |
| Readiness branche | `POST /api/git/branch` | `branch_report` |
| Commit message | `POST /api/git/commit-msg` | `commit_message` |
| Conflits | `POST /api/git/conflicts` | `conflict_report` |
| Review PR | `POST /api/git/pr/review` | `pr_report` |
| Readiness PR | `POST /api/git/pr/readiness` | `readiness_report` |

### Composants React
```
features/git/
├── GitView.tsx               # Layout onglets
├── GitSessionCard.tsx        # risk_score + modified_files
├── CommitMessageGenerator.tsx # Bouton → commit_message → Copy
├── BranchReadiness.tsx       # Form branch/base → verdict
├── ConflictPanel.tsx         # conflict_report en markdown
├── PRReviewPanel.tsx         # owner/repo/pr_number → pr_report
└── PRReadinessPanel.tsx      # ready/blockers/score
```

### Payload uniforme Git
```typescript
// Tous les endpoints Git acceptent project_path + session_id
const gitBase = { project_path: settings.projectPath, session_id };
// PR endpoints ajoutent: owner, repo, pr_number
```

---

## 9. Module CI/CD

### Endpoints utilisés
| Action | Endpoint |
|--------|----------|
| Analyser run CI | `POST /api/ci/analyze` |
| Démarrer polling | `POST /api/ci/poll/start` |
| Arrêter polling | `POST /api/ci/poll/stop` |
| Statut polling | `GET /api/ci/poll/status` |
| Score release | `POST /api/cd/score` |
| Statut env | `POST /api/cd/status` |

### Verdict CD + couleurs
```typescript
const verdictColor = {
  "DEPLOY_OK":      "green",   // score >= 80
  "DEPLOY_WARN":    "yellow",  // score 60-79
  "DEPLOY_BLOCKED": "red",     // score < 60 ou blocking_reasons
};
```

### Composants React
```
features/cicd/
├── CiCdView.tsx          # Layout onglets CI / CD
├── CiAnalyzeForm.tsx     # repo + run_id + pr_number
├── CiRunResult.tsx       # outcome + failure_type + root_cause
├── CiPollingControl.tsx  # Start/Stop polling + statut
├── ReleaseScoreCard.tsx  # Score gauge + component_scores
├── DeployStatusPanel.tsx # last_success + recent deploys
└── QualityGateCard.tsx   # sonar score + blocking_reasons
```

---

## 10. Module Proactive Suggestions

### Endpoint utilisé
`POST /api/chat/proactive` → `{ suggestions, has_critical, total }`

### Règles d'affichage
```
severity: "critical" → vscode.window.showErrorMessage (notification rouge)
severity: "warning"  → CodeLens + badge dans ProactivePanel
severity: "info"     → ProactivePanel uniquement
```

### Appel automatique
```typescript
// Dans extension.ts, à chaque changement de fichier actif
vscode.window.onDidChangeActiveTextEditor(async editor => {
  if (!editor) return;
  const result = await ciClient.proactive({
    project_path: settings.projectPath,
    target_file:  editor.document.fileName,
  });
  codeLensProvider.update(result.suggestions);
  if (result.has_critical) {
    vscode.window.showWarningMessage(
      `⚠️ ${result.total} proactive suggestion(s) — voir le panel`
    );
  }
});
```

### Composants React
```
features/proactive/
├── ProactivePanel.tsx    # Liste des suggestions
├── SuggestionCard.tsx    # type + severity + message + action button
└── SuggestionToast.tsx   # Version compacte pour notifications
```

---

## 11. Inline Completion Provider

```typescript
// src/providers/InlineCompletionProvider.ts
export class InlineCompletionProvider
  implements vscode.InlineCompletionItemProvider {

  private debounceTimer: NodeJS.Timeout | undefined;
  private readonly DEBOUNCE_MS = 400;
  private readonly TIMEOUT_MS  = 2000;

  async provideInlineCompletionItems(
    document: vscode.TextDocument,
    position: vscode.Position,
  ): Promise<vscode.InlineCompletionList> {
    return new Promise((resolve) => {
      clearTimeout(this.debounceTimer);
      this.debounceTimer = setTimeout(async () => {
        const prefix = document.getText(
          new vscode.Range(new vscode.Position(0, 0), position)
        );
        const suffix = document.getText(
          new vscode.Range(position, document.lineAt(document.lineCount - 1).range.end)
        );

        try {
          const result = await chatClient.inlineComplete({
            prefix_code:  prefix.slice(-1500),
            suffix_code:  suffix.slice(0, 300),
            language:     document.languageId,
            file_path:    document.fileName,
            project_path: settings.projectPath,
            cursor_line:  position.line,
            use_rag:      false, // false par défaut pour la perf
          });

          if (result.completion && result.confidence > 0.6) {
            resolve({
              items: [new vscode.InlineCompletionItem(result.completion)]
            });
          } else {
            resolve({ items: [] });
          }
        } catch {
          resolve({ items: [] });
        }
      }, this.DEBOUNCE_MS);
    });
  }
}
```

---

## 12. Settings Page

### Sections et champs

**Backend**
- Server URL (défaut `http://localhost:8765`)
- Test Connection → badge 🟢/🔴
- Auto-start backend (chemin Python)

**Security**
- Mode Local vs Remote
- Auth Token (masked, stocké SecretStorage)
- GitHub Token status
- [Clear All Secrets]

**AI**
- Streaming on/off
- Inline completion on/off
- Proactive suggestions on/off
- Show RAG sources
- Langue des réponses (FR/EN)

**Git**
- Default base branch (main/master)
- Default owner/repo pour PR
- Safe mode always on (défaut: true)

**CI/CD**
- GitHub repo (owner/repo)
- SonarCloud project key
- Environment (production/staging)
- Polling interval (secondes)

**UI**
- Theme (Dark/Light/VS Code)
- Compact mode
- Sidebar collapsed by default

---

## 13. Design System (Tokens existants de CHAT_UI_DESIGN.md)

```css
/* Couleurs */
--color-primary:   #4F9CF9;  /* Bleu — question/chat */
--color-success:   #10B981;  /* Vert — génération OK */
--color-danger:    #EF4444;  /* Rouge — risque/critique */
--color-warning:   #F59E0B;  /* Jaune — warning */
--color-info:      #8B5CF6;  /* Violet — info/proactive */
--color-bg:        #0F1117;  /* Background dark */
--color-surface:   #1A1F2E;  /* Cards */
--color-border:    #2A3042;  /* Borders */

/* Typography */
--font-size-h1:   24px;
--font-size-h2:   20px;
--font-size-body: 14px;
--font-size-sm:   13px;
--font-family:    'Inter', system-ui, sans-serif;

/* Spacing */
--space-xs:  4px;
--space-sm:  8px;
--space-md: 16px;
--space-lg: 24px;
--space-xl: 32px;

/* Border radius */
--radius-sm:  4px;
--radius-md:  8px;
--radius-lg: 12px;
--radius-xl: 16px;
```

---

## 14. Roadmap d'Implémentation

### Phase 0 — Scaffold (3 jours)
- [ ] `npx create-vscode-ext` + React Vite WebView
- [ ] Configurer Tailwind + shadcn/ui
- [ ] Implémenter `messageProtocol.ts`
- [ ] Créer `backendClient.ts` typé
- [ ] `SettingsStore` + `AuthStore` (SecretStorage)
- [ ] `GET /health` → `useHealth` hook

### Phase 1 — Shell Professionnel (4 jours)
- [ ] `MainLayout` 3 colonnes responsive
- [ ] `TopBar` avec backend status
- [ ] `Sidebar` navigation
- [ ] Page `Settings`
- [ ] Page `Login / Connect Backend`
- [ ] Error boundary + loading states

**✅ Livrable : Extension qui s'ouvre, affiche le statut backend, settings, navigation**

### Phase 2 — Chat Streaming (4 jours)
- [ ] `ChatView` + `ChatMessageList`
- [ ] SSE parser dans `useStreamingChat`
- [ ] `StreamingMessage` (tokens live)
- [ ] `MarkdownRenderer` + syntax highlight
- [ ] `CodeBlockActions` (Copy / Apply)
- [ ] Injection contexte éditeur (cursor_line, selected_text)
- [ ] `SessionHistory`

**✅ Livrable : Chat pro, streaming réel, contexte IDE**

### Phase 3 — Apply Patch (3 jours)
- [ ] `DiffPreviewModal` avec Monaco diff editor
- [ ] `POST /api/chat/apply` → `workspace_edit`
- [ ] `workspaceEdit.ts` → `vscode.workspace.applyEdit`
- [ ] `/apply/new-file` et `/apply/multi`
- [ ] Toast confirmation + feedback accepted/rejected

**✅ Livrable : Code applicable sans copier-coller**

### Phase 4 — Watch + Proactive (3 jours)
- [ ] WebSocket `/ws` → `useWebSocket` hook
- [ ] `WatchView` + `RealtimeEventFeed`
- [ ] Watch start/stop/status
- [ ] `POST /api/chat/proactive` au changement de fichier
- [ ] `CodeLensProvider` pour warnings
- [ ] Notifications VS Code pour critiques

**✅ Livrable : Assistant temps réel actif**

### Phase 5 — Smart Git (3 jours)
- [ ] `GitView` avec onglets
- [ ] `GitSessionCard` (risk_score)
- [ ] `CommitMessageGenerator`
- [ ] `BranchReadiness` + `PRReadinessPanel`
- [ ] `ConflictPanel` dry-run

**✅ Livrable : Smart Git utilisable depuis le plugin**

### Phase 6 — CI/CD (3 jours)
- [ ] `CiAnalyzeForm` + `CiRunResult`
- [ ] `CiPollingControl`
- [ ] `ReleaseScoreCard` (gauge visuelle)
- [ ] `DeployStatusPanel`

**✅ Livrable : DevOps intelligence dans VS Code**

### Phase 7 — Tests + Inline (2 jours)
- [ ] `TestGenerationView` + preview
- [ ] `InlineCompletionProvider` (debounce 400ms)
- [ ] Status bar latency

### Phase 8 — Polish Production (3 jours)
- [ ] Keyboard shortcuts
- [ ] Accessibilité (ARIA)
- [ ] Theme compatibility VS Code
- [ ] Packaging `.vsix`

---

## 15. MVP Prioritaire à Livrer en Premier

```
MVP v0.1 — Dev Assistant Console
─────────────────────────────────
✅ Login / Connect Backend
✅ Health check display
✅ Chat streaming avec contexte éditeur
✅ Apply patch (dry_run → workspace.applyEdit)
✅ Watch WebSocket + realtime feed
✅ Proactive suggestions → notifications
✅ Git status + commit message generator
✅ Settings (token SecretStorage, URL)

❌ Pas encore : inline completion, CI/CD complet,
               multi-file refactor, team dashboard
```

---

## 16. Sécurité — Règles Non-Négociables

| Règle | Implémentation |
|-------|---------------|
| Tokens jamais en localStorage | `context.secrets` (SecretStorage) |
| CSP strict dans WebView | `meta http-equiv="Content-Security-Policy"` |
| Confirmation avant apply | `DiffPreviewModal` obligatoire |
| `write_mode: "dry_run"` par défaut | Jamais `apply` sans confirmation |
| Safe mode Git toujours activé | `safe_mode: true` dans tous les appels Git |
| Sanitize markdown | `DOMPurify` dans `MarkdownRenderer` |
| Validation paths fichiers | Déjà fait côté backend (containment check) |

---

*Plan frontend — Code Auditor Plugin v0.1*  
*Backend cible : v8.0.0 — voir `BACKEND_API_REFERENCE.md`*
