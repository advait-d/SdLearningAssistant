# SD Learning Assistant — Frontend Integration Context

> This document is the **single source of truth** for any LLM or developer
> writing frontend code that connects to this backend. Read every section
> before generating any code. Do not invent or assume details not
> described here.

---

## 1. What This App Does

The **SD Learning Assistant** is an AI-powered backend that answers software
engineering questions in three modes:

| Mode | When it activates | What the UI should show |
|---|---|---|
| **Concept Explanation** | User asks "what is X" or "how does Y work" | Plain prose answer, no special badge |
| **System Design** | User asks to design a system at scale | Structured markdown with `##` headings |
| **Design Review** | User shares their own design and asks for feedback | Structured markdown critique |
| **Out of Scope** | Unrelated topic | Polite refusal message, no retry needed |

The backend also runs an **evaluator** after every LLM response. If confidence
is low the evaluator replaces the answer with 2–3 clarification questions
(a "fallback"). The UI must handle this case distinctly.

---

## 2. Backend Base URL

```
Production:  https://<your-backend>.railway.app   (or wherever deployed)
Local dev:   http://localhost:8000
```

Store this in an environment variable:

```js
// .env.local (Next.js) or Vite .env
NEXT_PUBLIC_API_URL=https://<your-backend>.railway.app
VITE_API_URL=https://<your-backend>.railway.app
```

CORS is pre-configured for all `*.vercel.app` origins and `localhost:3000` /
`localhost:5173`. No proxy or API rewrite is needed.

---

## 3. API Reference

### 3.1  POST `/api/v1/chat`  ← the only endpoint you need

**Request headers:**
```
Content-Type: application/json
```

**Request body:**
```jsonc
{
  "message":    "Design a URL shortener for 1 billion users.",  // string, 1–4000 chars
  "session_id": "user-abc123"  // string, 1–128 chars — YOU generate and persist this
}
```

**Success response — HTTP 200:**
```jsonc
{
  "response":          "## System Architecture\n...",  // markdown string
  "intent":            "SYSTEM_DESIGN_QUESTION",       // see §4
  "confidence_score":  0.87,                           // float 0.0–1.0, null if OUT_OF_SCOPE
  "is_fallback":       false,                          // true = clarification questions returned
  "session_id":        "user-abc123"                   // echo of your session_id
}
```

**Error response — HTTP 422 (validation) or 500 (server error):**
```jsonc
{
  "detail":     "An unexpected internal error occurred.",
  "error_code": "INTERNAL_ERROR"   // may be null
}
```

**HTTP 422** means your request body was malformed (empty message, session_id
too long, etc.). Show a local validation error — do not retry.

**HTTP 500** is a backend fault. Show "Something went wrong, please try again"
and allow a retry.

### 3.2  GET `/health`

Returns `{ "status": "ok", "uptime_seconds": 42.1, "version": "1.0.0" }`.
Use on startup to verify backend reachability before showing the UI.

---

## 4. Intent Labels — What Each Means for the UI

```
CONCEPT_EXPLANATION    → render response as normal markdown chat bubble
SYSTEM_DESIGN_QUESTION → render response with section headers (see §5)
DESIGN_REVIEW          → render response with section headers (see §5)
OUT_OF_SCOPE           → render as a warning/info bubble, no section parsing
```

You receive the intent in `response.intent`. Use it to choose a rendering
strategy, not to branch on content.

---

## 5. Response Markdown Structure

The LLM always returns **GitHub-flavoured markdown**. The section headings
differ by intent:

### SYSTEM_DESIGN_QUESTION responses always contain:

```markdown
## System Architecture
(high-level architecture prose)

## Key Components
(bullet list of components + responsibilities)

## Trade-offs
(pros and cons)
```

### DESIGN_REVIEW responses always contain:

```markdown
## Strengths
(bullet list)

## Weaknesses
(bullet list)

## Recommendations
(actionable advice)

## Security Review
(security concerns)
```

### CONCEPT_EXPLANATION responses

Free-form markdown. No guaranteed heading structure — render as-is.

### Fallback (is_fallback === true) responses

Plain prose + a bulleted list of 2–3 clarification questions. Example:

```
To give you the best system design advice, I need to understand your
requirements better. Please clarify the following:
* What is the expected read-to-write ratio?
* Do you need global distribution or single-region?
```

Render these with a distinct visual treatment (e.g. yellow/amber background,
a "🤔 Needs Clarification" badge) so the user knows to answer the questions.

---

## 6. Session Management

The `session_id` links conversation turns so the backend remembers context.

**Rules:**
- Generate **once per browser session** using `crypto.randomUUID()`.
- Persist in `sessionStorage` (cleared when tab closes) — do NOT use
  `localStorage` (sessions should not survive browser restarts).
- Send the **same** `session_id` on every message in the tab.
- Never expose the `session_id` to the user.

```js
function getOrCreateSessionId() {
  const KEY = 'sd_assistant_session_id';
  let id = sessionStorage.getItem(KEY);
  if (!id) {
    id = crypto.randomUUID();
    sessionStorage.setItem(KEY, id);
  }
  return id;
}
```

The backend keeps the **last 5 turns** (10 messages) per session. After that,
older messages are silently dropped. There is no explicit "reset session" API
— clearing sessionStorage and generating a new UUID starts a fresh session.

---

## 7. Loading States

The backend response time is **2–8 seconds** (LLM call + evaluation pass).
You MUST implement a loading state.

Required states:

| State | Trigger | UI treatment |
|---|---|---|
| `idle` | No request in flight | Input enabled, send button active |
| `loading` | Request sent, no response yet | Input disabled, send button shows spinner, add a "thinking" bubble to the message list |
| `success` | 200 received | Render response bubble, return to idle |
| `error` | Non-200 or network failure | Show error banner, re-enable input for retry |

**Do not** poll the `/health` endpoint during a request. One fetch per user
message is sufficient.

---

## 8. Rendering Markdown

The `response` field is always a markdown string. Use a library to render it:

- **React:** `react-markdown` with `remark-gfm` for tables and task lists
- **Vue:** `vue-markdown-render`
- **Vanilla JS:** `marked.js` + `DOMPurify` to sanitise before `innerHTML`

Code blocks inside responses should be syntax-highlighted
(`highlight.js` or Prism).

---

## 9.  UI Badges to Render

Based on the response fields, show contextual badges alongside each assistant
message:

```js
// Pseudocode
if (response.intent === 'OUT_OF_SCOPE') {
  showBadge('🚫 Out of Scope', 'gray');
} else if (response.is_fallback) {
  showBadge('🤔 Needs Clarification', 'amber');
} else if (response.confidence_score !== null) {
  if (response.confidence_score >= 0.8) showBadge('✅ High Confidence', 'green');
  else if (response.confidence_score >= 0.6) showBadge('⚠️ Medium Confidence', 'yellow');
  else showBadge('⚠️ Low Confidence', 'orange');
}
```

`confidence_score` is `null` for OUT_OF_SCOPE — do not render a confidence
badge in that case.

---

## 10. Complete Fetch Function (JavaScript)

This is the canonical implementation. Use it verbatim or adapt it to your
framework's data-fetching pattern:

```js
const API_BASE = process.env.NEXT_PUBLIC_API_URL  // or import.meta.env.VITE_API_URL
              || 'http://localhost:8000';

/**
 * Send a message to the SD Learning Assistant.
 *
 * @param {string} message     - The user's question or design brief.
 * @param {string} sessionId   - Persistent session identifier (see §6).
 * @param {AbortSignal} signal - Optional AbortController signal for cancellation.
 * @returns {Promise<ChatResponse>}
 *
 * @typedef {Object} ChatResponse
 * @property {string}      response          - Markdown string
 * @property {string}      intent            - Intent label (see §4)
 * @property {number|null} confidence_score  - 0.0–1.0 or null
 * @property {boolean}     is_fallback       - True if clarification questions returned
 * @property {string}      session_id        - Echo of your session_id
 */
async function sendMessage(message, sessionId, signal) {
  const response = await fetch(`${API_BASE}/api/v1/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, session_id: sessionId }),
    signal,
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: 'Unknown error' }));
    const httpError = new Error(err.detail || `HTTP ${response.status}`);
    httpError.status = response.status;
    httpError.code   = err.error_code || null;
    throw httpError;
  }

  return response.json(); // resolves to ChatResponse shape above
}
```

### Calling it from a React component

```jsx
import { useState, useRef, useCallback } from 'react';

function useChatSend(sessionId) {
  const [status, setStatus]   = useState('idle');   // 'idle' | 'loading' | 'error'
  const [error, setError]     = useState(null);
  const abortRef              = useRef(null);

  const send = useCallback(async (message, onSuccess) => {
    // Cancel any in-flight request
    abortRef.current?.abort();
    abortRef.current = new AbortController();

    setStatus('loading');
    setError(null);

    try {
      const data = await sendMessage(message, sessionId, abortRef.current.signal);
      setStatus('idle');
      onSuccess(data);
    } catch (err) {
      if (err.name === 'AbortError') return; // user cancelled, stay loading
      setStatus('error');
      setError(err.message);
    }
  }, [sessionId]);

  return { status, error, send };
}
```

---

## 11. Environment & CORS Notes

- The backend already permits `*.vercel.app` via regex (`allow_origin_regex`).
  You do **not** need to set `credentials: 'include'` — cookies are not used.
- If your Vercel project uses a custom domain, add it to the `CORS_ORIGINS`
  environment variable on the backend (comma-separated).
- For local development with `localhost:3000` or `localhost:5173` — both are
  already in the allow list. No proxy configuration needed.

---

## 12. Do Not Do These Things

| Anti-pattern | Why |
|---|---|
| Polling `/health` during a request | Unnecessary; one fetch per message is enough |
| Storing `session_id` in `localStorage` | Sessions should reset on browser close |
| Rendering `response` as raw `innerHTML` without sanitising | XSS risk |
| Sending empty `message` | Backend returns HTTP 422 — validate client-side first |
| Showing `confidence_score` for OUT_OF_SCOPE | It will be `null` — guard before rendering |
| Hard-coding `http://localhost:8000` in production | Use env variable |
