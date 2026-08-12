# Configuration

All settings are environment variables prefixed `LEGEND_`, read from the environment or
from a `.env` file. [`.env.example`](../.env.example) is the annotated template; this page
is the reference.

**The defaults are not arbitrary, and several look wrong until you read why.** The
authoritative source is the comment on each field in `app/config.py`, which carries the
measurement that produced it — including the cases where the obvious value was tried and
measured to be worse. Read the comment before changing a default.


- `LEGEND_OLLAMA_HOST` — default `http://127.0.0.1:11434`
- `LEGEND_MAX_LOADED_MODELS` — default `4`, one slot per tier. **Must be set on the
  daemon, not here** — nothing in this process can change a running Ollama's cap.
  Measured to make no difference to throughput; lower it freely if RAM is tight.
- `LEGEND_DEFAULT_ROUTE` — used when every stage is unsure, default `chat`
- `LEGEND_DISABLE_CLASSIFIER` — skip stage 3, go straight to the default route
- `LEGEND_ROUTER_ALIAS` — which `models.yaml` alias fills the router role (trivial
  replies + stage-3 classifier), default `general`
- `LEGEND_ASSISTANT_NAME` — the assistant's name, default `Lucy`. Empty restores the
  anonymous persona, which tells the model to say it has no name rather than invent one.
- `LEGEND_DEFAULT_EFFORT` — `auto` (default), or pin every request to one level
- `LEGEND_VERIFY_ENABLED` — cross-model critic, default **false**; see Adjudication
- `LEGEND_SELF_CONSISTENCY` — answer twice and compare, default false
- `LEGEND_CRITIC_ALIAS` — which tier judges, default `think`
- `LEGEND_RETRIEVAL_ENABLED` — default true (a missing corpus is simply inert)
- `LEGEND_RETRIEVAL_DB` — index location, default `data/corpus.db`
- `LEGEND_RETRIEVAL_MIN_SCORE` — cosine cut-off, default `0.70`; calibrate with `--probe`
- `LEGEND_RETRIEVAL_TOP_K` — chunks injected, default 3
- `LEGEND_RETRIEVAL_CITE` — append the computed `Sources:` line, default true
- `LEGEND_READER_ROUTE` — which *route's* model reads retrieved text, default `chat`, so
  it tracks whatever that tier runs on rather than pinning an alias


Not listed above, and worth knowing about:

- `LEGEND_RETRIEVAL_MEMORY_MIN_SCORE` — cut-off for stored memories, default `0.55`.
  Lower than the document threshold on purpose: a memory is one sentence, and
  short-to-short similarity runs lower for the same relevance.
- `LEGEND_TOOLS_ENABLED` — default true.
- `LEGEND_TOOL_FAMILIES` — which families may ever be offered, default
  `["basics", "web", "notes"]`. This is the hard override; the gate is the automatic one.
- `LEGEND_TOOL_DISPATCHER_ALIAS` — the model that *picks* the tool, default `general`.
  Not the model that answers; see [architecture.md](architecture.md#tools).
- `LEGEND_SEARXNG_URL` — local SearXNG for the `web` family, default
  `http://127.0.0.1:8080`. Needs `json` under `search.formats` or every request 403s.
- `LEGEND_VAULT_PATH` — an Obsidian vault for the `notes` family. **The one setting with
  no sensible default**, because it is a path on your machine. The family reports itself
  unavailable while it is unset, which is a working state rather than an error.
- `LEGEND_PERSONA_CAPABILITIES` — whether the persona tells the model what it can look
  up, default false. Measured, and the answer was no; the flag stays so the comparison
  can be re-run.

## Model and route definitions

Two YAML files, both committed:

- **`models.yaml`** — the tiers: which GGUF each alias resolves to, residency
  (`keep_alive`), context size, per-tier sampling defaults, token budgets, and which
  persona length the tier gets. Parked models stay here commented out rather than
  deleted, so restoring one is uncommenting an entry and re-running
  `scripts/import_models.py`.
- **`routes.yaml`** — the routes, each with a description and the labeled examples that
  the embedding centroids are built from at startup, plus the similarity thresholds the
  cascade's stage 2 compares against.

Anything restored in `models.yaml` also needs its route re-added to `routes.yaml`.
