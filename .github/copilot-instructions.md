# GitHub Copilot/AI Agent Instructions for ARF Infrastructure

The **Agentic Reliability Framework (ARF)** repo implements an OSS‑only advisory engine for evaluating Azure infrastructure changes. The intelligence lives in `core/` and the "simulator" is the primary API. Enterprise code (MCP server, execution) is outside this repository.

## 🧱 High‑Level Architecture

- **`agentic_reliability_framework/core/`** – domain layer. Contains:
  - **`governance/intents.py`** – algebraic data types for intent requests (provision, config deploy, grant access). Pydantic models with validators; `InfrastructureIntent` is a `Union` discriminated by `intent_type`.
  - **`governance/policies.py`** – composable policy algebra (atomic/composite/`allow_all` helpers) and deterministic/probabilistic evaluators. Policy trees are built with `&`, `|`, `~` operators.
  - **`governance/cost_estimator.py`** – lookup table + YAML loader; `estimate_monthly_cost` with optional cost‑delta helpers. Caches with `lru_cache`.
  - **`governance/risk_engine.py`** – Bayesian scoring: conjugate beta priors (online) and optional HMC logistic regression (offline via `pymc`/`arviz`). `RiskEngine` returns `(score, explanation, contributions)`.
  - **`governance/healing_intent.py`** – immutable dataclass sent to Enterprise. OSS only marks results `OSS_ADVISORY_ONLY` via `mark_as_oss_advisory()` and enforces auditable fields.
  - **`governance/azure/azure_simulator.py`** – orchestrator. Glue that computes cost, policies, risk and returns a `HealingIntent`. This is the *public* entry point for most OSS workloads.

- **`agentic_reliability_framework/runtime/`** – generic runtime engine used by other ARF components. Contains `engine.py` which wires a `PolicyEngine` to events; not needed for most infrastructure work, but read to understand how analysis flows through the broader system.

- **`agentic_reliability_framework/core/config/constants.py`** – OSS boundary constants (`MAX_POLICY_VIOLATIONS`, `MCP_MODES_ALLOWED`, etc.). Many modules import these; respect them when adding features. Values are validated on import.

## 🧩 Key Patterns & Conventions

- **Immutability & Validation**
  - Domain objects: Pydantic (`Intent`) or frozen `@dataclass` (`HealingIntent`).
  - Validators (`@field_validator`) enforce business rules; look at `RESOURCE_SIZE_PATTERNS`, `VALID_AZURE_REGIONS` for examples.

- **Enums for fixed vocabularies** (e.g. `ResourceType`, `PermissionLevel`). If you add variants, update *_patterns/_maps in intents and tests.

- **Policy builder helpers**: use `allow_all()`, `deny_all()` or compose primitives. The deterministic `PolicyEvaluator` accepts a context dict (e.g. `'cost_estimate'`). Tests often call `allow_all()` to shortcut policy logic.

- **OSS limits**: always clip lists of violations to `MAX_POLICY_VIOLATIONS`. New public APIs should validate `max_policies` using `validate_infrastructure_config()`.

- **Package exports**: update `__all__` in `core/governance/__init__.py` when you add a new public symbol. That file defines the top‑level `from agentic_reliability_framework.core.governance import ...` API.

- **Serialization**: `HealingIntent` has custom `to_json()`/`from_json()` in the same file – use them when writing tests or persisting objects.

- **Imports** are always absolute, prefixed with `agentic_reliability_framework`.

## ⚙️ Developer Workflows

1. **Environment**
   ```sh
   conda env create -f environment.yml   # creates `arf` env
   conda activate arf
   pip install -e .
   ```
2. **Tests**
   - Run all infra tests: `pytest tests/infrastructure/`.
   - Additional tests in `tests/core/` and `tests/runtime/` exist for non‑infra functionality.
   - Pytest fixtures are used heavily; look at `tests/infrastructure/conftest.py` if added later.
   - Environment variable `PYTEST_CURRENT_TEST` is checked in `core/config/constants.py` to avoid expensive startup logic during tests.

3. **Building & Packaging**
   - Standard setuptools (`python -m build`) driven by `pyproject.toml`.
   - Use `pip install -e .` in development; the package name is `agentic-reliability-framework`.

4. **Debugging**
   - Most modules are plain Python; insert `import pdb; pdb.set_trace()` or use VS Code breakpoints.
   - The risk engine may spin up an HMC model; disable training during tests by mocking or by skipping heavy operations.

## 🔗 Integration & External Dependencies

- **Azure‑specific logic** lives in `core/governance/azure/*` but there are no real Azure SDK calls – it's a simulator. If you add real clients, keep them separated behind adapter interfaces under `core/adapters`.

- **Machine‑learning tooling**: `pymc` and `arviz` for HMC; `scikit-learn` only for `StandardScaler`. These libraries are heavy; tests mock or bypass training.

- **Policies, costs, risk scores** are loosely coupled via plain Python objects; no RPC or async boundaries. The simulator depends on them via composition.

- **Configuration files** (YAML pricing, optional) are loaded with `yaml.safe_load`. Paths are not normalized; tests provide fixtures to simulate missing files.

- **Claude API**: set `ANTHROPIC_API_KEY` environment variable. The code falls back to mock responses when absent; tests often patch the `claude_adapter`.

- **Hugging Face** integration is optional; some agents use the HF token. Provide `HF_TOKEN` env var if you run demos or build models locally.

- **HMC training**: run `RiskEngine.train_hmc(...)` programmatically or via the admin HTTP endpoint (`POST /api/v1/admin/train_hmc` in the server repo). The model writes `hmc_model.json` which is hot‑loaded on startup; ensure consumers handle a missing file gracefully.

- **Interactive demo**: A lightweight Gradio app exists in the companion Hugging Face Space (see README). To run locally:

  ```bash
  pip install gradio
  python app.py   # copy from the HF repo or examples
  ```
This is useful for manual experimentation but not required for core development.

## 🛠️ Project‑Specific Gotchas

- The `IntentSource` and `IntentStatus` enums have many values; adding new ones requires updating serialization and any downstream consumers (tests often assert on the string value).

- `HealingIntent.mark_as_oss_advisory()` mutates status/flags; use it when constructing intents in infrastructure modules.

- Policy combinators (`&`, `|`, `~`) create new policy objects; don't rely on identity equality in tests—compare violations instead.

- Risk score thresholds are hard‑coded in the simulator (`>0.8` deny, `>0.4` escalate). If you propose changing them, update the tests accordingly.

- **Environment string gotchas**: use literal strings "prod", "dev", etc. (validators in intents enforce this). Unit tests often fail when using the `Environment` alias instead of the actual value.

- **HealingIntent construction**: prefer factory helpers (`from_analysis`, `from_infrastructure_intent`) rather than building the dataclass manually; the helpers handle OSS metadata and hashing.

- **Cost estimation cache**: `estimate_monthly_cost` is `lru_cache`d using the intent object as part of its key. Modifying an intent after creation can silently change the cache behavior.

- **HMC model file**: `hmc_model.json` is read at startup. Code paths should handle its absence; training writes a new file.

- **Agent recommendations**: orchestrator weights results by `confidence`; make sure agent output dicts include a numeric `confidence` field.

## 📚 Learning & Updating the Risk Engine

- Online updates: call `RiskEngine.update_outcome(intent, success)` after each action to update the beta priors.

- Offline training: prepare a pandas DataFrame with columns like `outcome`, `hour`, `env_prod`, `user_role`, `category`, etc., and call `RiskEngine.train_hmc(df)`. Training uses PyMC/NUTS; disable in tests or mock for speed.

- Trained coefficients are saved to `hmc_model.json`; the engine hot‑loads them on initialization.

## 📁 Key Files to Reference

- `core/governance/risk_engine.py` – Bayesian logic and HMC integration.
- `core/governance/healing_intent.py` – OSS/Enterprise contract definitions.
- `runtime/engine.py` – main processing loop and orchestration.
- `tests/infrastructure/test_risk_engine.py` – example of testing risk outputs.

---

💡 **Feedback needed:** If any of the above sections are unclear or missing project‑specific details you'd expect, please ask so we can iterate on this instruction file.

