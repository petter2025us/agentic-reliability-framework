# ARF Infrastructure Governance Module (OSS)

Agentic Reliability Framework (ARF) is an open‑source advisory engine that
simulates governance decisions for cloud infrastructure. Its mission is to
provide **provably safe, mathematically grounded recommendations**—approve,
deny, or escalate—when users request provisioning, configuration, or access
changes. The OSS module contains the core intelligence; enterprise layers build
on it to enforce actions in real clouds.

This repository is the actively maintained OSS branch; the original
`petterjuan/agentic-reliability-framework` repo is frozen due to access
constraints. All future development happens here.

---

## 🚀 High‑Level Overview

ARF is structured as a **hybrid intelligence engine** coupling fast online
updates with offline pattern learning. A simulator takes an
`InfrastructureIntent` (e.g. provision a VM, deploy configuration, grant access)
and returns a `HealingIntent` recommendation. Internally it evaluates:

1. **Cost** using built‑in pricing tables or user‑supplied YAML.
2. **Policies** expressed as composable predicates.
3. **Risk score** from our Bayesian engine.
4. **Semantic context** retrieved from past incidents.

These pieces are assembled by `AzureInfrastructureSimulator` (or other cloud
adapters) and produce human‑readable justifications.

<img src="docs/architecture.png" alt="ARF architecture diagram" />

---

## 📘 Quick‑Start Tutorial

For a hands‑on introduction to ARF, check out the **[Tutorial](TUTORIAL.md)**.
It covers installation, core concepts, step‑by‑step examples, multi‑agent
analysis, and how to run the interactive demo.

---

## 🧠 Intelligence Engine

ARF’s core innovation lies in its **intelligence engine**, which blends
on‑the‑fly learning with deep offline analysis and contextual memory.

### Bayesian Online Learning with Conjugate Priors

To update risk continuously as outcomes arrive, ARF maintains a
Beta–Binomial model. For each intent category the engine tracks success/failure
counts \(s,f\) and maintains a posterior

\[
\text{Beta}(\alpha + s,\;\beta + f)
\]

with default prior \(\alpha=\beta=1\). The predicted risk is the posterior
mean

\[
\mathbb{E}[p]=\frac{\alpha + f}{\alpha+\beta+s+f}.
\]

Conjugacy makes updates trivial and ensures the system adapts in real time.

### Offline Pattern Discovery via HMC (NUTS)

Certain factors (time of day, user role, environment, etc.) exhibit complex,
nonlinear interactions that are hard to capture online. For these we train a
logistic regression using Hamiltonian Monte Carlo with the No‑U‑Turn Sampler:

\[
\Pr(\text{failure}\mid x) = \sigma(w_0 + w_{\text{role}} + w_{\text{env}}
  + w_{\sin\omega t} + w_{\cos\omega t} + \cdots),
\]

where the cyclical encoding \((\sin, \cos)\) of time \(t\) captures daily
rhythms. The HMC implementation (via PyMC3/ArviZ) learns a posterior over
weights; the model is serialized to `hmc_model.json` and hot‑loaded by the
simulator. The offline engine can be retrained periodically with new data.

### Hybrid Architecture

At lookup time the final risk score is a weighted blend of the online beta
estimate and the offline HMC prediction:

\[
\text{risk} = \lambda\cdot\text{risk}_{\text{online}}
 + (1-\lambda)\cdot\text{risk}_{\text{offline}},
\]

where \(\lambda\) is configurable (default 0.5). This hybrid design offers
fast adaptation and deep pattern recognition.

### Semantic Memory

Incidents are stored using sentence‑transformers to embed text, and FAISS
indexes for nearest‑neighbor retrieval. When an intent is evaluated the engine
fetches similar past incidents, providing context and precedent in the output.

### Deterministic Probability Thresholding (DPT)

We introduce **Deterministic Probability Thresholding**, a decision rule that
differs from Bayesian credible regions, frequentist \(p\)‑values, or fuzzy
logic. Rather than probabilistic intervals or arbitrary scores, DPT compares
the posterior probability directly against fixed thresholds:

- **approve** if \(P(\text{failure}) < \tau_{\text{low}}\),
- **deny** if \(P(\text{failure}) > \tau_{\text{high}}\),
- **otherwise escalate**.

The thresholds \(\tau_{\text{low/high}}\) are deterministic constants
(typically 0.2/0.8) that can be set per‑policy. This makes decisions
transparent, auditable, and immune to the “credibility paradox” of overlapping
intervals.

### Cyclical Time Encoding

Daily and weekly cycles are encoded using sine and cosine transforms

\[
\sin(2\pi t/24),\qquad \cos(2\pi t/24)
\]

to give the offline model sensitivity to time‑of‑day effects without
introducing discontinuities at midnight.

---

## 🔒 OSS vs Enterprise

The repository you are reading is the **OSS** slice, which implements all
advisory logic and exposes the public API. The enterprise layer (in a
different repo) wraps ARF with actual Azure/AWS/… clients, enforces decisions,
stores outcomes securely, and adds proprietary safety heuristics. To keep the
open‑source core safe:

- Intent outputs are marked `OSS_ADVISORY_ONLY` when external enforcement is
  required.
- Cost and policy modules contain no cloud credentials.
- Critical constants (e.g. `MAX_POLICY_VIOLATIONS`) are reviewed by the
  security team.

Anyone can fork and run the OSS engine, but production enforcement requires
enterprise integration.

---

## 🔗 Links

- **Live demo:** https://agentic-reliability-framework.demo.example.com  
- **Full documentation:** https://docs.agentic-reliability-framework.io  
- **Azure simulator API reference:** [docs/infrastructure.md](docs/infrastructure.md)

## 📦 Additional File: Example Pricing YAML (`pricing.yml`)

```yaml
vm:
  Standard_D2s_v3: 70.0
  Standard_D4s_v3: 140.0
  custom_extra_large: 999.0
storage_account:
  "50GB": 5.0
  "1TB": 100.0
```
