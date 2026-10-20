# Vector Embedding Model & 3D Transformation Instrument

Interactive 2560-dimensional semantic embedding engine and 3D Three.js projection instrument powered by **Qwen/Qwen3-Embedding-4B** on Apple Silicon (`mps`), CUDA, or CPU.

## Learning Objectives

- **High-Dimensional Representation**: Understand how transformer embedding models map discrete words and phrases onto the unit hypersphere in $\mathbb{R}^{2560}$.
- **Linear Semantic Structure**: Observe how relational directions (gender, royalty, geography, tense) emerge as linear displacement vectors $\Delta = \vec{v}\sb{\text{target}} - \vec{v}\sb{\text{source}}$.
- **Analogical Reasoning via Vector Arithmetic**: Compute transformations $\vec{v}\sb{\text{result}} = \text{normalize}(\vec{v}\sb{\text{base}} + \Delta)$ (e.g., `uncle + (woman - man) -> aunt`) and measure cosine similarity ($\cos \theta$) and angular distance ($\theta^\circ$).
- **Fast Vector Retrieval**: Implement sub-millisecond nearest-neighbor search over a cached vocabulary matrix via BLAS dot products ($\mathbf{M}\sb{\text{vocab}} \cdot \vec{v}\sb{q}$).
- **Dimensionality Reduction & Fidelity**: Project $\mathbb{R}^{2560}$ vectors into $\mathbb{R}^3$ using Principal Component Analysis (PCA) while tracking explained variance across principal axes (`PC1`, `PC2`, `PC3`).

## Features

1. **2560-Dimensional Word & Phrase Embeddings** ([`main.py`](./main.py)):
   - Computes unit-normalized 2560-dim vectors via `Qwen/Qwen3-Embedding-4B` cached locally in `./models/`.
2. **Vector Arithmetic Expressions**:
   - Evaluates multi-term additions and subtractions such as `"king - man + woman"`, `"woman - man"`, and `"paris - france + italy"`.
3. **Analogical Transformation Function**:
   - Computes $\vec{v}\sb{\text{result}} = \text{normalize}(\vec{v}\sb{\text{base}} + (\vec{v}\sb{\text{target}} - \vec{v}\sb{\text{source}}))$, e.g. `uncle + (woman - man) -> aunt` (cosine similarity `0.8201`).
4. **Sub-Millisecond Vocabulary Nearest-Neighbor Search** ([`vocab_manager.py`](./vocab_manager.py)):
   - Pre-indexes 2,500 curated benchmark and common English words in `vocab_cache.npz` for `< 1.5 ms` BLAS dot-product lookup.
   - Dynamically indexes any custom word entered during a session.
5. **Precision 3D Web Instrument** ([`server.py`](./server.py), [`static/index.html`](./static/index.html)):
   - **01 EXPR**: Plot individual concepts or arbitrary linear combinations.
   - **02 TRANSFORM**: Dedicated `Base (v₀)`, `Target (+)`, and `Source (−)` workbench with 1-click axis inversion (`Swap ↔`), benchmark presets, and basis vector toggles.
   - **03 COMPARE & Matrix (`[M]`)**: Measure exact $\mathbb{R}^{2560}$ pairwise cosine similarity and angular separation ($\theta^\circ$), or open the full $N \times N$ similarity heatmap modal.
   - **Zero-Idle WebGL Engine**: Demand-driven render loop (`0% GPU` when stationary), explicit WebGL geometry/material disposal, and live 3D PCA explained-variance telemetry.

## Quickstart

### 1. Create Environment & Install Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Launch the 3D Web Instrument
```bash
.venv/bin/python server.py
```
Open **[http://127.0.0.1:5050](http://127.0.0.1:5050)** in your browser.
*(Note: First run downloads ~7.5 GB of `Qwen/Qwen3-Embedding-4B` weights into `./models/` and builds `vocab_cache.npz`.)*

### 3. CLI Usage
```bash
# Single word embedding + nearest neighbors
.venv/bin/python main.py "ocean"

# Vector arithmetic expression
.venv/bin/python main.py "king - man + woman"

# Semantic transformation: Base + (Target - Source)
.venv/bin/python main.py transform uncle man woman

# Pairwise cosine similarity
.venv/bin/python main.py similarity king queen

# Interactive REPL
.venv/bin/python main.py
```

### 4. Run Automated Tests
```bash
.venv/bin/python -m unittest discover -s tests -v
```
