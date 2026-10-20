# Vector Embedding Model & 3D Transformation Studio

Interactive 2560-dimensional semantic embedding engine and 3D Three.js visualization studio powered by **Qwen/Qwen3-Embedding-4B** on Apple Silicon (`mps`), CUDA, or CPU.

## Features

1. **2560-Dimensional Word & Phrase Embeddings** ([`main.py`](./main.py)):
   - Computes unit-normalized 2560-dim vectors via `Qwen/Qwen3-Embedding-4B` cached locally in `./models/`.
2. **Vector Arithmetic Expressions**:
   - Evaluates multi-term additions and subtractions such as `"king - man + woman"`, `"woman - man"`, and `"paris - france + italy"`.
3. **Analogical Transformation Function**:
   - Computes $\vec{v}_{\text{result}} = \text{normalize}(\vec{v}_{\text{base}} + (\vec{v}_{\text{target}} - \vec{v}_{\text{source}}))$, e.g. `uncle + (woman - man) -> aunt` (cosine similarity `0.8201`).
4. **Sub-Millisecond Vocabulary Nearest-Neighbor Search** ([`vocab_manager.py`](./vocab_manager.py)):
   - Pre-indexes 2,500 curated benchmark and common English words in `vocab_cache.npz` for `< 1.5 ms` BLAS dot-product lookup.
   - Dynamically indexes any custom word entered during a session.
5. **Interactive 3D Web Studio** ([`server.py`](./server.py), [`static/index.html`](./static/index.html)):
   - **Expression Tab**: Plot individual words or arbitrary vector formulas.
   - **Transformation Tab**: Dedicated `Base`, `Target (+)`, and `Source (-)` studio with benchmark presets and toggles for component vectors and input filtering.
   - **Compare Tab**: Compute exact 2560-dim pairwise cosine similarity between any two concepts and link them in 3D.
   - **3D Scene Controls**: PCA projection (`2560-dim -> 3D`), orbit/pan/zoom camera, click-to-focus raycasting, individual vector removal, and session state persistence across refreshes.

## Quickstart

### 1. Activate Environment & Install Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Launch the 3D Web Visualizer
```bash
.venv/bin/python server.py
```
Open **[http://127.0.0.1:5050](http://127.0.0.1:5050)** in your browser.

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
