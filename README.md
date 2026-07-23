# HypercubeML

**Machine learning on the Boolean hypercube** — shared topology, sparse XOR addressing, C++23 cores with Python SDKs.

HypercubeML is the public home for a family of libraries that put modern ML primitives on the *same* geometric substrate: a `DIM`-dimensional Boolean hypercube with **N = 2^DIM** vertices. Connectivity is not stored; it is **computed** with bit flips and XOR. Values on the vertices are ordinary continuous floats — only the *graph* is binary.

| | |
|:--|:--|
| **Topology** | Boolean hypercube · vertex-transitive · no borders to pad |
| **Addressing** | Neighbor of `v` along axis `i` is `v XOR (1 << i)` |
| **Scale** | Practical range typically DIM 4–16 (16 … 65,536 vertices) |
| **Stack** | Dependency-light C++23 · optional Python wheels · Apache 2.0 |

---

## Why a hypercube?

Most ML stacks inherit Euclidean defaults: grids, pads, adjacency lists, and graphs you must generate, store, and version. The hypercube is a different bet.

**One structure, three jobs.**

| Job | What the cube gives you |
|-----|-------------------------|
| **Space** | Every vertex has exactly DIM nearest neighbors (one bit flip each). Local kernels and sparse gathers are uniform — no edge cases, no hubs, no degree lottery. |
| **Time** | Delay lines and multi-slice gathers sit on the same indices. Memory can be *addressed*, not only hoped for in a random recurrent graph. |
| **Attention / retrieval** | Hamming balls and XOR masks define sparse neighborhoods for associative memory without a dense N×N table. |

**Properties that random sparse graphs do not get for free**

- **Zero adjacency storage** — no edge list to build, serialize, or cache-miss through; neighbors are arithmetic.
- **Vertex-transitivity** — every site sees an identical local world, so weight sharing (convolutions) is exact, not “almost, except borders.”
- **Logarithmic diameter** — any two of N vertices are at most DIM = log₂N hops apart: sparse local wiring with global reach.
- **Reproducible structure** — same DIM means the same graph in every language and implementation; reservoirs and kernels reconstruct from seeds and a handful of scalars.

**What this is not**

- Not a claim that activations are bits (they are floats).
- Not a drop-in replacement for 2D vision stacks (images must be *packed* onto the cube when you care about pixels).
- Not a monorepo — each product is a focused library with its own SDK and release train.

---

## The idea in one picture

```text
                    DIM-bit vertex index  (0 … N−1)
                              │
              ┌───────────────┼───────────────┐
              │               │               │
         flip bit 0      flip bit 1      …  flip bit DIM−1
              │               │               │
              ▼               ▼               ▼
         neighbor        neighbor         neighbor
              \               |               /
               \              |              /
                ──── local Hamming neighborhood ────
                              │
         continuous state / features live on vertices
```

Same neighborhood primitive shows up as:

- **CNN** kernel taps (self + one weight per bit axis)  
- **ESN** recurrent gather (neighbors × delay-line depth)  
- **Hopfield** sparse attention mask (Hamming ball / fraction of the cube)

---

## Ecosystem

Products live today under [@dliptak001](https://github.com/dliptak001) and are catalogued here. More libraries will join this family; this page is the map.

### Product matrix

| Library | Role | One-line value | Python | C++ |
|---------|------|----------------|--------|-----|
| **[HypercubeCNN](https://github.com/dliptak001/HypercubeCNN)** | Convolutional body + linear head | CNN on the cube: XOR neighborhoods, shared taps, end-to-end backprop | `hypercube-cnn` | `HypercubeCNNCore` |
| **[HypercubeESN](https://github.com/dliptak001/HypercubeESN)** | Reservoir + learned readout | Echo-state dynamics on the cube; HCNN as topology-native readout | `hypercube-esn` | full pipeline + examples |
| **[HypercubeHopfield](https://github.com/dliptak001/HypercubeHopfield)** | Associative memory | Modern (softmax) Hopfield with sparse hypercube attention | `hypercube-hopfield` | `HopfieldNetwork` |

### How they fit together

```text
  HypercubeCNN          HypercubeESN              HypercubeHopfield
  ────────────          ───────────              ─────────────────
  learned features      fixed recurrent field     content-addressable
  on the cube           on the cube               memory on the cube
        ▲                      │
        │                      │  state is a signal on vertices
        └──── HCNN readout ────┘
              (native topology; no flatten-to-grid detour)
```

- **HypercubeCNN** is the shared *learned* workhorse: classification, regression, and the trained head on reservoir state.
- **HypercubeESN** is reservoir computing where wiring *is* the cube (and an addressable delay line deepens temporal memory). Training stays cheap: freeze the reservoir, train the readout — classically linear, here a hypercube CNN.
- **HypercubeHopfield** is retrieval and pattern completion: modern Hopfield energy / softmax attention restricted to XOR-defined neighborhoods for capacity and cost that scale sanely with N.

Use one library alone, or compose: drive an ESN, read it with HCNN, store/recall patterns with Hopfield — same addressing language throughout.

### Quick links

| Product | Repo | Docs entry | PyPI |
|---------|------|------------|------|
| CNN | [HypercubeCNN](https://github.com/dliptak001/HypercubeCNN) | [CPP SDK](https://github.com/dliptak001/HypercubeCNN/blob/main/docs/CPP_SDK.md) · [Python SDK](https://github.com/dliptak001/HypercubeCNN/blob/main/docs/Python_SDK.md) | [hypercube-cnn](https://pypi.org/project/hypercube-cnn/) |
| ESN | [HypercubeESN](https://github.com/dliptak001/HypercubeESN) | [CPP SDK](https://github.com/dliptak001/HypercubeESN/blob/main/docs/CPP_SDK.md) · [Python SDK](https://github.com/dliptak001/HypercubeESN/blob/main/docs/Python_SDK.md) | [hypercube-esn](https://pypi.org/project/hypercube-esn/) |
| Hopfield | [HypercubeHopfield](https://github.com/dliptak001/HypercubeHopfield) | [CPP SDK](https://github.com/dliptak001/HypercubeHopfield/blob/main/docs/CPP_SDK.md) · [Python SDK](https://github.com/dliptak001/HypercubeHopfield/blob/main/docs/Python_SDK.md) | [hypercube-hopfield](https://pypi.org/project/hypercube-hopfield/) |

---

## HypercubeCNN — convolution without a grid

A spatial CNN slides a small kernel across a rectangle. **HypercubeCNN** does the same *idea* on a different domain: each channel holds **N = 2^DIM** activations; each site’s neighborhood is **self + DIM bit-flip neighbors** (**K = DIM + 1** shared taps). No padding, no border special cases — the cube is closed under XOR.

- Stack conv (and optional **antipodal** pool: pair each vertex with its complement, fold DIM by one).
- Single linear readout over the full final feature map (FLATTEN).
- Classification or regression; pure C++23 core; Python bindings for train/infer.

**Good fit when** inputs already live at length `2^D`, or you own a pack onto N (images, fingerprints, reservoir state, product features).

→ [github.com/dliptak001/HypercubeCNN](https://github.com/dliptak001/HypercubeCNN)

---

## HypercubeESN — reservoirs with geometry

Reservoir computing splits hard temporal learning in two: a **fixed** recurrent dynamical system lifts the input into a rich state; a **trained** readout maps that state to targets. HypercubeESN makes the reservoir a **signal on the hypercube**: each update gathers over Hamming-distance-1 neighbors (and, with history depth M, over an addressable delay line of past slices). Connectivity is never stored; spectral radius and seeds specify the whole system.

The readout is **HypercubeCNN** — kernels that speak the same XOR geometry as the reservoir, so state is not flattened onto a foreign 2D grid.

- Online / streaming training paths and a battery of examples (prediction, classification, NARMA, memory capacity, text, anomaly, …).
- C++23 pipeline + Python `fit` / streaming APIs.

→ [github.com/dliptak001/HypercubeESN](https://github.com/dliptak001/HypercubeESN)

---

## HypercubeHopfield — sparse modern associative memory

Classical Hopfield nets collapse patterns into a weight matrix and hit a low capacity ceiling. **Modern** Hopfield networks store patterns explicitly and retrieve with **softmax attention** (log-sum-exp energy), with capacity that scales far more generously with N.

HypercubeHopfield keeps that modern retrieval rule but **restricts attention to a Hamming ball** (and optional neighbor fraction) on the cube. Neighbor masks are XOR-defined and distance-sorted — sparse local attention with super-linear practical capacity, not a dense all-to-all table.

- Auto- and hetero-associative demos, diagnostics suite, C++ and Python SDKs.

→ [github.com/dliptak001/HypercubeHopfield](https://github.com/dliptak001/HypercubeHopfield)

---

## Shared design principles

Across the suite:

1. **Topology is a first-class contract** — capacity is power-of-two by construction; hosts pack or produce length-N signals deliberately.
2. **Sparse by math, not by sparsity patterns you hand-tune** — degree and neighborhoods come from DIM (and, for Hopfield, ball radius / fraction).
3. **Systems-friendly C++** — small public surfaces, CMake / FetchContent, few or no heavy third-party deps in the cores.
4. **Python for iteration** — wheels where published; same contracts as the C++ APIs as far as practical.
5. **Honest examples** — demos prove the stack learns and integrates; they are recipes, not leaderboard claims.

---

## Getting started

Pick the entry that matches your problem:

| You want… | Start here |
|-----------|------------|
| Learned features / classify or regress on length-N state | [HypercubeCNN](https://github.com/dliptak001/HypercubeCNN) |
| Temporal / streaming / reservoir-style memory | [HypercubeESN](https://github.com/dliptak001/HypercubeESN) |
| Store and recall patterns, complete noisy cues | [HypercubeHopfield](https://github.com/dliptak001/HypercubeHopfield) |
| ESN + native conv readout | ESN (depends on CNN as the trained head) |

Python (when wheels are published for that package):

```bash
pip install hypercube-cnn
pip install hypercube-esn
pip install hypercube-hopfield
```

C++: clone the product you need, or pull it with CMake `FetchContent` / `find_package` as documented in each repo’s CPP SDK guide.

---

## Organization status

[HypercubeML](https://github.com/HypercubeML) is the **brand and map** for this ecosystem. Product repositories currently remain under [dliptak001](https://github.com/dliptak001) so existing clone URLs, CI, and package metadata stay undisturbed. Links above always point at the live code.

New Hypercube-based products will be announced here as they ship.

---

## License & contact

Individual products are released under **Apache License 2.0** unless a repo states otherwise. Issues and discussions live on each product repository.

Maintainer: [David Liptak](https://github.com/dliptak001) · Organization: [HypercubeML](https://github.com/HypercubeML)

