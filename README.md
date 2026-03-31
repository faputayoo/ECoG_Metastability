# ECoG Metastability Analysis

Analysis of neural metastability across brain states (awake vs. unconscious) in macaque ECoG recordings, with computational model validation.

## Data

128-channel ECoG recordings from two macaques (George, Chibi) under four conditions:

- Ketamine (KT)
- Medetomidine (MD)
- Propofol (PF)
- Natural sleep (SLP)

Data source: [NeuroTycho](http://neurotycho.org/) (Yanagawa et al., 2012). Place raw `.mat` files under `data/`.

## Methods

**Empirical metrics** (computed per frequency band with CAR preprocessing):

- Kuramoto order parameter — synchrony ⟨R(t)⟩ and metastability Var[R(t)]
- Weighted Phase Lag Index (wPLI)
- Functional Connectivity Dynamics (FCD) variance
- Coalition entropy

**Statistics**: permutation tests, Cohen's d with bootstrap 95% CI, Benjamini-Hochberg FDR, Bayes factors.

**Computational models**: Jansen-Rit (6-node) and Wilson-Cowan (6-node) networks simulating dose-dependent anesthetic effects. Model predictions compared against empirical direction of change.

## Project Structure

```
run_pipeline.py          # main script
src/
  config.py              # parameters, dataset paths, frequency bands
  data_loading.py        # .mat file I/O, epoch extraction
  preprocessing.py       # CAR, bad channel detection, bandpass + Hilbert
  metrics.py             # Kuramoto, wPLI, FCD, coalition entropy
  statistics.py          # permutation test, effect sizes, Bayes factor
  models.py              # Jansen-Rit & Wilson-Cowan networks
  visualization.py       # all 8 figures
figures/                 # output PNGs
```

## Usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy matplotlib seaborn tqdm
python run_pipeline.py
```

Outputs 8 figures to `figures/`.

## Key Results

| Metric | Band | Cohen's d | p | BF10 |
|--------|------|-----------|---|------|
| wPLI | broadband | +1.605 | 0.016 | 24.6 (strong) |
| Kuramoto sync | theta | −1.092 | 0.032 | 6.1 (moderate) |
| Kuramoto meta | delta | −0.962 | 0.033 | 4.2 (moderate) |

Model validation (direction match): Jansen-Rit 7/8 (88%), Wilson-Cowan 4/8 (50%).
