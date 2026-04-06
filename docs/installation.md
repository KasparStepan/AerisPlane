# Installation

## Requirements

- Python ≥ 3.10
- numpy, scipy, matplotlib, neuralfoil (installed automatically)

## Standard install

```bash
pip install aerisplane
```

## Editable install (development)

```bash
git clone https://github.com/KasparStepan/AerisPlane.git
cd AerisPlane
pip install -e ".[dev]"
```

## Optional extras

```bash
pip install aerisplane[oas]          # OpenAeroStruct for detailed structural analysis
pip install aerisplane[optimize]     # pygmo global optimizers
pip install aerisplane[interactive]  # Plotly interactive flow visualisation
pip install aerisplane[all]          # everything
```
