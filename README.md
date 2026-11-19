# Unstructured DNA Scaffold

CPTS 570 project: generate DNA sequences that stay unfolded and look natural.
First checkpoint: ViennaRNA in DNA mode, tested on promoter sequences from the
Hugging Face `InstaDeepAI/nucleotide_transformer_downstream_tasks` dataset.

## Quickstart

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
pytest -v                         # expect: 4 passed (this is the setup check)
```

## Files

- `folding.py`: ViennaRNA in DNA mode (Mathews 2004, no GU, no lonely pairs).
  `fold_dna()` loads the DNA parameters once. Without them ViennaRNA silently
  scores DNA with RNA energies.
- `test_folding.py`: folding tests, including known reference values.

## Next

- Load and clean the promoter sequences, fold them, summarize the MFE values.
- Reward function comes after this checkpoint.
