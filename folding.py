"""ViennaRNA wrapper configured for single-stranded DNA.

Settings follow Szlachta et al. 2018 (Genome Biology 19:89): DNA energy
parameters (Mathews 2004), GU pairs and lonely pairs disallowed.
"""
import RNA

_DNA_PARAMS_LOADED = False

# DNA Params from DNA package
def load_dna_params() -> None:
    """Load the DNA parameter set once per process.

    The load is global, and it must happen before a fold compound is built,
    so every folding call in this package goes through fold_dna().
    """
    global _DNA_PARAMS_LOADED
    if not _DNA_PARAMS_LOADED:
        RNA.params_load_DNA_Mathews2004()
        _DNA_PARAMS_LOADED = True


def fold_dna(seq: str, temperature: float = 37.0) -> tuple[str, float]:
    """Return (dot-bracket structure, MFE in kcal/mol) for a DNA sequence."""
    load_dna_params()
    md = RNA.md()
    md.noGU = 1
    md.noLP = 1
    md.temperature = temperature
    structure, mfe = RNA.fold_compound(seq.upper(), md).mfe()
    return structure, float(mfe)
