import RNA

from folding import fold_dna


def test_hairpin_folds():
    stem = "GCGCGCGCGC"
    seq = stem + "TTTT" + stem[::-1].translate(str.maketrans("ACGT", "TGCA"))
    structure, mfe = fold_dna(seq)
    assert "(" in structure and mfe < -1.0


def test_poly_a_is_flat():
    structure, mfe = fold_dna("A" * 100)
    assert mfe == 0.0 and set(structure) == {"."}


def test_dna_params_are_actually_loaded():
    """Guards against silently scoring DNA with RNA energies."""
    seq = "GCGGATTTAGCTCAGTTGGGAGAGCGCCAGACTGAAGATCTGGAGGTCCTGTGTTCGATCCACAGAATTCGCACCA"
    _, dna_mfe = fold_dna(seq)
    RNA.params_load_RNA_Turner2004()
    _, rna_mfe = RNA.fold_compound(seq, RNA.md()).mfe()
    RNA.params_load_DNA_Mathews2004()
    assert abs(dna_mfe - rna_mfe) > 1.0


def test_reference_values():
    """Known sequences, ViennaRNA 2.7.2, DNA settings, 37 C. A big gap means
    DNA mode is off or the version differs."""
    structure, mfe = fold_dna("GCGCGCGCGCTTTTGCGCGCGCGC")
    assert structure == "((((((((((....))))))))))"
    assert abs(mfe - (-17.30)) <= 0.05
    assert fold_dna("AC" * 25)[1] == 0.0
