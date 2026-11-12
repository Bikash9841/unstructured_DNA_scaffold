import random

from reward import dna_reward_func, safe_reward, total_multiplicative


# Builds a reproducible random A/C/G/T string of the given length.
def random_dna(n, seed=0):
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(n))


# Low-complexity exploit sequences must score 0, not a high reward.
def test_exploits_score_zero():
    rng = random.Random(0)
    assert total_multiplicative("A" * 300) == 0.0
    assert total_multiplicative("".join(rng.choice("AC") for _ in range(300))) == 0.0


# Random 300 bp DNA should land in a sensible middle range, not at 0 or 1.
def test_random_dna_scores_in_middle():
    scores = [total_multiplicative(random_dna(300, seed)) for seed in range(10)]
    assert all(0.2 < s < 0.7 for s in scores)


# Messy generator text must not crash and must give the same score as clean text.
def test_safe_reward_handles_messy_text():
    clean = random_dna(300, 1)
    assert safe_reward("") == 0.0
    assert safe_reward("ACG") == 0.0
    assert safe_reward(clean.lower()) == safe_reward(clean)
    assert safe_reward("[BOS]" + clean) == safe_reward(clean)


# The batch function returns one float per completion, in order.
def test_batch_format():
    batch = [random_dna(300, 2), "A" * 300]
    out = dna_reward_func(batch)
    assert len(out) == 2
    assert out[0] > 0.0
    assert out[1] == 0.0
