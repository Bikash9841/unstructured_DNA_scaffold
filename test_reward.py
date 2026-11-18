import random

from reward import dna_reward_func, safe_reward, terms, total_multiplicative


# random_dna
# Input: n, how many letters to make. seed, which random sequence to build.
# Output: a DNA string of length n.
# Purpose: builds a random A/C/G/T sequence for the tests. The same seed always
# gives the same sequence, so a failing test can be reproduced.
def random_dna(n, seed=0):
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(n))


# test_exploits_score_zero
# Input: poly-A and A/C-only sequences, 300 bp each.
# Output: both must score exactly 0.
# Purpose: these sequences cannot fold, so the folding score alone would rate them
# perfect. The complexity score catches them, and multiplying the terms turns the
# whole reward into 0. This is the test that stops the generator from cheating.
def test_exploits_score_zero():
    rng = random.Random(0)
    assert total_multiplicative("A" * 300) == 0.0
    assert total_multiplicative("".join(rng.choice("AC") for _ in range(300))) == 0.0


# test_random_dna_scores_in_middle
# Input: ten random 300 bp sequences.
# Output: every score must land between 0.2 and 0.7.
# Purpose: ordinary DNA should sit in the middle of the range, not at 0 or 1. If
# everything scored 0 the generator would get no signal to learn from.
def test_random_dna_scores_in_middle():
    scores = [total_multiplicative(random_dna(300, seed)) for seed in range(10)]
    assert all(0.2 < s < 0.7 for s in scores)


# test_safe_reward_handles_messy_text
# Input: empty text, a 3-letter string, lowercase DNA, and DNA with a [BOS] token.
# Output: the first two score 0, and the last two score the same as clean DNA.
# Purpose: the generator's raw output is messy. This checks that cleaning works
# and that nothing crashes on text too short to score.
def test_safe_reward_handles_messy_text():
    clean = random_dna(300, 1)
    assert safe_reward("") == 0.0
    assert safe_reward("ACG") == 0.0
    assert safe_reward(clean.lower()) == safe_reward(clean)
    assert safe_reward("[BOS]" + clean) == safe_reward(clean)


# test_batch_format
# Input: a list holding one random sequence and one poly-A sequence.
# Output: a list of two numbers, the first above 0 and the second exactly 0.
# Purpose: checks the batch function returns one score per sequence, in the same
# order it received them.
def test_batch_format():
    batch = [random_dna(300, 2), "A" * 300]
    out = dna_reward_func(batch)
    assert len(out) == 2
    assert out[0] > 0.0
    assert out[1] == 0.0


# test_grpo_call_format
# Input: the exact arguments GRPOTrainer passes (prompts, completions,
# completion_ids and a spare field), with four sequences including bad text.
# Output: four numbers, all between 0 and 1, with the bad ones scoring 0.
# Purpose: makes sure the reward plugs into GRPO without changes on Bikash's side.
def test_grpo_call_format():
    batch = [random_dna(300, 3), "", "ACG", "A" * 300]
    out = dna_reward_func(prompts=["p"] * 4, completions=batch, completion_ids=[[1]] * 4, extra="x")
    assert len(out) == 4
    assert all(isinstance(v, float) for v in out)
    assert all(0.0 <= v <= 1.0 for v in out)
    assert out[1] == 0.0 and out[2] == 0.0


# test_chat_format_completions
# Input: the same sequence twice, once as plain text and once wrapped in a
# chat-style message list.
# Output: the two scores must match.
# Purpose: GRPO sends either format depending on how the prompts are set up, so
# both have to work.
def test_chat_format_completions():
    text = random_dna(300, 4)
    chat = [[{"role": "assistant", "content": text}]]
    assert dna_reward_func(completions=chat) == dna_reward_func(completions=[text])


# test_total_is_product_of_terms
# Input: five random sequences and one poly-A sequence.
# Output: each reward must equal its four term scores multiplied together.
# Purpose: confirms the reward really is the product of the parts, so reading the
# four terms tells you exactly why a sequence scored what it did.
def test_total_is_product_of_terms():
    batch = [random_dna(300, s) for s in range(5)] + ["A" * 300]
    total = dna_reward_func(completions=batch)
    for seq, score in zip(batch, total):
        folding, complexity, composition, repeat = terms(seq)
        assert abs(score - folding * complexity * composition * repeat) < 1e-9


# test_reward_is_stable_across_target_lengths
# Input: 15 random sequences at each of 200, 300, 400 and 500 bp.
# Output: the average scores must stay within 0.1 of each other.
# Purpose: the folding score divides by length, which keeps scores close across
# the 200 to 500 bp range this project targets. It is not a perfect correction:
# shorter DNA folds less per base, so sequences well under 200 bp score noticeably
# higher (about 0.52 at 50 bp against 0.38 at 300 bp). Keep generated sequences
# near one length, or the generator can gain reward just by writing shorter ones.
def test_reward_is_stable_across_target_lengths():
    means = {}
    for length in (200, 300, 400, 500):
        scores = [total_multiplicative(random_dna(length, seed)) for seed in range(15)]
        means[length] = sum(scores) / len(scores)
    assert max(means.values()) - min(means.values()) < 0.1