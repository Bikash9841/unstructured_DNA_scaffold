import math
import itertools
from collections import Counter

from folding import fold_dna


# _ent: normalized Shannon entropy of the k-mers in a sequence.
# s (str): DNA sequence. k (int): k-mer length, e.g. 2 or 3.
# Slides a window of size k over s and counts each k-mer. With p = count / total,
# H = -sum(p * ln(p)), divided by ln(4^k) so the max is 1.
# Returns a float from 0 to 1. Near 1 is random-looking, near 0 is repetitive.
def _ent(s, k):
    counts = Counter(s[i:i + k] for i in range(len(s) - k + 1))
    n = sum(counts.values())
    h = -sum((v / n) * math.log(v / n) for v in counts.values())
    return h / math.log(4 ** k)


# _ramp: linear ramp that maps a value to a score from 0 to 1.
# x (float): value to score. lo (float): score is 0 at or below this.
# hi (float): score is 1 at or above this.
# Computes (x - lo) / (hi - lo), clipped to 0 to 1.
# Returns a float from 0 to 1.
def _ramp(x, lo, hi):
    return min(1.0, max(0.0, (x - lo) / (hi - lo)))


# folding_term: how unfolded a sequence is, using the ViennaRNA fold.
# seq (str): DNA sequence. w (float): weight on score a, b gets 1 - w.
# Gets the structure and MFE from fold_dna. With L = length:
#   a = exp((MFE / L) / 0.1), energy per base (0.1 is a scale to tune)
#   b = 1 - paired / L, where paired = L minus the "." count
# Returns w * a + (1 - w) * b, from 0 to 1. 1.0 means no folding.
def folding_term(seq, w=0.5):
    structure, mfe = fold_dna(seq)
    length = len(seq)
    a = math.exp((mfe / length) / 0.1)
    paired = length - structure.count(".")
    b = 1 - paired / length
    return w * a + (1 - w) * b


# complexity_term: penalizes low-complexity DNA like poly-A or A/C-only.
# seq (str): DNA sequence.
# Averages the 2-mer and 3-mer entropy, then ramps it: 0 at 0.6, 1 at 0.8.
# Returns a float from 0 to 1.
def complexity_term(seq):
    h = (_ent(seq, 2) + _ent(seq, 3)) / 2
    return _ramp(h, 0.6, 0.8)


# composition_term: penalizes GC content outside 40 to 60 percent.
# seq (str): DNA sequence.
# With g = (G count + C count) / length, takes the smaller of a rising ramp
# (0 at 0.2, 1 at 0.4) and a falling ramp (1 at 0.6, 0 at 0.8).
# Returns a float from 0 to 1.
def composition_term(seq):
    g = (seq.count("G") + seq.count("C")) / len(seq)
    rising = _ramp(g, 0.2, 0.4)
    falling = 1 - _ramp(g, 0.6, 0.8)
    return min(rising, falling)


# repeat_term: penalizes long runs of one letter, like AAAAAAAAAA.
# seq (str): DNA sequence.
# Loops over runs with itertools.groupby and keeps the longest. Score is
# 1 - ramp(longest, 8, 20): full score up to 8, zero at 20 (first guesses).
# Returns a float from 0 to 1.
def repeat_term(seq):
    longest = max(len(list(group)) for _, group in itertools.groupby(seq))
    return 1 - _ramp(longest, 8, 20)


# terms: all four scores for one sequence, handy for inspecting.
# seq (str): DNA sequence.
# Returns (folding, complexity, composition, repeat), each from 0 to 1.
def terms(seq):
    return (
        folding_term(seq),
        complexity_term(seq),
        composition_term(seq),
        repeat_term(seq),
    )


# total_additive: weighted-sum reward, as in the proposal. Kept for comparison.
# seq (str): DNA sequence. w (tuple of 3): weights for folding, complexity,
# composition. Default (0.5, 0.25, 0.25).
# total = w0 * folding + w1 * complexity * repeat + w2 * composition
# Returns a float from 0 to 1. In testing, A/C-only DNA scored 0.75 here,
# above random DNA at 0.70, so this version can be gamed.
def total_additive(seq, w=(0.5, 0.25, 0.25)):
    folding, complexity, composition, repeat = terms(seq)
    return w[0] * folding + w[1] * complexity * repeat + w[2] * composition


# clean_sequence: turns raw generator text into a plain A/C/G/T string.
# text (str): decoded model output, may have lowercase, N or stray tokens.
# Uppercases the text and keeps only the characters A, C, G, T.
# Returns a str. It can be empty if the text had no valid letters.
def clean_sequence(text):
    return "".join(c for c in text.upper() if c in "ACGT")


# safe_reward: reward for raw generator text that never crashes.
# text (str): decoded model output. min_len (int): shortest sequence we score,
# 50 is a placeholder since very short strings give meaningless entropy.
# Cleans the text, then returns 0.0 if it is shorter than min_len, otherwise
# total_multiplicative of the cleaned sequence.
# Returns a float from 0 to 1.
def safe_reward(text, min_len=50):
    seq = clean_sequence(text)
    if len(seq) < min_len:
        return 0.0
    return total_multiplicative(seq)


# dna_reward_func: batch reward in the format the TRL GRPOTrainer expects.
# completions (list of str): one decoded sequence per generation.
# **kwargs: extra fields the trainer passes in, unused here.
# Loops over the completions and applies safe_reward to each.
# Returns a list of floats, one per completion, in the same order.
def dna_reward_func(completions, **kwargs):
    return [safe_reward(c) for c in completions]


# total_multiplicative: the reward to use.
# seq (str): DNA sequence.
# total = folding * complexity * composition * repeat. If any term is 0 the
# whole reward is 0, which shuts down the A/C-only and poly-A exploits.
# Returns a float from 0 to 1. Natural 300 bp DNA scored about 0.31 in testing.
def total_multiplicative(seq):
    folding, complexity, composition, repeat = terms(seq)
    return folding * complexity * composition * repeat