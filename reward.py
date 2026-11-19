import math
import itertools
from collections import Counter

from folding import fold_dna


# Reference points for "folds like ordinary DNA", used to turn the raw folding
# numbers into 0-to-1 scores. These are yardsticks, not physical constants: they
# decide what score typical DNA gets, and shifting them moves every score the
# same way.
# The energy one is the human genome-wide average from Szlachta et al. 2018
# (Genome Biology 19:89): -30.2 kcal/mol over 300-nt windows, so -0.101 per base.
# They folded 19.9 million windows in ViennaRNA DNA mode. Their settings allowed
# GT wobble pairs and G-quadruplexes and ours do not, so this is a close reference
# rather than an exact match.
# The pairing one is what random 300 bp DNA does in our own setup (200 sequences,
# ViennaRNA 2.7.2, DNA mode, noGU, noLP, 37 C).
# Shorter sequences fold less per base, so they score a little higher and longer
# ones a little lower. That is fine while every generated sequence is about the
# same length.
MFE_PER_BASE_SCALE = -30.2 / 300
PAIRED_FRAC_SCALE = 0.525


# _ent
# Input: s, a DNA string. k, the window size (2 for pairs of letters, 3 for triples).
# Output: a number from 0 to 1.
# Purpose: measures how varied the short patterns are. It slides a window of size k
# along the sequence and counts each pattern. With p = how often a pattern appears,
# entropy = -sum(p * ln(p)), divided by ln(4^k) so the best possible score is 1.
# Close to 1 means random-looking. Close to 0 means the same few patterns repeat.
def _ent(s, k):
    counts = Counter(s[i:i + k] for i in range(len(s) - k + 1))
    n = sum(counts.values())
    h = -sum((v / n) * math.log(v / n) for v in counts.values())
    return h / math.log(4 ** k)


# _ramp
# Input: x, the value to score. lo and hi, the two cutoffs.
# Output: a number from 0 to 1.
# Purpose: a straight-line score. Gives 0 when x is at or below lo, 1 when x is at
# or above hi, and a proportional value in between: (x - lo) / (hi - lo).
def _ramp(x, lo, hi):
    return min(1.0, max(0.0, (x - lo) / (hi - lo)))


# folding_term
# Input: seq, a DNA string. w, how much weight the energy score a gets (default 0.5).
# Output: a number from 0 to 1. 1 means no folding at all.
# Purpose: scores how unfolded the sequence is, using ViennaRNA. With L = length,
# both halves divide by L so a long sequence is not punished just for being long:
#   a = exp(-(MFE / L) / MFE_PER_BASE_SCALE)        how strong the folding energy is
#   b = exp(-(paired / L) / PAIRED_FRAC_SCALE)      how many bases are paired
# A ratio of 1 means it folds like random DNA and scores about 0.37. A ratio of 0
# means nothing folds and scores 1. The result is w * a + (1 - w) * b.
def folding_term(seq, w=0.5):
    structure, mfe = fold_dna(seq)
    length = len(seq)
    energy_ratio = (mfe / length) / MFE_PER_BASE_SCALE
    paired_ratio = (1 - structure.count(".") / length) / PAIRED_FRAC_SCALE
    a = math.exp(-energy_ratio)
    b = math.exp(-paired_ratio)
    return w * a + (1 - w) * b


# complexity_term
# Input: seq, a DNA string.
# Output: a number from 0 to 1. 1 means the sequence looks varied and natural.
# Purpose: catches repetitive DNA such as poly-A or A/C-only. It averages the
# 2-letter and 3-letter pattern entropy, then scores it: 0 at 0.6 or below,
# 1 at 0.8 or above.
def complexity_term(seq):
    h = (_ent(seq, 2) + _ent(seq, 3)) / 2
    return _ramp(h, 0.6, 0.8)


# composition_term
# Input: seq, a DNA string.
# Output: a number from 0 to 1. 1 means GC content is between 40 and 60 percent.
# Purpose: keeps the base mix balanced. With g = (G count + C count) / length,
# the score rises from 0 at g = 0.2 to 1 at g = 0.4, stays 1 until 0.6, then falls
# to 0 at g = 0.8.
def composition_term(seq):
    g = (seq.count("G") + seq.count("C")) / len(seq)
    rising = _ramp(g, 0.2, 0.4)
    falling = 1 - _ramp(g, 0.6, 0.8)
    return min(rising, falling)


# repeat_term
# Input: seq, a DNA string.
# Output: a number from 0 to 1. 1 means no long runs of one letter.
# Purpose: penalizes runs like AAAAAAAAAA. It loops over the runs of identical
# letters (itertools.groupby) and takes the longest one. A run of 8 or less scores 1,
# a run of 20 or more scores 0, and in between it falls in a straight line.
def repeat_term(seq):
    longest = max(len(list(group)) for _, group in itertools.groupby(seq))
    return 1 - _ramp(longest, 8, 20)


# terms
# Input: seq, a DNA string.
# Output: four numbers, in this order: (folding, complexity, composition, repeat).
# Purpose: computes every score for one sequence so you can see which one is
# pulling the reward down.
def terms(seq):
    return (
        folding_term(seq),
        complexity_term(seq),
        composition_term(seq),
        repeat_term(seq),
    )


# total_additive
# Input: seq, a DNA string. w, three weights for folding, complexity and composition.
# Output: a number from 0 to 1.
# Purpose: the weighted-sum version from the proposal, kept only for comparison.
# total = w0 * folding + w1 * complexity * repeat + w2 * composition.
# In testing, A/C-only DNA scored 0.75 here, higher than random DNA at 0.70, so
# this version can be gamed. Use total_multiplicative instead.
def total_additive(seq, w=(0.5, 0.25, 0.25)):
    folding, complexity, composition, repeat = terms(seq)
    return w[0] * folding + w[1] * complexity * repeat + w[2] * composition


# clean_sequence
# Input: text, the raw string from the generator (may be lowercase or have stray tokens).
# Output: a string containing only the letters A, C, G, T. Can be empty.
# Purpose: makes the generator's text safe to score. It uppercases everything and
# drops any character that is not A, C, G or T.
def clean_sequence(text):
    return "".join(c for c in text.upper() if c in "ACGT")


# safe_reward
# Input: text, the raw string from the generator. min_len, the shortest sequence
# we score (50 is a placeholder).
# Output: one number from 0 to 1.
# Purpose: scores one generated sequence without crashing on bad text. It cleans
# the text, returns 0 if fewer than min_len letters are left, and otherwise
# returns total_multiplicative of the cleaned sequence.
def safe_reward(text, min_len=50):
    seq = clean_sequence(text)
    if len(seq) < min_len:
        return 0.0
    return total_multiplicative(seq)


# dna_reward_func
# Input: completions, the list of generated sequences GRPO passes in, one per
# generation. **kwargs, extra fields GRPO sends that we do not use.
# Output: a list of numbers from 0 to 1, one per sequence, in the same order.
# Purpose: the function to give GRPO as the reward. It loops over the batch and
# applies safe_reward to each sequence.
def dna_reward_func(completions, **kwargs):
    return [safe_reward(t) for t in _texts(completions)]


# _texts
# Input: completions, the list GRPO passes in. Each item is either a plain string
# or a chat-style message list like [{"role": "assistant", "content": "..."}].
# Output: a list of plain strings, one per item.
# Purpose: lets dna_reward_func work with either input format.
def _texts(completions):
    return [c[0]["content"] if isinstance(c, list) else c for c in completions]


# total_multiplicative
# Input: seq, a DNA string.
# Output: a number from 0 to 1. This is the final reward for one sequence.
# Purpose: reward = folding * complexity * composition * repeat. Multiplying means
# a zero in any one score makes the whole reward zero, which shuts down the
# A/C-only and poly-A exploits. Random 300 bp DNA scores about 0.37, and natural
# 300 bp promoters scored 0.22 to 0.35 in testing.
def total_multiplicative(seq):
    folding, complexity, composition, repeat = terms(seq)
    return folding * complexity * composition * repeat