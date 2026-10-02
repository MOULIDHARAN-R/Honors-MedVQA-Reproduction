"""Run from MedVqa/:  python -m scripts.test_answers"""
from src.answers import (split_alternatives, canonical_label, expand_optional,
                         acceptable_forms, AnswerVocab)

# '#' alternatives
assert split_alternatives("A#B") == ["A", "B"]
assert acceptable_forms("A#B") == ["A", "B"]
assert canonical_label("heart and great vessels#lung, mediastinum, pleura") == "heart and great vessels"

# optional text
f = acceptable_forms("(angioinvasive aspergillosis with) septic cerebral emboli")
assert set(f) == {"(angioinvasive aspergillosis with) septic cerebral emboli",
                  "angioinvasive aspergillosis with septic cerebral emboli",
                  "septic cerebral emboli"}, f
assert f[0] == "(angioinvasive aspergillosis with) septic cerebral emboli"   # literal first

# optional text at the end keeps the literal form
assert set(acceptable_forms("ct w/contrast (iv)")) == {"ct w/contrast (iv)", "ct w/contrast iv", "ct w/contrast"}

# both notations together
assert set(acceptable_forms("(x) a#b")) == {"(x) a", "x a", "a", "b"}

# several groups, tidy-up of spaces/commas
assert set(expand_optional("a (b) c (d)")) == {"a (b) c (d)", "a b c d", "a b c", "a c d", "a c"}
assert "a, c" in expand_optional("a (b), c")

# no parentheses / unbalanced / nested -> literal only
assert acceptable_forms("plain answer") == ["plain answer"]
assert acceptable_forms("broken (iv") == ["broken (iv"]
assert acceptable_forms("a (b (c))") == ["a (b (c))"]

# vocabulary: one class per training answer, no class per expansion
train = ["ct w/contrast (iv)", "ct w/contrast (gi)", "septic cerebral emboli",
         "(angioinvasive aspergillosis with) septic cerebral emboli", "no", "yes#maybe"]
v = AnswerVocab(train)
assert len(v) == 6, v.idx2ans
assert "maybe" not in v.ans2idx                    # only the first alternative is a class
for a in train:                                    # every class matches its own reference
    assert v.label(a) in v.accepted_ids(a)

# reference "(angioinvasive ...) septic cerebral emboli" accepts both existing classes
ids = v.accepted_ids("(angioinvasive aspergillosis with) septic cerebral emboli")
assert {v.idx2ans[i] for i in ids} == {"(angioinvasive aspergillosis with) septic cerebral emboli",
                                       "septic cerebral emboli"}

# expansion is reference-side only: (iv) and (gi) are NOT equivalent
iv_ids = {v.idx2ans[i] for i in v.accepted_ids("ct w/contrast (iv)")}
assert iv_ids == {"ct w/contrast (iv)"}, iv_ids

# unseen answer -> no accepted ids
assert v.accepted_ids("something never seen") == []

print("all answer-parsing tests passed")