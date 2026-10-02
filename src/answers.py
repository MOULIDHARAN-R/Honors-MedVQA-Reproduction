"""
answers.py  (place at MedVqa/src/answers.py)

Answer parsing and the answer vocabulary. No torch dependency.

VQA-Med answer notation (as used in the dataset files):
    A#B        two acceptable alternatives: A, or B
    (X) Y      X is optional: "X Y" and "Y" are both acceptable

The paper does NOT say how this notation was turned into classification classes, so the
following are OUR implementation choices ([CHOICE]):

  Training label (one class per training answer):
      the FIRST '#' alternative, kept LITERALLY (parentheses included), whitespace-trimmed.
      No optional expansion is turned into a separate class, so the class set contains only
      strings that actually occur in the data.

  Evaluation (all valid forms are kept):
      acceptable forms of a reference = for every '#' alternative:
          the literal alternative  +  every expansion of its optional (...) groups.
      A prediction (a class = a literal canonical string) is correct if that string is one
      of the reference's acceptable forms. Expansion is applied to the REFERENCE only,
      never to the prediction, so labels that differ only in a parenthesised part
      (e.g. "ct w/contrast (iv)" vs "ct w/contrast (gi)") are NOT treated as equivalent.
      The literal form always stays acceptable, so a class never fails to match its own reference.

  Parentheses that are unbalanced or nested are not expanded (literal only). More than
  MAX_OPTIONAL_GROUPS groups in one alternative are not expanded either.
"""
import re
from typing import List

_GROUP = re.compile(r"\(([^()]*)\)")
MAX_OPTIONAL_GROUPS = 4


def split_alternatives(answer: str, sep: str = "#") -> List[str]:
    """Literal '#'-separated alternatives, whitespace-trimmed, empties removed."""
    parts = [p.strip() for p in answer.split(sep)]
    parts = [p for p in parts if p]
    return parts or [answer.strip()]


def canonical_label(answer: str, sep: str = "#") -> str:
    """The single training label of an answer: first alternative, literal."""
    return split_alternatives(answer, sep)[0]


def _flat_balanced(s: str) -> bool:
    depth = 0
    for ch in s:
        if ch == "(":
            depth += 1
            if depth > 1:
                return False
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _tidy(s: str) -> str:
    s = " ".join(s.split())
    return re.sub(r"\s+([,;:.])", r"\1", s)


def expand_optional(alt: str) -> List[str]:
    """[literal] + every keep/drop combination of its optional (...) groups (literal first, unique)."""
    forms = [alt]
    groups = list(_GROUP.finditer(alt))
    if not groups or len(groups) > MAX_OPTIONAL_GROUPS or not _flat_balanced(alt):
        return forms
    n = len(groups)
    for mask in range(2 ** n - 1, -1, -1):          # bit i set = keep group i's text
        out, pos = [], 0
        for i, g in enumerate(groups):
            out.append(alt[pos:g.start()])
            if (mask >> i) & 1:
                out.append(g.group(1))
            pos = g.end()
        out.append(alt[pos:])
        form = _tidy("".join(out))
        if form and form not in forms:
            forms.append(form)
    return forms


def acceptable_forms(answer: str, sep: str = "#", expand: bool = True) -> List[str]:
    forms: List[str] = []
    for alt in split_alternatives(answer, sep):
        for f in (expand_optional(alt) if expand else [alt]):
            if f not in forms:
                forms.append(f)
    return forms


class AnswerVocab:
    """Classes = canonical labels of the TRAINING answers of one split."""

    def __init__(self, train_answers, sep: str = "#", expand_optional_parens: bool = True):
        self.sep = sep
        self.expand = expand_optional_parens
        self.idx2ans = sorted({canonical_label(a, sep) for a in train_answers})
        self.ans2idx = {a: i for i, a in enumerate(self.idx2ans)}

    def __len__(self):
        return len(self.idx2ans)

    def label(self, answer: str) -> int:
        """Training label; -1 if the canonical label is not in the vocabulary."""
        return self.ans2idx.get(canonical_label(answer, self.sep), -1)

    def accepted_ids(self, answer: str) -> List[int]:
        """Ids of every class that counts as a correct prediction for this reference."""
        forms = acceptable_forms(answer, self.sep, self.expand)
        return sorted({self.ans2idx[f] for f in forms if f in self.ans2idx})

    def save(self, path):
        import json
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.idx2ans, f, ensure_ascii=False, indent=1)


def parenthesized_answers(answers, sep: str = "#"):
    """Distinct answers containing '(' with their acceptable forms (for manual inspection)."""
    seen = sorted({a for a in answers if "(" in a})
    return [(a, acceptable_forms(a, sep, True)) for a in seen]