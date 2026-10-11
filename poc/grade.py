"""Answer extraction, grading, loop detection."""
import re


def last_boxed(s: str):
    i = s.rfind("\\boxed")
    if i < 0:
        return None
    j = s.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(s)):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return s[j + 1:k]
    return None


def answer_part(text: str) -> str:
    return text.split("</think>")[-1] if "</think>" in text else text


def _norm(a: str) -> str:
    a = a.strip()
    a = re.sub(r"\\text\{\s*([^}]*)\}", r"\1", a)
    a = re.sub(r"\\mbox\{\s*([^}]*)\}", r"\1", a)
    for x in ["\\left", "\\right", "\\!", "\\,", "\;", "\\ ", "$", "\\displaystyle"]:
        a = a.replace(x, "")
    a = a.replace("dfrac", "frac").replace("tfrac", "frac").replace("^\\circ", "").replace("^{\\circ}", "")
    a = a.replace("\\%", "").replace("%", "").replace("degrees", "").replace("units", "")
    a = a.replace(" ", "").rstrip(".")
    a = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", a)
    a = re.sub(r"\\sqrt(\d)", r"\\sqrt{\1}", a)
    a = a.lower()
    a = re.sub(r"^[a-zA-Z]\s*=", "", a)
    if re.fullmatch(r"-?[\d,]+(\.\d+)?", a):
        a = a.replace(",", "")
    return a


def _num(a: str):
    try:
        m = re.fullmatch(r"(-?)\\frac\{(-?\d+)\}\{(\d+)\}", a)
        if m:
            return (-1 if m.group(1) else 1) * int(m.group(2)) / int(m.group(3))
        return float(a)
    except Exception:
        return None


def correct(pred, gold) -> bool:
    if pred is None:
        return False
    p, g = _norm(pred), _norm(gold)
    if p == g:
        return True
    pn, gn = _num(p), _num(g)
    return pn is not None and gn is not None and abs(pn - gn) <= 1e-6 * max(1, abs(gn))


def has_loop(tokens, n=4, reps=8) -> bool:
    """Any n-gram repeated >= reps times back to back."""
    L = len(tokens)
    span = n * reps
    if L < span:
        return False
    for start in range(L - span + 1):
        block = tokens[start:start + n]
        ok = True
        for r in range(1, reps):
            if tokens[start + r * n:start + (r + 1) * n] != block:
                ok = False
                break
        if ok:
            return True
    return False
