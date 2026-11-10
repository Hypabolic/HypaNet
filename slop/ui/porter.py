"""The Porter (1980) stemmer, as defined by Martin Porter's ANSI C reference version.

Porter's own page (https://tartarus.org/martin/PorterStemmer/) calls the ANSI C version definitive: it equals the
original BCPL program, which differs from the published paper in three small points, all kept here as in the C
source (``-DEPARTURE-``): step 2 maps ``bli`` -> ``ble`` (the paper has ``abli`` -> ``able``), step 2 adds ``logi`` ->
``log``, and strings of length 1 or 2 are left alone. Porter's published vocabulary and output lists are produced by
this version, and ``ui/tests/test_porter.py`` checks every word of them.

Input is one lower-case word. Letters other than a-z are treated as consonants, as in the C code. Standalone: standard
library only, mirrored by ``lint/porter.js``.
"""

from __future__ import annotations

__all__ = ["stem"]


class _Stemmer:
    __slots__ = ("b", "k", "j")

    def __init__(self, word: str) -> None:
        self.b = list(word)
        self.k = len(word) - 1
        self.j = 0

    def cons(self, i: int) -> bool:
        ch = self.b[i]
        if ch in "aeiou":
            return False
        if ch == "y":
            return True if i == 0 else not self.cons(i - 1)
        return True

    def m(self) -> int:
        """Number of consonant sequences between 0 and j."""
        n = 0
        i = 0
        j = self.j
        while True:
            if i > j:
                return n
            if not self.cons(i):
                break
            i += 1
        i += 1
        while True:
            while True:
                if i > j:
                    return n
                if self.cons(i):
                    break
                i += 1
            i += 1
            n += 1
            while True:
                if i > j:
                    return n
                if not self.cons(i):
                    break
                i += 1
            i += 1

    def vowelinstem(self) -> bool:
        return any(not self.cons(i) for i in range(0, self.j + 1))

    def doublec(self, j: int) -> bool:
        if j < 1:
            return False
        if self.b[j] != self.b[j - 1]:
            return False
        return self.cons(j)

    def cvc(self, i: int) -> bool:
        if i < 2 or not self.cons(i) or self.cons(i - 1) or not self.cons(i - 2):
            return False
        return self.b[i] not in "wxy"

    def ends(self, s: str) -> bool:
        length = len(s)
        if s[-1] != self.b[self.k]:
            return False
        if length > self.k + 1:
            return False
        if "".join(self.b[self.k - length + 1 : self.k + 1]) != s:
            return False
        self.j = self.k - length
        return True

    def setto(self, s: str) -> None:
        j = self.j
        del self.b[j + 1 :]
        self.b.extend(s)
        self.k = j + len(s)

    def r(self, s: str) -> None:
        if self.m() > 0:
            self.setto(s)

    def step1ab(self) -> None:
        b = self.b
        if b[self.k] == "s":
            if self.ends("sses"):
                self.k -= 2
            elif self.ends("ies"):
                self.setto("i")
            elif b[self.k - 1] != "s":
                self.k -= 1
        if self.ends("eed"):
            if self.m() > 0:
                self.k -= 1
        elif (self.ends("ed") or self.ends("ing")) and self.vowelinstem():
            self.k = self.j
            if self.ends("at"):
                self.setto("ate")
            elif self.ends("bl"):
                self.setto("ble")
            elif self.ends("iz"):
                self.setto("ize")
            elif self.doublec(self.k):
                self.k -= 1
                if self.b[self.k] in "lsz":
                    self.k += 1
            elif self.m() == 1 and self.cvc(self.k):
                self.setto("e")

    def step1c(self) -> None:
        if self.ends("y") and self.vowelinstem():
            self.b[self.k] = "i"

    def step2(self) -> None:
        ch = self.b[self.k - 1]
        if ch == "a":
            if self.ends("ational"):
                self.r("ate")
            elif self.ends("tional"):
                self.r("tion")
        elif ch == "c":
            if self.ends("enci"):
                self.r("ence")
            elif self.ends("anci"):
                self.r("ance")
        elif ch == "e":
            if self.ends("izer"):
                self.r("ize")
        elif ch == "l":
            if self.ends("bli"):  # -DEPARTURE- (paper: abli -> able)
                self.r("ble")
            elif self.ends("alli"):
                self.r("al")
            elif self.ends("entli"):
                self.r("ent")
            elif self.ends("eli"):
                self.r("e")
            elif self.ends("ousli"):
                self.r("ous")
        elif ch == "o":
            if self.ends("ization"):
                self.r("ize")
            elif self.ends("ation"):
                self.r("ate")
            elif self.ends("ator"):
                self.r("ate")
        elif ch == "s":
            if self.ends("alism"):
                self.r("al")
            elif self.ends("iveness"):
                self.r("ive")
            elif self.ends("fulness"):
                self.r("ful")
            elif self.ends("ousness"):
                self.r("ous")
        elif ch == "t":
            if self.ends("aliti"):
                self.r("al")
            elif self.ends("iviti"):
                self.r("ive")
            elif self.ends("biliti"):
                self.r("ble")
        elif ch == "g":
            if self.ends("logi"):  # -DEPARTURE- (not in the paper)
                self.r("log")

    def step3(self) -> None:
        ch = self.b[self.k]
        if ch == "e":
            if self.ends("icate"):
                self.r("ic")
            elif self.ends("ative"):
                self.r("")
            elif self.ends("alize"):
                self.r("al")
        elif ch == "i":
            if self.ends("iciti"):
                self.r("ic")
        elif ch == "l":
            if self.ends("ical"):
                self.r("ic")
            elif self.ends("ful"):
                self.r("")
        elif ch == "s":
            if self.ends("ness"):
                self.r("")

    def step4(self) -> None:
        ch = self.b[self.k - 1]
        if ch == "a":
            ok = self.ends("al")
        elif ch == "c":
            ok = self.ends("ance") or self.ends("ence")
        elif ch == "e":
            ok = self.ends("er")
        elif ch == "i":
            ok = self.ends("ic")
        elif ch == "l":
            ok = self.ends("able") or self.ends("ible")
        elif ch == "n":
            ok = self.ends("ant") or self.ends("ement") or self.ends("ment") or self.ends("ent")
        elif ch == "o":
            ok = (self.ends("ion") and self.j >= 0 and self.b[self.j] in "st") or self.ends("ou")
        elif ch == "s":
            ok = self.ends("ism")
        elif ch == "t":
            ok = self.ends("ate") or self.ends("iti")
        elif ch == "u":
            ok = self.ends("ous")
        elif ch == "v":
            ok = self.ends("ive")
        elif ch == "z":
            ok = self.ends("ize")
        else:
            ok = False
        if ok and self.m() > 1:
            self.k = self.j

    def step5(self) -> None:
        self.j = self.k
        if self.b[self.k] == "e":
            a = self.m()
            if a > 1 or (a == 1 and not self.cvc(self.k - 1)):
                self.k -= 1
        if self.b[self.k] == "l" and self.doublec(self.k) and self.m() > 1:
            self.k -= 1

    def run(self) -> str:
        if self.k <= 1:  # -DEPARTURE- strings of length 1 or 2 are left alone
            return "".join(self.b)
        self.step1ab()
        if self.k > 0:
            self.step1c()
            self.step2()
            self.step3()
            self.step4()
            self.step5()
        return "".join(self.b[: self.k + 1])


def stem(word: str) -> str:
    """Porter stem of one lower-case word."""
    if not word:
        return word
    return _Stemmer(word).run()
