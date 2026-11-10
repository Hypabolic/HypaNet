// The Porter (1980) stemmer, as defined by Martin Porter's ANSI C reference version.
// Mirrors ui/porter.py line for line, including the three -DEPARTURE- points of the C source (bli -> ble,
// logi -> log, words of length 1 or 2 left alone). Input is one lower-case word; it is handled as an array of code
// points so that astral characters count once, as in Python. No dependencies.

class Stemmer {
  constructor(word) {
    this.b = Array.from(word);
    this.k = this.b.length - 1;
    this.j = 0;
  }

  cons(i) {
    const ch = this.b[i];
    if (ch === 'a' || ch === 'e' || ch === 'i' || ch === 'o' || ch === 'u') return false;
    if (ch === 'y') return i === 0 ? true : !this.cons(i - 1);
    return true;
  }

  // Number of consonant sequences between 0 and j.
  m() {
    let n = 0;
    let i = 0;
    const j = this.j;
    for (;;) {
      if (i > j) return n;
      if (!this.cons(i)) break;
      i++;
    }
    i++;
    for (;;) {
      for (;;) {
        if (i > j) return n;
        if (this.cons(i)) break;
        i++;
      }
      i++;
      n++;
      for (;;) {
        if (i > j) return n;
        if (!this.cons(i)) break;
        i++;
      }
      i++;
    }
  }

  vowelinstem() {
    for (let i = 0; i <= this.j; i++) if (!this.cons(i)) return true;
    return false;
  }

  doublec(j) {
    if (j < 1) return false;
    if (this.b[j] !== this.b[j - 1]) return false;
    return this.cons(j);
  }

  cvc(i) {
    if (i < 2 || !this.cons(i) || this.cons(i - 1) || !this.cons(i - 2)) return false;
    const ch = this.b[i];
    return !(ch === 'w' || ch === 'x' || ch === 'y');
  }

  ends(s) {
    const length = s.length;
    if (s[length - 1] !== this.b[this.k]) return false;
    if (length > this.k + 1) return false;
    for (let t = 0; t < length; t++) if (this.b[this.k - length + 1 + t] !== s[t]) return false;
    this.j = this.k - length;
    return true;
  }

  setto(s) {
    const j = this.j;
    this.b.length = j + 1;
    for (const ch of s) this.b.push(ch);
    this.k = j + s.length;
  }

  r(s) {
    if (this.m() > 0) this.setto(s);
  }

  step1ab() {
    if (this.b[this.k] === 's') {
      if (this.ends('sses')) this.k -= 2;
      else if (this.ends('ies')) this.setto('i');
      else if (this.b[this.k - 1] !== 's') this.k--;
    }
    if (this.ends('eed')) {
      if (this.m() > 0) this.k--;
    } else if ((this.ends('ed') || this.ends('ing')) && this.vowelinstem()) {
      this.k = this.j;
      if (this.ends('at')) this.setto('ate');
      else if (this.ends('bl')) this.setto('ble');
      else if (this.ends('iz')) this.setto('ize');
      else if (this.doublec(this.k)) {
        this.k--;
        const ch = this.b[this.k];
        if (ch === 'l' || ch === 's' || ch === 'z') this.k++;
      } else if (this.m() === 1 && this.cvc(this.k)) this.setto('e');
    }
  }

  step1c() {
    if (this.ends('y') && this.vowelinstem()) this.b[this.k] = 'i';
  }

  step2() {
    switch (this.b[this.k - 1]) {
      case 'a':
        if (this.ends('ational')) this.r('ate');
        else if (this.ends('tional')) this.r('tion');
        break;
      case 'c':
        if (this.ends('enci')) this.r('ence');
        else if (this.ends('anci')) this.r('ance');
        break;
      case 'e':
        if (this.ends('izer')) this.r('ize');
        break;
      case 'l':
        if (this.ends('bli')) this.r('ble'); // -DEPARTURE- (paper: abli -> able)
        else if (this.ends('alli')) this.r('al');
        else if (this.ends('entli')) this.r('ent');
        else if (this.ends('eli')) this.r('e');
        else if (this.ends('ousli')) this.r('ous');
        break;
      case 'o':
        if (this.ends('ization')) this.r('ize');
        else if (this.ends('ation')) this.r('ate');
        else if (this.ends('ator')) this.r('ate');
        break;
      case 's':
        if (this.ends('alism')) this.r('al');
        else if (this.ends('iveness')) this.r('ive');
        else if (this.ends('fulness')) this.r('ful');
        else if (this.ends('ousness')) this.r('ous');
        break;
      case 't':
        if (this.ends('aliti')) this.r('al');
        else if (this.ends('iviti')) this.r('ive');
        else if (this.ends('biliti')) this.r('ble');
        break;
      case 'g':
        if (this.ends('logi')) this.r('log'); // -DEPARTURE- (not in the paper)
        break;
      default:
        break;
    }
  }

  step3() {
    switch (this.b[this.k]) {
      case 'e':
        if (this.ends('icate')) this.r('ic');
        else if (this.ends('ative')) this.r('');
        else if (this.ends('alize')) this.r('al');
        break;
      case 'i':
        if (this.ends('iciti')) this.r('ic');
        break;
      case 'l':
        if (this.ends('ical')) this.r('ic');
        else if (this.ends('ful')) this.r('');
        break;
      case 's':
        if (this.ends('ness')) this.r('');
        break;
      default:
        break;
    }
  }

  step4() {
    let ok;
    switch (this.b[this.k - 1]) {
      case 'a': ok = this.ends('al'); break;
      case 'c': ok = this.ends('ance') || this.ends('ence'); break;
      case 'e': ok = this.ends('er'); break;
      case 'i': ok = this.ends('ic'); break;
      case 'l': ok = this.ends('able') || this.ends('ible'); break;
      case 'n': ok = this.ends('ant') || this.ends('ement') || this.ends('ment') || this.ends('ent'); break;
      case 'o':
        ok = (this.ends('ion') && this.j >= 0 && (this.b[this.j] === 's' || this.b[this.j] === 't')) || this.ends('ou');
        break;
      case 's': ok = this.ends('ism'); break;
      case 't': ok = this.ends('ate') || this.ends('iti'); break;
      case 'u': ok = this.ends('ous'); break;
      case 'v': ok = this.ends('ive'); break;
      case 'z': ok = this.ends('ize'); break;
      default: ok = false;
    }
    if (ok && this.m() > 1) this.k = this.j;
  }

  step5() {
    this.j = this.k;
    if (this.b[this.k] === 'e') {
      const a = this.m();
      if (a > 1 || (a === 1 && !this.cvc(this.k - 1))) this.k--;
    }
    if (this.b[this.k] === 'l' && this.doublec(this.k) && this.m() > 1) this.k--;
  }

  run() {
    if (this.k <= 1) return this.b.join(''); // -DEPARTURE- strings of length 1 or 2 are left alone
    this.step1ab();
    if (this.k > 0) {
      this.step1c();
      this.step2();
      this.step3();
      this.step4();
      this.step5();
    }
    return this.b.slice(0, this.k + 1).join('');
  }
}

// Porter stem of one lower-case word.
export function stem(word) {
  if (!word) return word;
  return new Stemmer(word).run();
}
