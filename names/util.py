from normality import ascii_text, category_replace

UNICODE_CATEGORIES = {
    'Cc': None,
    'Cf': None,
    'Cs': None,
    'Co': None,
    'Cn': None,
    'Lm': None,
    'Mn': None,
    'Mc': None,
    'Me': '\'',
    'Zs': ' ',
    'Zl': ' ',
    'Zp': ' ',
    'Pc': '.',
    'Pd': '.',
    'Ps': '.',
    'Pe': '.',
    'Pi': '.',
    'Pf': '.',
    'Po': '.',
    'Sc': '.',
    'Sk': '.',
    'So': '.'
}


def normalize(text):
    text = category_replace(text, replacements=UNICODE_CATEGORIES)
    text = ascii_text(text)
    if text is not None:
        return text.lower()
