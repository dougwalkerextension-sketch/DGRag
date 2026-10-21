"""Extract Poe's essay body without site navigation or editorial notes."""
import re
import unicodedata
from html.parser import HTMLParser

ESSAY_URL = 'https://www.eapoe.org/works/essays/philcomp.htm'
ESSAY_NAME = 'The Philosophy of Composition (1846)'


def title_key(text):
    text = unicodedata.normalize('NFKD', text).casefold()
    return ' '.join(re.findall(r'[a-z0-9]+', text))


def anthology_sections(text, ebook_id):
    """Match contents entries to actual body headings; never split on a TOC mention."""
    contents = re.search(r'^Contents\s*\n(?P<entries>(?:[ \t]+[^\n]+\n)+)', text, re.MULTILINE)
    if not contents:
        raise ValueError('Anthology contents are missing; cannot identify separate works')
    editorial = {'preface', 'life of poe', 'death of poe', 'notes to the second volume'}
    aliases = {'the unparalleled adventures of one hans pfaall':
               'the unparalleled adventures of one hans pfaal', 'william wilso': 'william wilson'}

    def heading_key(title):
        return title_key(re.sub(r'\(\*\d+\)', '', title))

    headings = {}
    for match in re.finditer(r'^[^\s][^\n]*$', text[contents.end():], re.MULTILINE):
        label = match.group()
        if label.upper() == label and any(c.isalpha() for c in label):
            headings.setdefault(heading_key(label), []).append((contents.end() + match.start(), label))
    boundaries = []
    for entry in contents['entries'].splitlines():
        key = heading_key(entry)
        if key in editorial and key != 'notes to the second volume':
            continue
        key = aliases.get(key, key)
        candidates = headings.get(key, [])
        if len(candidates) != 1:
            raise ValueError('Cannot uniquely locate anthology work: ' + entry.strip())
        start, label = candidates[0]
        title = re.sub(r'\s*\(\*\d+\)', '', label).strip().rstrip('.')
        boundaries.append((start, title, 'editorial' if key in editorial else 'literary-work'))
    if not boundaries or any(a[0] >= b[0] for a, b in zip(boundaries, boundaries[1:])):
        raise ValueError('Anthology work headings do not follow the contents order')
    # Keep introductory material separate from Poe's fictional speakers.
    sections = [{'title': 'Front matter and editorial introduction', 'start': 0,
                 'end': boundaries[0][0], 'work_type': 'editorial'}]
    sections += [{'title': title, 'start': start,
                  'end': boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text),
                  'work_type': kind} for i, (start, title, kind) in enumerate(boundaries)]
    for section in sections:
        section['work_id'] = ebook_id + '-' + title_key(section['title']).replace(' ', '-')
    return sections


def mentioned_works(question, history, catalog):
    """Explicit titles win; otherwise use the most recent user turn naming a work."""
    for content in [question] + [m['content'] for m in reversed(history) if m['role'] == 'user']:
        query = ' ' + title_key(content) + ' '
        found = []
        for work in catalog:
            key = title_key(work['work_title'])
            keys = {key, key.removeprefix('the ')}
            if 'hans pfaal' in key:
                keys.add(key.replace('hans pfaal', 'hans pfaall'))
            if any(' ' + candidate + ' ' in query for candidate in keys):
                found.append(work)
        if found:
            return found
    return []


class EssayParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.skip = 0
        self.parts = []
        self.complete = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if not self.depth:
            if tag == 'div' and 'pnltextm' in attrs.get('class', '').split():
                if self.complete:
                    raise ValueError('Essay page has multiple text bodies')
                self.depth = 1
            return
        if tag == 'div':
            self.depth += 1
        if self.skip or tag in ('script', 'style') or 'pgloc' in attrs.get('class', '').split():
            if tag not in ('br', 'hr', 'img', 'meta', 'link', 'input'):
                self.skip += 1
        if tag in ('p', 'h1', 'br') and not self.skip:
            self.parts.append('\n')

    def handle_endtag(self, tag):
        if not self.depth:
            return
        if self.skip:
            self.skip -= 1
        elif tag in ('p', 'h1'):
            self.parts.append('\n')
        if tag == 'div':
            self.depth -= 1
            if not self.depth:
                self.complete = True

    def handle_data(self, data):
        if self.depth and not self.skip:
            self.parts.append(re.sub(r'\s+', ' ', data))


def essay_body(raw):
    parser = EssayParser()
    parser.feed(raw)
    parser.close()
    # Collapse layout whitespace inside paragraphs, retaining paragraph boundaries.
    text = '\n\n'.join(' '.join(p.split()) for p in ''.join(parser.parts).split('\n') if p.strip())
    checks = ('THE PHILOSOPHY OF COMPOSITION', 'CHARLES DICKENS',
              'Mournful and Never-ending Remembrance', 'Shall be lifted')
    if not parser.complete or len(text) < 20000 or not all(term in text for term in checks):
        raise ValueError('Missing or incomplete Philosophy of Composition essay body')
    return text
