"""Extract Poe's essay body without site navigation or editorial notes."""
import re
from html.parser import HTMLParser

ESSAY_URL = 'https://www.eapoe.org/works/essays/philcomp.htm'
ESSAY_NAME = 'The Philosophy of Composition (1846)'


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
