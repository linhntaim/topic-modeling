import os
import re

from docs_manager.pdf_extractors.pdf_extractor import PdfExtractor
from docs_manager.pdf_extractors.pdfminer_extractor import PdfMinerExtractor


class DocsExtractor:
    def __init__(self, pdf_extractor: PdfExtractor = None):
        self._pdf_extractor: PdfExtractor = pdf_extractor if pdf_extractor is not None else PdfMinerExtractor()

    def from_db(self, docs_db, dir_path: str, raw_dir_path: str = None):
        if not os.path.isdir(dir_path):
            raise FileNotFoundError(f'Directory {dir_path} does not exist')
        if raw_dir_path is not None and not os.path.isdir(raw_dir_path):
            raw_dir_path = None

        # docs = docs_db.get_non_deleted()
        docs = docs_db.get_processed()
        for doc in docs:
            if raw_dir_path is None:
                doc_file_path = os.path.join(dir_path, doc['file'])
            else:
                doc_file_path = os.path.join(raw_dir_path, doc['file'] + '.txt')
                if not os.path.isfile(doc_file_path):
                    doc_file_path = os.path.join(dir_path, doc['file'])
            doc['text'] = self._get_text(doc_file_path)

        print(f'Got {len(docs)} docs')
        return docs

    def _get_text(self, file_path):
        file_name = os.path.basename(file_path)
        ext = file_name[-4:]
        if ext == '.txt':
            file_name = file_name[:-4]

        cache_file_path = './.cache/clean/' + file_name + '.txt'
        if os.path.isfile(cache_file_path):
            # print('- Get doc from "%s"' % file_name)
            text = open(cache_file_path, 'r', encoding='utf-8').read()
            # print('  (cache)')
            return text

        text = self._extract_text(file_path)

        print('- Clean doc from "%s"' % file_name)
        text = self._clean_text(text)
        print('  DONE')

        with open(cache_file_path, 'w', encoding='utf-8') as f:
            f.write(text)

        return text

    def _extract_text(self, file_path):
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f'File {file_path} does not exist')

        file_name = os.path.basename(file_path)
        ext = file_name[-4:]
        if ext == '.txt':
            file_name = file_name[:-4]

        cache_file_path = './.cache/raw/' + file_name + '.txt'
        if os.path.isfile(cache_file_path):
            # print('- Get doc from "%s"' % file_name)
            text = open(cache_file_path, 'r', encoding='utf-8').read()
            # print('  (cache)')
            return text

        print('- Get doc from "%s"' % file_name)
        if ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        elif ext == '.pdf':
            text = self._pdf_extractor.from_file(file_path)
        else:
            raise NotImplementedError(f'File type {ext} is not supported')
        print('  DONE')

        with open(cache_file_path, 'w', encoding='utf-8') as f:
            f.write(text)

        return text

    def _clean_text(self, text):
        # Remove journal info
        text = re.sub(r'Contents lists available at.*', '', text)
        text = re.sub(r'j *o *u *r *n *a *l *h *o *m *e *p *a *g *e *:.*', '', text)
        # Remove special chars
        text = re.sub(r'“', '"', text)
        text = re.sub(r'”', '"', text)
        text = re.sub(r'″', '"', text)
        text = re.sub(r'„', '"', text)
        text = re.sub(r'′′', '"', text)
        text = re.sub(r'‘‘', '"', text)
        text = re.sub(r'’', '\'', text)
        text = re.sub(r'′', '\'', text)
        text = re.sub(r'ʽ', '\'', text)
        text = re.sub(r'ʹ', '\'', text)
        text = re.sub(r'‘', '\'', text)
        text = re.sub(r'‑', '-', text)
        text = re.sub(r'­', '-', text)
        text = re.sub(r'：', ':', text)
        text = re.sub(r'¼', '=', text)
        text = re.sub(r'½', '[', text)
        text = re.sub(r'º', '°', text)
        text = re.sub(r'˚', '°', text)
        text = re.sub(r'，', ', ', text)
        text = re.sub(r'。', '. ', text)
        text = re.sub(r'∙', '.', text)
        text = re.sub(r'…', ' ... ', text)
        text = re.sub(r'⋯', ' ... ', text)
        text = re.sub(r'—', ' ', text)
        text = re.sub(r'―', ' ', text)
        text = re.sub(r'─', ' ', text)
        text = re.sub(r'–', '-', text)
        text = re.sub(r'−', '-', text)
        text = re.sub(r'þ', '+', text)
        text = re.sub(r'Þ', ')', text)
        text = re.sub(r'ð', '(', text)
        text = re.sub(r'œ', 'œ', text)
        text = re.sub(r'ﬂ', 'fl', text)
        text = re.sub(r'ﬁ', 'fi', text)
        text = re.sub(r'ı', 'i', text)
        text = re.sub(r'Ð', 'Đ', text)
        text = re.sub(r'Μ', 'M', text)
        text = re.sub(r'Ε', 'E', text)
        text = re.sub(r'𝑎𝑛𝑑', 'and', text)
        with open('.\\.special_chars.txt', 'r', encoding='utf-8') as f:
            special_chars = f.read()
            text = re.sub(r'[' + special_chars + ']+', '', text)
        # Spaces
        text = re.sub(r'\s+', ' ', text)
        # Words/Phrases which contain hyphen or slash and separated by line break
        text = re.sub(r'([a-zA-Z0-9]+[-/]|://) ([a-zA-Z0-9]+)', r'\1\2', text)
        # Remove Email
        text = re.sub(
            r'[a-zA-Z0-9!#$%&\'*+\-/=?^_`{|}~.]+@([a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?\.)+([a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?) \([^()]+\)',
            '', text)
        text = re.sub(
            r'[a-zA-Z0-9!#$%&\'*+\-/=?^_`{|}~.]+@([a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?\.)+([a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?)',
            '', text)
        # Remove URL
        text = re.sub(
            r'(https?://)?([a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?\.)+([a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?)(:\d+)?(/[a-zA-Z0-9\-._~!$&\'()*+,;=:@%]*)*(\?[a-zA-Z0-9\-._~!$&\'()*+,;=:@%]*)?(#[a-zA-Z0-9\-._~!$&\'()*+,;=:@%]*)?',
            '', text)
        # Remove in-text citation
        text = re.sub(
            r'\s*\([^()]+(\s+\d{4}(/(\d{2}|\d{4}))?(-\d{4}(/(\d{2}|\d{4}))?)?(\s*[a-z]{1,2}|[a-z],[a-z]|\([a-z]{1,2}|[a-z],[a-z]\)\b)?)(\s*|\s*[.,:;][^;()]*|\s+(in|accross|for)\s+[^;()]+)?\)',
            '', text)
        text = re.sub(r'\s*\(see\s+[^()]+\)', '', text)
        # Remove cid
        text = re.sub(r'\s*\(cid:\d+\)', '', text)
        # Remove page/table/figure/vol/no
        text = re.sub(r'\s+page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+pages\s+\d+(-| à )\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*\(?(tables?|figures?|fig\.?|pages?|volume|vol\.?|number|no\.?)\s+[a-zA-Z]*\d+[a-zA-Z]*\)?',
                      '', text, flags=re.IGNORECASE)
        # Words separated by '-': Remove '-'
        text = re.sub(r'\b([a-zA-Z]+)-(?=[a-zA-Z]+\b)', r'\1', text)
        # Underscore
        text = re.sub(r'_+', '_', text)

        return text
