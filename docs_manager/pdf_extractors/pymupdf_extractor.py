import re

import pymupdf.layout
import pymupdf4llm
import tabulate
from pymupdf4llm.helpers.document_layout import create_list_item_levels, list_item_to_text, text_to_text

from docs_manager.pdf_extractors.pdf_extractor import PdfExtractor


class PyMuPDFExtractor(PdfExtractor):
    def from_file(self, file_path):
        with pymupdf.open(file_path) as pdf:
            try:
                return self._to_text(pymupdf4llm.parse_document(pdf))
            except:
                # output = '<BASIC_EXTRACTION>\n'
                # for page in pdf:
                #     output += page.get_text()
                return ''

    def _to_text(self, parsed_doc):
        doc_blocks = self._to_blocks(parsed_doc)

        doc_blocks = self._remove_possible_headers(doc_blocks)

        blocks = [block for page_blocks in doc_blocks for block in page_blocks]
        blocks = self._ignore_since_sections(blocks, [
            'CRediT authorship contribution statement',
            'Credit author statement',
            'Declaration of competing interest',
            'Funding',
            'Data availability',
            'Acknowledgements',
            'Acknowledgement',
            # 'About the authors', # conflict
            'Notes on contributor',
            'References',
            'References Cited',
            re.compile(r'appendix( .+)?'),
        ])

        return ''.join([self._out(blocks, i) for i in range(len(blocks))])

    def _to_blocks(self, parsed_doc):
        doc_blocks = []

        for page_no, page in enumerate(parsed_doc.pages, start=1):
            page_blocks = []

            list_item_levels = create_list_item_levels(page.boxes)
            for i, box in enumerate(page.boxes):
                btype = box.boxclass
                if btype in ('page-header',
                             'page-footer',
                             'picture',
                             'formula',
                             'footnote',
                             'caption'):
                    continue

                if btype == 'list-item':
                    if len(box.textlines) > 0:
                        btext = list_item_to_text(box.textlines, list_item_levels[i])
                    else:
                        continue
                elif btype == 'table':
                    if page_no > 2:
                        continue

                    table = box.table['extract']
                    if len(table) > 0 and len(table[0]) > 0 and table[0][0].strip().lower() == 'abstract':
                        btext = (
                                tabulate.tabulate(table, tablefmt='plain')
                                + '\n\n'
                        )
                    else:
                        continue
                else:
                    btext = text_to_text(
                        box.textlines, ignore_code=True or page.ocrpage
                    )

                page_blocks.append({
                    'page_no': page_no,
                    'type': btype,
                    'text': btext,
                })

            doc_blocks.append(page_blocks)

        return doc_blocks

    def _remove_possible_headers(self, doc_blocks):
        possible_headers = []
        for page_blocks in doc_blocks:
            if len(page_blocks) > 0 and page_blocks[0]['type'] in ('section-header', 'text'):
                possible_headers.append(re.sub(r'^\s*\d*\s*|\s*\d*\s*$', '', page_blocks[0]['text'].lower()))
            else:
                possible_headers.append(False)
        for i in range(len(possible_headers)):
            if isinstance(possible_headers[i], bool):
                continue

            detected = False

            for j in range(i + 1, len(possible_headers)):
                if isinstance(possible_headers[j], bool):
                    continue
                if possible_headers[j] == possible_headers[i]:
                    doc_blocks[j] = doc_blocks[j][1:]
                    possible_headers[j] = False
                    detected = True

            if detected:
                doc_blocks[i] = doc_blocks[i][1:]

        return doc_blocks

    def _ignore_since_sections(self, blocks: list, section_names: list):
        for i in range(len(blocks)):
            block = blocks[i]
            if block['type'] == 'section-header':
                for section_name in section_names:
                    if isinstance(section_name, str):
                        if block['text'].strip().lower() == section_name.lower():
                            return blocks[:i]
                    elif isinstance(section_name, re.Pattern):
                        if section_name.fullmatch(block['text'].strip().lower()) is not None:
                            return blocks[:i]

        return blocks

    def _out(self, blocks, index, debug = False):
        if not debug:
            return blocks[index]['text']

        block = blocks[index]
        btype = block['type']
        btext = block['text']

        bpage = ''
        if index == 0 or block['page_no'] != blocks[index - 1]['page_no']:
            bpage = str(block['page_no']) + '\n'

        warned = btype not in ('list-item', 'table', 'section-header', 'text')
        btext = btext.strip()
        if len(btext) > 60:
            btext = btext[:25].strip() + ' ... ' + btext[-25:].strip()
        return f'{bpage}{btype}{'.WARN' if warned else ''}: {btext}' + '\n'
