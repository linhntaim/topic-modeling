import re
from itertools import chain

from docs_manager.docs_extractor import DocsExtractor


class SpecialCharsExtractor(DocsExtractor):
    def from_db(self, docs_db, dir_path: str, raw_dir_path: str = None):
        docs = super().from_db(docs_db, dir_path, raw_dir_path)

        special_chars = list(
            set(
                list(
                    chain.from_iterable([
                        list(self._remove_known_chars(doc['text'])) for doc in docs
                    ])
                )
            )
        )

        print(f'Special Chars: {len(special_chars)}')
        print(special_chars)

        with open('.\\.special_chars.txt', 'w', encoding='utf-8') as f:
            f.write(''.join(special_chars))

        with open('.\\.special_chars.txt', 'r', encoding='utf-8') as f:
            special_chars = f.read()
            print(special_chars)

        return docs

    def _get_text(self, file_path):
        return self._extract_text(file_path)

    def _remove_ascii_chars(self, text):
        return re.sub(r'[a-zA-Z0-9\s`~!@#$%^&*()\-_=+\[{\]}\\|;:\'",<.>/?]+', '', text)

    def _remove_known_chars(self, text):
        regexes = [
            # ASCII
            re.compile(r'[a-zA-Z0-9\s`~!@#$%^&*()\-_=+\[{\]}\\|;:\'",<.>/?]+'),
            # Known special chars
            re.compile(r'[“”″„’′ʽʹ‘‑，。∙…⋯−–─—―­þÞð：¼½°˚ﬂﬁıÐΜΕ]+'),
            re.compile(r'[œæǎāäåćçÇëęñöüÖøïțğḥīÜîṣšžßșŞᅡᆯᅮᅩᄉᆼᄀᄆ]+'),
            re.compile(r'[¥£₦￥Ȼ¢₹ȼ€]+'),
            re.compile(r'[𝑎𝑐𝑑𝑒𝑖𝑗𝑘𝑙𝑚𝑛𝑝𝑞𝑟𝑠𝑡𝑢𝑣𝑤𝑥𝑦𝑧𝐴𝐵𝐶𝐷𝐸𝐹𝐺𝐻𝐼𝐾𝑀𝑁𝑂𝑃𝑄𝑅𝑆𝑇𝑈𝑉𝑊𝑋𝑌𝑍𝛼𝜌𝜏𝜂𝛽απθβϕσφγελχρɛΔ∆ξƩηωτµƐμδ∑ΖŸ훿휀훼휋훽휎]+'),
            re.compile(r'[⊆∗⊉∈≠≈±∼∪∕→²₂∣≥⫆⊗∃×√§⁓≤³‰∞⋅÷]+'),
            # CJK
            re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df]+', flags=re.UNICODE),
            # Arabic
            re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+', flags=re.UNICODE),
            # Cyrillic
            re.compile(r'[\u0400-\u04FF]+', flags=re.UNICODE),
            # Thailand
            re.compile(r'[\u0E00-\u0E7F]+', flags=re.UNICODE),
            # Laos
            re.compile(r'[\u0E80-\u0EFF]+', flags=re.UNICODE),
            # Khmer
            re.compile(r'[\u1780-\u17FF]+', flags=re.UNICODE),
            # Vietnam
            re.compile(r'[ĂăÂâĐđÊêÔôƠơƯưÁáÀàÃãẢảẠạẤấẦầẪẫẨẩẬậẮắẰằẴẵẲẳẶặÉéÈèẼẽẺẻẸẹẾếỀềỄễỂểỆệÍíÌìĨĩỈỉỊịÓóÒòÕõỎỏỌọỐốỒồỖỗỔổỘộỚớỜờỠỡỞởỢợÚúÙùŨũỦủỤụỨứỪừỮữỬửỰựÝýỲỳỸỹỶỷỴỵ]+'),
        ]
        for regex in regexes:
            text = regex.sub('', text)
        return text
