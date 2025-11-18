import json


class English:
    def __init__(self):
        self.__us_to_gb_mapping = {}
        self.__us_to_gb_prefix_mapping = {}

    def prepare_us_to_gb(self):
        # https://github.com/hyperreality/American-British-English-Translator
        with open('.\\.american_spellings.json') as json_file:
            self.__us_to_gb_mapping = json.load(json_file)
        with open('.\\.american_prefix_spellings.json') as json_file:
            self.__us_to_gb_prefix_mapping = json.load(json_file)

    def convert_us_to_gb(self, token: str) -> str:
        # Support n-grams
        if token.find('_') != -1:
            return '_'.join([self.convert_us_to_gb(t) for t in token.split('_')])

        if token in self.__us_to_gb_mapping:
            return self.__us_to_gb_mapping[token]

        for us_prefix, gb_prefix in self.__us_to_gb_prefix_mapping.items():
            if us_prefix == token[:len(us_prefix)]:
                return gb_prefix + token[len(us_prefix):]

        return token
