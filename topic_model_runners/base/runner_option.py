class RunnerOption:
    def __init__(self, option: dict = None):
        if option is None:
            option = {}

        self._option = option
        self._option_str = {}

    def __getattr__(self, item):
        if item in self._option:
            return self._option[item]
        return None

    def _define(self, name: str, default_value, name_str: str | bool = None):
        if name not in self._option or self._option[name] is None:
            self._option[name] = default_value
        if name_str != False:
            self._option_str[name] = name_str if name_str is not None else name
        return self

    def __str__(self):
        return '_'.join(
            [self._option_str[name] + str((1 if value else 0) if isinstance(value, bool) else value) for name, value in
             self._option.items() if name in self._option_str])
