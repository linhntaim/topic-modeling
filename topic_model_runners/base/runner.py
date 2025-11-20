import os
import shutil
import time
from datetime import datetime

from topic_model_runners.base.runner_option import RunnerOption


class Runner:
    def __init__(
            self,
            option: RunnerOption | list[RunnerOption] = None,
            cache_enabled: bool = True,
            naming_prefix: str = '',
    ):
        self._naming_prefix = naming_prefix

        self._created_at: datetime = datetime.now()
        self._output_dir: str = '.\\.output\\' + self._naming_prefix + self._created_at.strftime("%Y%m%d_%H%M%S")

        self._options = None
        self._option = None
        if isinstance(option, list):
            self._options = option
        else:
            self._option = option if option is not None else RunnerOption()

        self._cache_enabled = cache_enabled
        self._cache_dir = None
        self._cached = False

        self._store = {}

        self._result = None

    def set_output_dir(self, output_dir: str):
        self._output_dir = output_dir
        return self

    def get_option(self):
        return self._option if self._options is None else self._options

    def enable_cache(self, enabled: bool = True):
        self._cache_enabled = enabled
        return self

    def _cache_ready(self):
        if self._cache_enabled:
            self._cache_dir = self._generate_cache_dir(self.generate_cache_key())
            if os.path.isdir(self._cache_dir):
                self._cached = True
                print('(cached)')
            else:
                self._cached = False
                os.makedirs(self._cache_dir, exist_ok=True)
        else:
            self._cache_dir = None
            self._cached = False

    def _cache_clear(self):
        shutil.rmtree(self._cache_dir, ignore_errors=True)

    def generate_cache_key(self) -> str:
        return self._naming_prefix + str(self._option)

    def _generate_cache_dir(self, cache_key: str) -> str:
        return '.\\.cache\\runner\\' + cache_key

    def run(self):
        os.makedirs(self._output_dir, exist_ok=True)
        print('Start @ %s' % self._created_at.strftime('%Y-%m-%d %H:%M:%S'))

        self._execute()

        duration = datetime.now().timestamp() - self._created_at.timestamp()
        print('Total run time: %d sec' % duration)

    def _execute(self):
        self._before_execution()

        if self._options is None:
            self._result = self._run()
        else:
            self._result = []
            num_options = len(self._options)
            for i, option in enumerate(self._options):
                print('Option %d / %d' % (i + 1, num_options))
                start_at = time.time()

                self._option = option
                result = self._run()
                self._result.append(result)

                duration = time.time() - start_at
                print(f'Option time: {duration:.0f} sec')

        self._after_execution()

    def _before_execution(self):
        return

    def _run(self):
        self._cache_ready()

        try:
            result = self._make()
            evaluation = self._evaluate(result)

            return result, evaluation, self._option
        except Exception as e:
            self._cache_clear()

            raise e

    def _make(self):
        return None

    def _evaluate(self, result):
        return None

    def _after_execution(self):
        return
