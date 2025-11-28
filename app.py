import glob
import json
import os.path
import re
import warnings

import questionary
import typer
from dotenv import load_dotenv
from questionary import Choice
from typing_extensions import Annotated

from docs_manager.docs_db import DocsDB
from docs_manager.docs_extractor import DocsExtractor
from docs_manager.pdf_extractors.pymupdf_extractor import PyMuPDFExtractor
from topic_model_runners.base.dtm_option import DtmOption
from topic_model_runners.gensim_lda_option import GensimLdaOption
from topic_model_runners.gensim_lda_runner import GensimLdaRunner
from topic_model_runners.mallet_lda_option import MalletLdaOption
from topic_model_runners.mallet_lda_runner import MalletLdaRunner

warnings.filterwarnings('ignore')

# Load environment variables from .env file
load_dotenv()

app = typer.Typer()


def to_bool(s):
    return s.lower() in ('yes', 'true', '1')


def to_int(s):
    return int(s)


def to_float(s):
    return float(s)


def to_float_or_string(s):
    return float(s) if re.search(r'[a-z]+', s) is None else s


def safe_question(question):
    value = question.ask()
    if value is None:
        raise typer.Exit()
    return value.strip()


def build_list_of_params(multi_value_params, param_names):
    list_of_params = []

    for param_name in param_names:
        list_of_params_by_name = [{param_name: value} for value in multi_value_params[param_name]]
        if len(list_of_params) == 0:
            list_of_params = list_of_params_by_name
            continue

        tmp_list_of_params = []
        for params_1 in list_of_params:
            for params_2 in list_of_params_by_name:
                tmp_list_of_params.append(params_1 | params_2)
        list_of_params = tmp_list_of_params

    return list_of_params


def run_topic_modeling(
        name,
        display_name,
        profile,
        param_converter,
        dtm_param_names,
        topic_model_params,
        create_topic_modeling_runner,
):
    profile = profile.strip()

    if profile == '':
        profile = safe_question(
            questionary.select(
                f'Choose a profile to run the {display_name} with:',
                choices=[os.path.basename(file_name)[:-5] for file_name in glob.glob(f'.\\profiles\\{name}\\*.json')],
                default='default',
            )
        )

    profile_file_path = f'.\\profiles\\{name}\\{profile}.json'
    if not os.path.isfile(profile_file_path):
        print(f'Profile {profile} not found.')
        raise typer.Exit()
    with open(profile_file_path, 'r', encoding='utf-8') as file:
        profile = json.loads(file.read())

    rangeable_params = ['num_topics']
    while True:
        ok_choice = Choice('OK', value='ok')
        choices = [
            Choice(f'{param}: [{', '.join([str(value) for value in values])}]', value=param)
            for param, values in profile.items()
        ]
        choices.append(ok_choice)

        param = safe_question(
            questionary.select(
                f'Pick a parameter to edit or OK to start running {display_name}:',
                choices=choices,
                default=ok_choice.value,
            )
        )

        if param == 'ok':
            print('\n'.join([choice.title for choice in choices if choice.value != 'ok']))
            break

        values = safe_question(
            questionary.text(
                f'{param}:',
                instruction='(values separated by spaces)',
            )
        )
        if values != '':
            if param in rangeable_params and ':' in values:
                nums = values.split(':')
                num_nums = len(nums)

                try:
                    if num_nums == 2:
                        profile[param] = list(range(int(nums[0]), int(nums[1])))
                    elif num_nums == 3:
                        profile[param] = list(range(int(nums[0]), int(nums[1]), int(nums[2])))
                    else:
                        raise 'Invalid number range'
                except:
                    print('Invalid number range')
                    continue
            else:
                profile[param] = [param_converter[param](value) for value in re.split(r'\s+', values)]

    list_of_dtm_params = build_list_of_params(
        profile,
        dtm_param_names,
    )
    list_of_topic_model_params = build_list_of_params(
        profile,
        topic_model_params,
    )

    for dtm_params in list_of_dtm_params:
        create_topic_modeling_runner(
            DocsExtractor(PyMuPDFExtractor()).from_db(
                DocsDB(os.getenv('DOCS_FILE')),
                os.getenv('DOCS_DIR'),
                os.getenv('DOCS_RAW_DIR')
            ),
            dtm_params,
            list_of_topic_model_params
        ).run()
    print('\a')


@app.command()
def mallet(
        profile: Annotated[
            str,
            typer.Option(help='Profile to run topic modeling.'),
        ] = '',
):
    run_topic_modeling(
        'mallet',
        'LDA MALLET',
        profile,
        {
            'ngram': lambda s: to_bool(s),
            'tfidf': lambda s: to_bool(s),
            'abbr_as_ngram': lambda s: to_bool(s),

            'num_topics': lambda s: to_int(s),
            'terms_per_topic': lambda s: to_int(s),
            'alpha': lambda s: to_float(s),
            'beta': lambda s: to_float(s),
            'random_seed': lambda s: to_int(s),
            'num_iterations': lambda s: to_int(s),
            'optimize_interval': lambda s: to_int(s),
            'optimize_burn_in': lambda s: to_int(s),
        },
        ['ngram', 'tfidf', 'abbr_as_ngram'],
        [
            'num_topics',
            'terms_per_topic',
            'alpha',
            'beta',
            'random_seed',
            'num_iterations',
            'optimize_interval',
            'optimize_burn_in',
        ],
        lambda docs, dtm_params, list_of_topic_model_params: MalletLdaRunner(
            docs,
            DtmOption(dtm_params),
            [MalletLdaOption(topic_model_params) for topic_model_params in list_of_topic_model_params],
            cache_enabled=True
        ),
    )


@app.command()
def gensim(
        profile: Annotated[
            str,
            typer.Option(help='Profile to run topic modeling.'),
        ] = '',
):
    run_topic_modeling(
        'gensim',
        'LDA Gensim',
        profile,
        {
            'ngram': lambda s: to_bool(s),
            'tfidf': lambda s: to_bool(s),
            'abbr_as_ngram': lambda s: to_bool(s),

            'num_topics': lambda s: to_int(s),
            'terms_per_topic': lambda s: to_int(s),
            'alpha': lambda s: to_float_or_string(s),
            'beta': lambda s: to_float_or_string(s),
            'random_seed': lambda s: to_int(s),
            'chunksize': lambda s: to_int(s),
            'passes': lambda s: to_int(s),
            'update_every': lambda s: to_int(s),
            'iterations': lambda s: to_int(s),
        },
        ['ngram', 'tfidf', 'abbr_as_ngram'],
        [
            'num_topics',
            'terms_per_topic',
            'alpha',
            'beta',
            'random_seed',
            'chunksize',
            'passes',
            'update_every',
            'iterations',
        ],
        lambda docs, dtm_params, list_of_topic_model_params: GensimLdaRunner(
            docs,
            DtmOption(dtm_params),
            [GensimLdaOption(topic_model_params) for topic_model_params in list_of_topic_model_params],
            cache_enabled=True
        ),
    )


if __name__ == '__main__':
    app()
