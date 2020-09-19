from api.utils import ValueType
from importlib.abc import ExecutionLoader
import os
import uuid
import logging
import subprocess
import pathlib

import config

logger = logging.getLogger('distil')

def generate_id():
    _id = str(uuid.uuid4())
    return _id

def setup_logging(logging_level, log_file='distil.log', system_version="UNSET"):
    # Setup logger and handlers
    logger = logging.getLogger('distil')
    # Don't propagate to any other loggers to dedupe logging due to d3m package
    logger.propagate = False

    fh = logging.FileHandler(log_file)
    ch = logging.StreamHandler()

    # Set logging level
    logger.setLevel(logging_level)
    fh.setLevel(logging_level)
    ch.setLevel(logging_level)

    # Log formatter
    # LEVEL - TIMESTAMP - VERSION - MODULE - MESSAGE
    sys_string = '%(levelname)s - %(asctime)s - v-{} - %(name)s - %(message)s'.format(system_version)
    log_formatter = logging.Formatter(sys_string)
    fh.setFormatter(log_formatter)
    ch.setFormatter(log_formatter)

    # Add setup handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def get_worker_version():
    """
    Outputs git branch name and hash for first log.
    Equivalent: `echo $(git rev-parse --abbrev-ref HEAD)-$(git rev-parse HEAD)`
    Running this in NOT a git repo will not be fun!
    """
    sha = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=".")
    branch = os.getenv('BRANCH_NAME', False)
    if not branch or branch == '__UNSET__':
        branch = subprocess.check_output(['git',
                                          'rev-parse',
                                          '--abbrev-ref',
                                          'HEAD'], cwd=".")
        branch = branch.decode('ascii').strip()
    return branch + "-" + sha.decode('ascii').strip()

def make_job_fn(task_id):
    filename = task_id + '.dill'
    filepath = pathlib.Path(config.OUTPUT_DIR, filename)
    return filepath.resolve()

def make_preds_filename(task_id, output_key=None, output_type=None):
    """Return the absolute path to the predictions filename.

    None of that os.path.join garbage, just good sweet Path.absolute()

    Down with os.path.join. I have learned my lesson.
    """
    if output_type == ValueType.PARQUET_URI:
        extension = 'parquet'
    elif output_type == ValueType.CSV_URI:
        extension = 'csv'
    else:
        extension = 'csv'
        logger.warn(f"unhandled output type '{output_type}'.  Default to '{ValueType.CSV_URI}'")

    if output_key is not None:
        path = pathlib.Path(config.OUTPUT_DIR, f'{format(task_id)}_{output_key}.{extension}').resolve()
    else:
        path = pathlib.Path(config.OUTPUT_DIR, f'{format(task_id)}.{extension}').resolve()

    return path
