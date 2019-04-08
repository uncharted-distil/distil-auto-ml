import json
import time
import pathlib
import logging
import datetime

import config

import main_utils as utils
import models

from server.server import Server
from server.export import export_run

from exline import pipeline as ex_pipeline
from exline.scoring import Scorer
from server import export

from d3m import runtime
from d3m.container import dataset
from d3m.metadata import pipeline

import pickle
import pandas as pd

QUATTO_LIVES = {}


# Configure output dir
pathlib.Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def save_job(runtime, task_id):
    filepath = utils.make_job_fn(task_id)
    with open(filepath, 'wb') as f:
        pickle.dump(runtime, f)

def produce_task(logger, session, task):
    try:
        logger.info('Starting produce task ID {}'.format(task.id))
        dats = QUATTO_LIVES[task.fit_solution_id]

        fitted_pipeline = dats['pipeline']
        test_dataset = dataset.Dataset.load(task.dataset_uri)
        results = ex_pipeline.produce(fitted_pipeline, test_dataset)

        test_dataset = test_dataset['learningData']
        predictions_df = pd.DataFrame(test_dataset['d3mIndex'])
        predictions_df[dats['target_name']] = results[dats['target_name']]

        preds_path = utils.make_preds_filename(task.id)
        predictions_df.to_csv(preds_path, index=False)

        session.commit()
    except Exception as e:
        logger.warn('Exception running task ID {}: {}'.format(task.id, e), exc_info=True)
        task.error = True
        task.error_message = str(e)
    finally:
        # Update DB with task results
        # and mark task 'ended' and when
        task.ended = True
        task.ended_at = datetime.datetime.utcnow()
        session.commit()

def score_task(logger, session, task):
    try:
        logger.info('Starting score task ID {}'.format(task.id))
        task.started_at = datetime.datetime.utcnow()
        score_config = session.query(models.ScoreConfig) \
                              .filter(models.ScoreConfig.id==task.score_config_id) \
                              .first()

        dats = QUATTO_LIVES[task.solution_id]
        scorer = Scorer(logger, task, score_config, dats)
        score_values = scorer.run()
        for score_value in score_values:
            score = models.Scores(
                solution_id=task.solution_id,
                score_config_id=score_config.id,
                value=score_value)
            session.add(score)
            session.commit()
    except Exception as e:
        logger.warn('Exception running task ID {}: {}'.format(task.id, e), exc_info=True)
        task.error = True
        task.error_message = str(e)
    finally:
        # Update DB with task results
        # and mark task 'ended' and when
        task.ended = True
        task.ended_at = datetime.datetime.utcnow()
        session.commit()


def exline_task(logger, session, task):
    try:


        logger.info('Starting exline task ID {}'.format(task.id))
        task.started_at = datetime.datetime.utcnow()
        prob = task.problem
        prob = json.loads(prob)
        target_col_name = False
        for target in prob['inputs'][0]['targets']:
            target['resource_id'] = target.pop('resourceId')
            target['column_index'] = target.pop('columnIndex')
            target['column_name'] = target.pop('columnName')
            if not target_col_name:
                target_col_name = target['column_name']
        prob['id'] = utils.generate_id()
        prob['digest'] = '__unset__'

        search_template = pipeline.Pipeline.from_json(task.pipeline) if task.pipeline else None
        pipe, dataset = ex_pipeline.create(task.dataset_uri, prob, search_template)
        fitted_pipeline, pipeline_run = ex_pipeline.fit(pipe, prob, dataset)

        pipeline_json = fitted_pipeline.pipeline.to_json()
        save_me = {'pipeline': fitted_pipeline, 'target_name': target_col_name}

        QUATTO_LIVES[task.id] = save_me
        save_job(save_me, task.id)
        task.pipeline = pipeline_json
        # TODO: fix below after validation, or put in Export call
        # pipeline_run.to_yaml('myfile')
    except Exception as e:
        logger.warn('Exception running task ID {}: {}'.format(task.id, e), exc_info=True)
        task.error = True
        task.error_message = str(e)
    finally:
        # Update DB with task results
        # and mark task 'ended' and when
        task.ended = True
        task.ended_at = datetime.datetime.utcnow()
        session.commit()


def job_loop(logger, session):
    task = False
    try:
        task = session.query(models.Tasks) \
                      .order_by(models.Tasks.created_at.asc()) \
                      .filter(models.Tasks.ended == False) \
                      .first()
    except Exception as e:
        logger.warn('Exception getting task: {}'.format(e), exc_info=True)
    # If there is work to be done...
    if task:
        if task.type == "EXLINE":
            exline_task(logger, session, task)
        elif task.type == "SCORE":
            score_task(logger, session, task)
        elif task.type == "PRODUCE":
            produce_task(logger, session, task)


def main(once=False):
    # override config vals D3M values
    export.override_config()

    # Set up logging
    logging_level = logging.DEBUG if config.DEBUG else logging.INFO
    system_version = utils.get_worker_version()
    logger = utils.setup_logging(logging_level,
                                 log_file=config.LOG_FILENAME,
                                 system_version=system_version)
    logger.info("System version {}".format(system_version))

    logging.basicConfig(level=logging.DEBUG)

    # Create and start the gRPC server
    server = Server()
    server.start(config.PORT)

    # Get DB access
    session = models.start_session(config.DB_LOCATION)

    # Main job loop
    while True:
        job_loop(logger, session)
        # Check for a new job every second
        time.sleep(1)


if __name__ == '__main__':
    main()
