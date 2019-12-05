import json
import time
import pathlib
import logging
import datetime
import io
import jsonschema
from google.protobuf import json_format

import config

import utils
import models
import uuid
from server.server import Server
from server.export import export_run

from processing import pipeline as ex_pipeline
from processing.scoring import Scorer
from server import export
import api.utils as api_utils
from api import problem_pb2

from d3m import runtime
from d3m.container import dataset
from d3m.metadata import pipeline, problem, pipeline_run

import pickle
import pandas as pd

QUATTO_LIVES = {}


# Configure output dir
pathlib.Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def produce_task(logger, session, task):
    try:
        logger.info('Starting produce task ID {}'.format(task.id))
        dats = QUATTO_LIVES[task.solution_id]


        fitted_pipeline = dats['pipeline']
        test_dataset = dataset.Dataset.load(task.dataset_uri)
        results = ex_pipeline.produce(fitted_pipeline, test_dataset)

        # pull out the results the caller requested, ignore any others that were exposed
        output_keys = json.loads(task.output_keys)
        for output_key in output_keys:
            if output_key in results.values:
                preds_path = utils.make_preds_filename(task.fit_solution_id, output_key=output_key)
                results.values[output_key].to_csv(preds_path, index=False)
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

def fit_task(logger, session, task):
    try:
        logger.info('Starting task task ID {}'.format(task.id))
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
        logger.info('Starting distil task ID {}'.format(task.id))
        task.started_at = datetime.datetime.utcnow()

        # load the problem supplied by the search request into a d3m Problem type if one is provided
        if task.problem:
            problem_proto = json_format.Parse(task.problem, problem_pb2.ProblemDescription())
            problem_d3m = api_utils.decode_problem_description(problem_proto)

            target_name = problem_d3m['inputs'][0]['targets'][0]['column_name']
        else:
            problem_d3m = None
            target_name = None

        resolver = pipeline.Resolver(load_all_primitives=False) # lazy load
        search_template = pipeline.Pipeline.from_json(task.pipeline, resolver=resolver) if task.pipeline else None
        pipe, dataset = ex_pipeline.create(task.dataset_uri, problem_d3m, search_template, resolver)

        # Check to see if this is a fully specified pipeline.  If so, we'll run it as a non-standard since
        # it doesn't need to be serialized.
        run_as_standard = not ex_pipeline.is_fully_specified(search_template)
        fitted_pipeline, result = ex_pipeline.fit(pipe, problem_d3m, dataset, is_standard_pipeline=run_as_standard)

        pipeline_json = fitted_pipeline.pipeline.to_json()
        str_buf = io.StringIO()
        try:
            result.pipeline_run.to_yaml(str_buf)
            pipeline_run_yaml = str_buf.getvalue()
        except jsonschema.exceptions.ValidationError as v:
            # If a conforming result wasn't returned validation will fail.  Most common case for this is
            # running an analytic as a fully specificed pipeline that returned a dataframe with out any
            # rows (an empty result), or no dataframe at all (another possible way to express an empty result).
            # In this case, we'll set the run results to None which is properly handled downstream.
            pipeline_run_yaml = None
            logger.warn('Could not parse result')

        save_me = {'pipeline': fitted_pipeline, 'target_name': target_name}
        QUATTO_LIVES[task.id] = save_me
        task.pipeline = pipeline_json
        task.pipeline_run = pipeline_run_yaml

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

def search_task(logger, session, search):

    try:
        logger.info('Starting distil search ID {}'.format(search.id))
        search.started_at = datetime.datetime.utcnow()

        # load the problem supplied by the search request into a d3m Problem type if one is provided
        if search.problem:
            problem_proto = json_format.Parse(search.problem, problem_pb2.ProblemDescription())
            problem_d3m = api_utils.decode_problem_description(problem_proto)

            target_name = problem_d3m['inputs'][0]['targets'][0]['column_name']
        else:
            problem_d3m = None
            target_name = None

        search_template = None
        pipe, dataset = ex_pipeline.create(search.dataset_uri, problem_d3m, search_template)
        pipeline_json = pipe.to_json(nest_subpipelines=True)
        pipeline = models.Pipelines(id=str(uuid.uuid4()),
                                   search_id=search.id,
                                   pipelines=pipeline_json,
                                   ended=True,
                                   error=False)
        session.add(pipeline)

    except Exception as e:
        logger.warn('Exception running search ID {}: {}'.format(search.id, e), exc_info=True)
        search.error = True
        search.error_message = str(e)
        # TODO error pipeline entry out.
    finally:
        # Update DB with task results
        # and mark task 'ended' and when
        search.ended = True
        search.stopped_at = datetime.datetime.utcnow()
        session.commit()

def job_loop(logger, session):
    task = False
    search = False

    # check for searches first and create pipelines
    try:
        search = session.query(models.Searches) \
                      .order_by(models.Searches.created_at.asc()) \
                      .filter(models.Searches.ended == False) \
                      .first()
    except Exception as e:
        logger.warn('Exception getting task: {}'.format(e), exc_info=True)

    if search:
        search_task(logger, session, search)

    # look for tasks to run
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
        elif task.type == "FIT":
            fit_task(logger, session, task)


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

    # Get DB access
    session = models.start_session(config.DB_LOCATION)

    # Create and start the gRPC server
    server = Server()
    server.start(config.PORT)

    # Main job loop
    while True:
        job_loop(logger, session)
        # Check for a new job every second
        time.sleep(1)


if __name__ == '__main__':
    main()
