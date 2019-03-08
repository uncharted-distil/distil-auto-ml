import json
import time
import pathlib
import logging
import datetime

import config

import utils
import models

from server.server import Server

from exline.main import exline_all

import dill
import pandas as pd


# Configure output dir
pathlib.Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def save_job(engine, task_id):
    filepath = utils.make_job_fn(task_id)
    with open(filepath, 'wb') as f:
        dill.dump(engine, f)

def exline_task(logger, session, task):
    try:
        logger.info('Starting exline task ID {}'.format(task.id))
        task.started_at = datetime.datetime.utcnow()
        logger.info("DATASET_URI: {}".format(task.dataset_uri))
        #logger.info("PROBLEM: {}".format(task.problem))
        prob = task.problem
        prob = json.loads(prob)
        for target in prob['inputs'][0]['targets']:
            target['resource_id'] = target.pop('resourceId')
            target['column_index'] = target.pop('columnIndex')
            target['column_name'] = target.pop('columnName')
        prob['id'] = '__unset__'
        prob['digest'] = '__unset__'
        #logger.info(prob)
        exline_all(logger, task.dataset_uri, prob)
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
    

def main(once=False):
    # Set up logging
    logging_level = logging.DEBUG if config.DEBUG else logging.INFO
    system_version = utils.get_worker_version()
    logger = utils.setup_logging(logging_level,
                                 log_file=config.LOG_FILENAME,
                                 system_version=system_version)
    logger.info("System version {}".format(system_version))

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
