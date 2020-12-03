import json
from processing.pipeline import load_data
import time
import pathlib
import logging
import datetime
import io
import jsonschema
from api.utils import ValueType

import config

import utils
import models
import uuid
from server.server import Server

from processing import pipeline as ex_pipeline
from processing.scoring import Scorer
from server import export

from d3m.container import dataset
from d3m.metadata import pipeline, problem

from server import messages

# Configure output dir
pathlib.Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def decode_dataset_uri(dataset_uri):
    return dataset_uri.split(",")

def produce_task(logger, session, server, task):
    try:
        logger.info("Starting produce task ID {}".format(task.id))

        # pull out the results the caller requested, ignore any others that were exposed
        output_keys = json.loads(task.output_keys)

        # call produce on a fitted pipeline
        fitted_runtime = server.get_fitted_runtime(task.fit_solution_id)
        dataset_uris = decode_dataset_uri(task.dataset_uri)
        test_datasets = []
        for dataset_uri in dataset_uris:
            test_dataset = server.get_loaded_dataset(dataset_uri)
            if test_dataset is None:
                test_dataset = load_data(dataset_uri)
            test_datasets.append(train_dataset)
        results = ex_pipeline.produce(
            fitted_runtime, test_datasets, outputs_to_expose=output_keys
        )

        # loop over the (ordered) list of requested output types until we find one that we support
        output_types = json.loads(task.output_types)
        selected_output_type = None
        for output_type in output_types:
            if output_type in messages.ALLOWED_TYPES:
                selected_output_type = output_type
                break
        if not selected_output_type:
            logger.warn(f"no output type specified - defaulting to {ValueType.CSV_URI}")
            selected_output_type = ValueType.CSV_URI

        for output_key in output_keys:
            if output_key in results.values:
                if selected_output_type == ValueType.PARQUET_URI:
                    preds_path = utils.make_preds_filename(
                        task.request_id,
                        output_key=output_key,
                        output_type=selected_output_type,
                    )
                    results.values[output_key].to_parquet(preds_path, index=False)
                elif selected_output_type == ValueType.CSV_URI:
                    preds_path = utils.make_preds_filename(
                        task.request_id,
                        output_key=output_key,
                        output_type=selected_output_type,
                    )
                    results.values[output_key].to_csv(preds_path, index=False)
        session.commit()
    except Exception as e:
        logger.warn(
            "Exception running task ID {}: {}".format(task.id, e), exc_info=True
        )
        task.error = True
        task.error_message = str(e)
    finally:
        # Update DB with task results
        # and mark task 'ended' and when
        task.ended = True
        task.ended_at = datetime.datetime.utcnow()
        session.commit()


def score_task(logger, session, server, task):
    try:
        logger.info("Starting score task ID {}".format(task.id))
        task.started_at = datetime.datetime.utcnow()
        score_config = (
            session.query(models.ScoreConfig)
            .filter(models.ScoreConfig.id == task.score_config_id)
            .first()
        )

        # reconstruct the problem object from the saved json if present and extract the target index
        problem_obj = (
            problem.Problem.from_json_structure(json.loads(task.problem))
            if task.problem
            else None
        )
        target_idx = -1
        if problem_obj != None:
            inputs = problem_obj["inputs"]
            if len(inputs) > 1:
                logger.warn(
                    f"found {len(inputs)} inputs - using first and ignoring others"
                )

            targets = inputs[0]["targets"]
            if len(targets) > 1:
                logger.warn(
                    f"found {len(targets)} targets - using first and ignoring others"
                )

            target_idx = targets[0]["column_index"]
        else:
            raise TypeError("no problem definition available for scoring")

        # check for successfully completed fit, run if not
        fitted_runtime = server.get_fitted_runtime(task.solution_id)
        if fitted_runtime is None:
            fit_task(logger, session, server, task)
            fitted_runtime = server.get_fitted_runtime(task.solution_id)

        scorer = Scorer(logger, task, score_config, fitted_runtime, target_idx)
        score_values, metric_used = scorer.run()
        for score_value in score_values:
            score = models.Scores(
                solution_id=task.solution_id,
                score_config_id=score_config.id,
                value=score_value,
                metric_used=metric_used,
            )
            session.add(score)
            session.commit()
    except Exception as e:
        logger.warn(
            "Exception running task ID {}: {}".format(task.id, e), exc_info=True
        )
        task.error = True
        task.error_message = str(e)
    finally:
        # Update DB with task results
        # and mark task 'ended' and when
        task.ended = True
        task.ended_at = datetime.datetime.utcnow()
        session.commit()


def fit_task(logger, session, server, task):
    try:
        logger.info("Starting distil task ID {}".format(task.id))
        task.started_at = datetime.datetime.utcnow()

        # reconstruct the problem object from the saved json if present
        problem_obj = (
            problem.Problem.from_json_structure(json.loads(task.problem))
            if task.problem
            else None
        )

        # fetch the pipeline from the DB
        resolver = pipeline.Resolver(load_all_primitives=False)  # lazy load
        pipeline_obj = (
            pipeline.Pipeline.from_json(task.pipeline, resolver=resolver)
            if task.pipeline
            else None
        )

        # pull out the results the caller requested, ignore any others that were exposed
        output_keys = json.loads(task.output_keys) if task.output_keys else {}

        # Check to see if this is a fully specified pipeline.  If so, we'll run it as a non-standard since
        # it doesn't need to be serialized.
        run_as_standard = not task.fully_specified

        dataset_uris = decode_dataset_uri(task.dataset_uri)
        train_datasets = []
        for dataset_uri in dataset_uris:
            train_dataset = server.get_loaded_dataset(dataset_uri)
            if train_dataset is None:
                train_dataset = load_data(dataset_uri)
            train_datasets.append(train_dataset)
        fitted_runtime, result = ex_pipeline.fit(
            pipeline_obj,
            problem_obj,
            train_datasets,
            is_standard_pipeline=run_as_standard,
            outputs_to_expose=output_keys,
        )

        # loop over the (ordered) list of requested output types until we find one that we support
        output_types = json.loads(task.output_types) if task.output_types else {}

        selected_output_type = None
        for output_type in output_types:
            if output_type in messages.ALLOWED_TYPES:
                selected_output_type = output_type
                break
        if not selected_output_type:
            logger.warn(f"no output type specified - defaulting to {ValueType.CSV_URI}")
            selected_output_type = ValueType.CSV_URI

        for output_key in output_keys:
            if output_key in result.values:
                if selected_output_type == ValueType.PARQUET_URI:
                    preds_path = utils.make_preds_filename(
                        task.request_id,
                        output_key=output_key,
                        output_type=selected_output_type,
                    )
                    result.values[output_key].to_parquet(preds_path, index=False)
                elif selected_output_type == ValueType.CSV_URI:
                    preds_path = utils.make_preds_filename(
                        task.request_id,
                        output_key=output_key,
                        output_type=selected_output_type,
                    )
                    result.values[output_key].to_csv(preds_path, index=False)

        # fitted runtime needs to have the fitted pipeline ID we've generated
        fitted_runtime.pipeline.id = task.fit_solution_id

        # since score does not get the fitted solution id, need to allow for solution id lookup
        server.add_fitted_runtime(task.fit_solution_id, fitted_runtime)
        server.add_fitted_runtime(task.solution_id, fitted_runtime)

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
            logger.warn("Could not parse result")

        task.pipeline_run = pipeline_run_yaml

    except Exception as e:
        logger.warn(
            "Exception running task ID {}: {}".format(task.id, e), exc_info=True
        )
        task.error = True
        task.error_message = str(e)
    finally:
        # Update DB with task results
        # and mark task 'ended' and when
        task.ended = True
        task.fitted = True
        task.ended_at = datetime.datetime.utcnow()
        session.commit()


def search_task(logger, session, server, search):

    try:
        logger.info("Starting distil search ID {}".format(search.id))
        search.started_at = datetime.datetime.utcnow()

        # Generate search ID
        search_id = search.id

        resolver = pipeline.Resolver(load_all_primitives=False)  # lazy load
        search_template_obj = None
        if search.search_template is not None:
            search_template_obj = pipeline.Pipeline.from_json(
                search.search_template, resolver=resolver
            )

        # flag to run fully specified pipelines as non-standard for extra flexibiltiy
        fully_specified = ex_pipeline.is_fully_specified(search_template_obj)

        # load the problem supplied by the search request into a d3m Problem type if one is provided
        problem_obj = (
            problem.Problem.from_json_structure(json.loads(search.problem))
            if search.problem
            else None
        )

        # based on our problem type and data type, create a pipeline
        dataset_uris = decode_dataset_uri(search.dataset_uri)
        dataset_uri = ""
        if len(dataset_uris) > 0:
            dataset_uri = dataset_uris[0]
        pipeline_objs, dataset, ranks = ex_pipeline.create(
            dataset_uri,
            problem_obj,
            search.time_limit,
            search.max_models,
            search_template_obj,
            resolver=resolver,
        )
        server.add_loaded_dataset(dataset_uri, dataset)
        for i, pipeline_obj in enumerate(pipeline_objs):
            pipeline_json = pipeline_obj.to_json(nest_subpipelines=True)

            # save the pipeline to the DB
            solution_pipeline = models.Pipelines(
                id=str(uuid.uuid4()),
                search_id=search.id,
                pipelines=pipeline_json,
                fully_specified=fully_specified,
                ended=True,
                error=False,
                rank=ranks[i],
            )
            session.add(solution_pipeline)
        session.commit()

    except Exception as e:
        logger.warn(
            "Exception running search ID {}: {}".format(search.id, e), exc_info=True
        )
        search.error = True
        search.error_message = str(e)
        # TODO error pipeline entry out.
    finally:
        # Update DB with task results
        # and mark task 'ended' and when
        search.ended = True
        search.stopped_at = datetime.datetime.utcnow()
        session.commit()


def job_loop(logger, session, server):
    task = False
    search = False

    # check for searches first and create pipelines
    try:
        search = (
            session.query(models.Searches)
            .order_by(models.Searches.created_at.asc())
            .filter(models.Searches.ended == False)
            .first()
        )
    except Exception as e:
        logger.warn("Exception getting task: {}".format(e), exc_info=True)

    if search:
        search_task(logger, session, server, search)

    # look for tasks to run
    try:
        task = (
            session.query(models.Tasks)
            .order_by(models.Tasks.created_at.asc())
            .filter(models.Tasks.ended == False)
            .first()
        )
    except Exception as e:
        logger.warn("Exception getting task: {}".format(e), exc_info=True)

    # If there is work to be done...
    try:
        if task:
            if task.type == "FIT":
                fit_task(logger, session, server, task)
            elif task.type == "SCORE":
                score_task(logger, session, server, task)
            elif task.type == "PRODUCE":
                produce_task(logger, session, server, task)
    except Exception as e:
        logger.warn("Exception running task: {}".format(e), exc_info=True)


def main(once=False):
    # override config vals D3M values
    export.override_config()

    # Set up logging
    logging_level = logging.DEBUG if config.DEBUG else logging.INFO
    system_version = utils.get_worker_version()
    logger = utils.setup_logging(
        logging_level, log_file=config.LOG_FILENAME, system_version=system_version
    )
    logger.info("System version {}".format(system_version))
    logging.basicConfig(level=logging_level)
    logger.info(f"Logging level to {logging_level}")

    logger.info(f"Baseline time out {config.TIME_LIMIT}")
    logger.info(f"Full hyperparameter tuning enabled {config.HYPERPARAMETER_TUNING}")
    logger.info(f"GPU support {config.GPU}")

    # Get DB access
    session = models.start_session(config.DB_LOCATION)

    # Create and start the gRPC server
    server = Server()
    server.start(config.PORT)

    # Main job loop
    while True:
        job_loop(logger, session, server)
        # Check for a new job every second
        time.sleep(1)


if __name__ == "__main__":
    main()
