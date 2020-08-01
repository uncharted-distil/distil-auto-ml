import datetime

from sqlalchemy import (Column, DateTime,
                        String, Integer,
                        ForeignKey, Text,
                        Boolean, Float,
                        Binary)
from sqlalchemy.ext.declarative import declarative_base

from sqlalchemy import create_engine, event, engine
from sqlalchemy.orm import sessionmaker

from sqlalchemy.pool import StaticPool


Base = declarative_base()

@event.listens_for(engine.Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()

def get_session(db_location):
    # TODO: not disable check_same_thread
    engine = create_engine('sqlite:///' + db_location,
                           connect_args={'check_same_thread': False},
                           poolclass=StaticPool)
    session = sessionmaker()
    session.configure(bind=engine)
    return session()

def start_session(db_location):
    # TODO: not disable_check_same_thread
    engine = create_engine('sqlite:///' + db_location,
                           connect_args={'check_same_thread': False},
                           poolclass=StaticPool)
    session = sessionmaker()
    session.configure(bind=engine)
    Base.metadata.create_all(engine)
    return session()

class Searches(Base):
    __tablename__ = 'searches'
    id = Column(String,
                primary_key=True)
    created_at = Column(
        DateTime,
        default=datetime.datetime.utcnow)
    started_at = Column(DateTime)
    stopped_at = Column(DateTime)
    running = Column(Boolean, default=False)
    ended = Column(Boolean, default=False)
    error = Column(String)
    user_agent = Column(String)
    api_version = Column(String)
    problem = Column(String)
    dataset_uri = Column(String)
    search_template = Column(String)
    time_limit = Column(Integer)
    max_models = Column(Integer)

class Requests(Base):
    __tablename__ = 'requests'
    id = Column(String, primary_key=True)
    type = Column(String)
    solution_id = Column(String, ForeignKey('solutions.id'))
    fit_solution_id = Column(String, ForeignKey('fit_solution.id'))
    task_id = Column(String, ForeignKey('tasks.id'))
    created_at = Column(
        DateTime,
        default=datetime.datetime.utcnow)
    internal_score = Column(Float)

class FitSolution(Base):
    __tablename__ = 'fit_solution'
    id = Column(String, primary_key=True)
    solution_id = Column(String, ForeignKey('solutions.id'))
    task_id = Column(String, ForeignKey('tasks.id'))
    created_at = Column(
        DateTime,
        default=datetime.datetime.utcnow)
    internal_score = Column(Float)

class Solutions(Base):
    __tablename__ = 'solutions'
    id = Column(String, primary_key=True)
    search_id = Column(String, ForeignKey('searches.id'))
    pipeline_id = Column(String, ForeignKey('pipelines.id'))
    created_at = Column(
        DateTime,
        default=datetime.datetime.utcnow)
    internal_score = Column(Float)

class Tasks(Base):
    __tablename__ = 'tasks'
    id = Column(String, primary_key=True)
    search_id = Column(String, ForeignKey('searches.id'))
    solution_id = Column(String, ForeignKey('solutions.id'))
    fit_solution_id = Column(String, ForeignKey('fit_solution.id'))
    request_id = Column(String, ForeignKey('requests.id'))
    # VALIDATE or SCORE
    type = Column(String)
    # Start -- SCORE specific --
    score_config_id = Column(String, ForeignKey('score_config.id'))
    # End -- SCORE specific
    worker_id = Column(String)
    dataset_uri = Column(String)
    # Start -- PRODUCE specific --
    output_keys = Column(String)
    # End -- PRODUCE specific --
    pipeline = Column(String)
    pipeline_run = Column(String)
    problem = Column(String)
    created_at = Column(
        DateTime,
        default=datetime.datetime.utcnow)
    started_at = Column(DateTime)
    ended = Column(Boolean,
                   default=False)
    fitted = Column(Boolean,
                   default=False)
    ended_at = Column(
        DateTime)
    error = Column(
        Boolean,
        default=False)
    error_message = Column(String)
    # Start -- FIT specific --
    fully_specified = Column(Boolean, default=False)
    # End -- FIT specific

class Pipelines(Base):
    __tablename__ = 'pipelines'
    id = Column(String, primary_key=True)
    search_id = Column(String, ForeignKey('searches.id'))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    pipelines = Column(String)  # [{}...{}]
    error = Column(Boolean, default=False)
    ended = Column(Boolean, default=False)
    fully_specified = Column(Boolean, default=False)
    rank = Column(Integer, default=1)

class ScoreConfig(Base):
    __tablename__ = 'score_config'
    id = Column(String, primary_key=True)
    # e.g. 'F1_MICRO'
    metric = Column(String)
    # e.g. 'HOLDOUT'
    method = Column(String)
    num_folds = Column(Integer,
                       default=3)
    train_test_ratio = Column(Integer)
    train_size = Column(Float,
                        default=0.75)
    shuffle = Column(Boolean,
                     default=True)
    random_seed = Column(Integer,
                         default=505)
    stratified = Column(Boolean,
                        default=False)


class Scores(Base):
    __tablename__ = 'scores'
    id = Column(Integer, primary_key=True)
    solution_id = Column(String, ForeignKey('solutions.id'))
    fit_solution_id = Column(String, ForeignKey('fit_solution.id'))
    score_config_id = Column(Integer, ForeignKey('score_config.id'))
    value = Column(Float)


class Messages(Base):
    __tablename__ = 'messages'
    id = Column(Integer, primary_key=True)
    task_id = Column(String, ForeignKey('tasks.id'))
    message = Column(Text)
    progress = Column(Text)
