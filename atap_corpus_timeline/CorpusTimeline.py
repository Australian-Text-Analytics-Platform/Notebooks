import logging
from logging.handlers import RotatingFileHandler
from os.path import abspath, join, dirname
from typing import Optional

import panel as pn
from atap_context_extractor import ContextExtractor
from atap_corpus._types import TCorpora
from atap_corpus_loader import CorpusLoader
from atap_corpus_slicer import CorpusSlicer

from atap_corpus_timeline.CorpusVisualiser import CorpusVisualiser

pn.extension("plotly", notifications=True)


class CorpusTimeline(pn.viewable.Viewer):
    LOGGER_NAME: str = "corpus-timeline"

    @staticmethod
    def setup_logger(logger_name: str, run_logger: bool):
        logger = logging.getLogger(logger_name)
        logger.propagate = False
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        if not run_logger:
            logger.addHandler(logging.NullHandler())
            return

        formatter = logging.Formatter(
            '%(asctime)s %(levelname)6s - %(name)s:%(lineno)4d %(funcName)20s() - %(message)s')
        log_file_location = abspath(join(dirname(__file__), '..', 'log.txt'))
        # Max size is ~10MB with 1 backup, so a max size of ~20MB for log files
        max_bytes: int = 10000000
        backup_count: int = 1
        file_handler = RotatingFileHandler(log_file_location, maxBytes=max_bytes, backupCount=backup_count)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        logger.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

        logger.info('Logger started')

    @staticmethod
    def log(msg: str, level: int):
        logger = logging.getLogger(CorpusTimeline.LOGGER_NAME)
        logger.log(level, msg)

    def __init__(self, corpus_loader: Optional[CorpusLoader] = None, run_logger: bool = False, **params):
        super().__init__(**params)

        CorpusTimeline.setup_logger(CorpusTimeline.LOGGER_NAME, run_logger)

        if corpus_loader:
            self.corpus_loader: CorpusLoader = corpus_loader
        else:
            self.corpus_loader: CorpusLoader = CorpusLoader(root_directory='.', run_logger=run_logger)
        self.corpora: TCorpora = self.corpus_loader.get_mutable_corpora()

        self.corpus_slicer: CorpusSlicer = CorpusSlicer(corpus_loader, run_logger)
        self.context_extractor: ContextExtractor = ContextExtractor(corpus_loader, run_logger)
        self.timeline_panel: CorpusVisualiser = CorpusVisualiser(corpus_loader, self.LOGGER_NAME)
        self.corpus_loader.add_tab("Corpus Timeline", self.timeline_panel.servable())

    def __panel__(self):
        return self.corpus_loader.servable()

    def get_corpus_loader(self) -> CorpusLoader:
        return self.corpus_loader

    def get_mutable_corpora(self) -> TCorpora:
        return self.corpora
