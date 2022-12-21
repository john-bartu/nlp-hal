from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod, ABCMeta
from typing import List

import nltk.tokenize
import numpy as np
from fuzzysearch import find_near_matches

module_logger = logging.getLogger(__name__)


class LogicAdapter(ABC):

    @abstractmethod
    def can_process(self, input_text, session: dict) -> bool:
        return True

    @abstractmethod
    def process(self, input_text: str, session: dict) -> Response:
        pass


class Stream(ABC):

    @abstractmethod
    def handle(self, output: Response):
        pass


class RegexLogicAdapter(LogicAdapter, metaclass=ABCMeta):
    @property
    @abstractmethod
    def pattern(self) -> re.Pattern:
        pass

    @property
    @abstractmethod
    def keywords(self) -> List[str]:
        pass

    def can_process(self, statement: str, session: dict):
        return self.pattern.findall(statement)

    @abstractmethod
    def process(self, input_text: str, session: dict) -> Response:
        pass

    def calculate_confidence(self, match: str, input_statement: str) -> float:
        match_index = len(match) / len(input_statement)
        keyword_index = self._contains_keyword(input_statement)
        return match_index * 0.4 + keyword_index * 0.6

    def _contains_keyword(self, input_statement: str):
        return any(
            keyword in [
                word.lower()
                for word in
                input_statement.split()
            ]
            for keyword
            in self.keywords
        )


class Response:
    def __init__(self, response_text: str, confidence: float):
        self.response_text: str = response_text
        self.confidence: float = confidence

    def __repr__(self):
        return f"{self.confidence:.2f}:{self.response_text[:32]}..."


class CoreBot:

    def __init__(self) -> None:
        super().__init__()
        self.logic_adapters: List[LogicAdapter] = []
        self.output_adapters: List[Stream] = []
        self.pre_processors: List[PreProcessorAdapter] = [EntityExtractorAdapter()]

    def add_logic_adapter(self, logic_adapter: LogicAdapter):
        self.logic_adapters.append(logic_adapter)

    def add_logic_adapters(self, logic_adapters: List[LogicAdapter]):
        self.logic_adapters.extend(logic_adapters)

    def add_output_adapters(self, stream_adapters: List[Stream]):
        self.output_adapters.extend(stream_adapters)

    def ask(self, input_text: str):
        module_logger.info('\t\tBEGIN OF UTTERANCE')
        module_logger.info(f"Asked: {input_text}")
        available_responses = []

        session = {}

        for processor in self.pre_processors:
            processor.process(input_text, session)

        module_logger.info("Session: " + str(session))

        for adapter in self.logic_adapters:
            if adapter.can_process(input_text, session):
                response = adapter.process(input_text, session)
                module_logger.debug(f"New Response: {response}")
                available_responses.append(response)

        if len(available_responses) == 0:
            return ""

        match = np.array([res.confidence for res in available_responses]).argsort()[-1]
        module_logger.info(f"Best match: {available_responses[match].response_text}")

        for adapter in self.output_adapters:
            adapter.handle(available_responses[match])
        module_logger.info('\t\tEND OF UTTERANCE\n')


class PreProcessorAdapter(ABC):

    @property
    @abstractmethod
    def keywords(self) -> List[str]:
        pass

    @abstractmethod
    def process(self, input_text: str, session: dict):
        pass


class EntityExtractorAdapter(PreProcessorAdapter):
    keywords = {
        'colour': ['red', 'white', 'orange', 'blue'],
        'animal': ['cat', 'dog', 'tiger', 'elephant'],
        'city': ['cracow', 'warsaw', 'oslo', 'new york']
    }

    def process(self, input_text: str, session: dict):
        for token in nltk.tokenize.casual_tokenize(input_text):
            for key, entities in self.keywords.items():
                for entity in entities:
                    if len(find_near_matches(entity.lower(), token.lower(), max_l_dist=round(len(entity) / 4))) > 0:
                        session[key] = entity
