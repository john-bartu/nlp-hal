from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod, ABCMeta
from typing import List

import numpy as np

module_logger = logging.getLogger(__name__)


class LogicAdapter(ABC):

    @abstractmethod
    def can_process(self, input_text) -> bool:
        return True

    @abstractmethod
    def process(self, input_text: str) -> Response:
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

    def can_process(self, statement: str):
        return self.pattern.findall(statement)

    @abstractmethod
    def process(self, input_text: str) -> Response:
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
        self.logic_adapters: list[LogicAdapter] = []
        self.output_adapters: list[Stream] = []

    def add_logic_adapter(self, logic_adapter: LogicAdapter):
        self.logic_adapters.append(logic_adapter)

    def add_logic_adapters(self, logic_adapters: list[LogicAdapter]):
        self.logic_adapters.extend(logic_adapters)

    def add_output_adapters(self, stream_adapters: list[Stream]):
        self.output_adapters.extend(stream_adapters)

    def ask(self, input_text: str):
        module_logger.info('\t\tBEGIN OF UTTERANCE')
        module_logger.debug(f"Asked: {input_text}")
        available_responses = []
        for adapter in self.logic_adapters:
            if adapter.can_process(input_text):
                response = adapter.process(input_text)
                module_logger.debug(f"New Response: {response}")
                available_responses.append(response)

        if len(available_responses) == 0:
            return ""

        match = np.array([res.confidence for res in available_responses]).argsort()[-1]
        module_logger.info(f"Best match: {available_responses[match].response_text}")

        for adapter in self.output_adapters:
            adapter.handle(available_responses[match])
        module_logger.info('\t\tEND OF UTTERANCE\n')
