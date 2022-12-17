from __future__ import annotations

import re
from abc import ABC, abstractmethod, ABCMeta
from typing import List

import numpy as np


class LogicAdapter(ABC):

    @abstractmethod
    def process(self, input_text: str) -> Response:
        pass

    @abstractmethod
    def can_process(self, input_text) -> bool:
        return True


class RegexAdapter(LogicAdapter, metaclass=ABCMeta):
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
        return f"{self.confidence}:{self.response_text}"


class SimpleBot:

    def __init__(self) -> None:
        super().__init__()
        self.logic_adapters: list[LogicAdapter] = []

    def add_logic_adapter(self, logic_adapter: LogicAdapter):
        self.logic_adapters.append(logic_adapter)

    def add_logic_adapters(self, logic_adapters: list[LogicAdapter]):
        self.logic_adapters.extend(logic_adapters)

    def ask(self, input_text: str) -> str:
        available_responses = []
        for adapter in self.logic_adapters:
            if adapter.can_process(input_text):
                available_responses.append(adapter.process(input_text))

        print(f"----\n\tQ:{input_text}\n\tA:{available_responses}")
        if len(available_responses) == 0:
            return ""

        match = np.array([res.confidence for res in available_responses]).argsort()[-1]
        return available_responses[match].response_text
