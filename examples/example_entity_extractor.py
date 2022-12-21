from __future__ import annotations

import logging

import nltk.tokenize

from core.adapters import LowConfidenceAdapter
from core.logic import CoreBot, Stream, Response, LogicAdapter

logging.basicConfig(level=logging.INFO)


class ConsoleStreamAdapter(Stream):

    def handle(self, output: Response):
        logging.info(f'print({output.response_text})')


class LovingAnimalAdapter(LogicAdapter):

    def can_process(self, input_text, session: dict) -> bool:
        if 'animal' in session:
            return True

    def __init__(self):
        super().__init__()

    def process(self, input_text: str, session: dict) -> Response:
        text = f'You love: {session["animal"]}'
        return Response(text, 1)


class CityWeatherCheck(LogicAdapter):

    def can_process(self, input_text, session: dict) -> bool:
        if 'city' in session and 'weather' in nltk.tokenize.casual_tokenize(input_text):
            return True

    def __init__(self):
        super().__init__()

    def process(self, input_text: str, session: dict) -> Response:
        text = f'Checking weather in {session["city"]} - sunny'
        return Response(text, 1)


if __name__ == '__main__':
    bot = CoreBot()
    bot.add_logic_adapters([
        LovingAnimalAdapter(),
        CityWeatherCheck(),
        LowConfidenceAdapter(0.2, ["Sorry i dont understand.", "Could you repeat please?"]),
    ])

    bot.add_output_adapters([
    ])

    bot.ask("I really love cat")
    bot.ask("My favourite color is blue")
    bot.ask("What is the weather in Cracow?")
