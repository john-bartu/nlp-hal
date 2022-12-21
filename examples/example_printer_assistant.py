from __future__ import annotations

import logging
import re

from core.adapters import CorpusLogicAdapter, LowConfidenceAdapter
from core.logic import Response, CoreBot, RegexLogicAdapter

logging.basicConfig(level=logging.DEBUG)

test_dialog = [
    [
        "Black lines down the page",
        "Wipe the scanner glass strip with a dry lint free soft cloth.\n"
        "Clean the primary corona wire inside the drum unit by sliding the green tab.\n"
        "The drum unit may be damaged. Put in a new drum unit."
    ],
    [
        "White lines across the page",
        "Clean the drum unit\n"
        "The issue may disappear by itself. Print multiple blank pages to clear this issue, especially if the machine has not been used for a long time."
    ],

    [
        "Curled or wavy paper after print",
        "Check the paper type and quality. High temperatures and high humidity will cause paper to curl\n"
        "Choose Reduce Paper Curl mode in the printer driver when you do not use our recommended print media. "
    ],
    [
        "Who are you",
        "I am Brother printer help assistant"
    ]
]


class ErrorCodeLogicAdapter(RegexLogicAdapter):
    pattern = re.compile(r'#([\da-z]{2,})', flags=re.IGNORECASE)
    keywords = ['error', 'code', '#']

    code_message = {
        '10': 'There is a problem on the duplex unit.',
        '38': 'The machine does not work due to a paper jam.',
        '43': 'The internal temperature is too low or too high.',
        '48': 'There is a problem on the print head.',
        '4F': 'The machine does not work due to a paper jam or some sensor problem.',
        '8F': 'There is a problem on the duplex unit.'
    }

    def __init__(self):
        super().__init__()

    def process(self, input_text: str, session: dict) -> Response:
        error_code = self.pattern.findall(input_text)[0].upper()
        if error_code in self.code_message:
            return Response(self.code_message[error_code], 1)
        else:
            return Response("Unknown error code", 0)


if __name__ == '__main__':
    bot = CoreBot()
    bot.add_logic_adapters(
        [
            CorpusLogicAdapter(test_dialog),
            LowConfidenceAdapter(0.2, ["Sorry i dont understand.", "Could you repeat please?"]),
            ErrorCodeLogicAdapter()
        ]
    )
    bot.add_output_adapters(
        [
            # CoquiTTSStreamAdapter(),
            # ConsoleStreamAdapter()
        ]
    )
    bot.ask("i have black lines on page")
    bot.ask("i have white lines down the page")
    bot.ask("Print's curled paper")
    bot.ask("I have error code #8f")
    bot.ask("I have #48 error code what can I do")
    bot.ask("error #10")
    bot.ask("who you are?")
