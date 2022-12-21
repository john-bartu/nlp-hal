from __future__ import annotations

import sys
import logging

from audio_porcessing.text_to_speech import CoquiTTSStreamAdapter
from core.adapters import CorpusLogicAdapter, LowConfidenceAdapter, BinaryConvertRegexLogicAdapter
from core.logic import CoreBot, Stream, Response


logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)], format='%(levelname)5s - %(message)s')

test_dialog = [
    [
        "Where are you from?",
        "I was made in the basements of Cracow University of Technology, by wise students: John B and Artem B in the 70's "
    ],
    ["The quick brown fox", "jumps over the lazy dog!"],
    ["What is the weather?", "There is about 70°C on CPU and 50°C GPU. Thermally conductive paste: wet!"],
    ["How are you", "System is up, running with no bugs."]
]


class ConsoleStreamAdapter(Stream):

    def handle(self, output: Response):
        logging.info(f'print({output.response_text})')


class ExampleApiStreamAdapter(Stream):

    def handle(self, output: Response):
        body = {
            'text': output.response_text,
            'confidence': output.confidence
        }
        logging.info("curl -X POST -H 'Content-Type: application/json' -d " + str(body) + " https://example.com")


if __name__ == '__main__':
    bot = CoreBot()
    bot.add_logic_adapters([
        CorpusLogicAdapter(test_dialog),
        LowConfidenceAdapter(0.2, ["Sorry i dont understand.", "Could you repeat please?"]),
        BinaryConvertRegexLogicAdapter()
    ])

    bot.add_output_adapters([
        ConsoleStreamAdapter(),
        ExampleApiStreamAdapter(),
        CoquiTTSStreamAdapter(),
        # TtsStreamAdapter(),
    ])

    bot.ask("Where are you from?")
    bot.ask("weather")
    bot.ask("quick brown")
    bot.ask("how are you")
    bot.ask("bin 1010101010")
    bot.ask("testa pattern")
