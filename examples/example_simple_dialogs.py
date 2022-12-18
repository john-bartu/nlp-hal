from __future__ import annotations

import coloredlogs

from core.adapters import CorpusLogicAdapter, LowConfidenceAdapter, BinaryConvertRegexLogicAdapter
from core.logic import CoreBot

coloredlogs.install(level="DEBUG")

test_dialog = [
    [
        "Where are you from?",
        "I was made in the basements of Cracow University of Technology, by wise students: John B and Artem B in the 70's "
    ],
    ["The quick brown fox", "jumps over the lazy dog!"],
    ["What is the weather?", "There is about 70°C on CPU and 50°C GPU. Thermally conductive paste: wet!"],
    ["How are you", "System is up, running with no bugs."]
]

if __name__ == '__main__':
    bot = CoreBot()
    bot.add_logic_adapters(
        [
            CorpusLogicAdapter(test_dialog),
            LowConfidenceAdapter(0.2, ["Sorry i dont understand.", "Could you repeat please?"]),
            BinaryConvertRegexLogicAdapter()
        ]
    )
    bot.ask("Where are you from?")
    bot.ask("weather")
    bot.ask("quick brown")
    bot.ask("how are you")
    bot.ask("bin 1010101010")
    bot.ask("testa pattern")
