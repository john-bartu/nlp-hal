import hashlib
import logging
import os.path
import re

from TTS.api import TTS
from pydub import AudioSegment
from pydub.playback import play

from core.logic import Stream, Response

# from gtts import gTTS

module_logger = logging.getLogger(__name__)


# class TtsStreamAdapter(Stream):
#
#     def __init__(self):
#         self.TEMP_PATH: str = "./tts_temp"
#         if not os.path.isdir(self.TEMP_PATH):
#             os.mkdir(self.TEMP_PATH)
#         self.regex_filter = re.compile(r'[^\w.?!]+')
#
#     def text_cleanup(self, text: str):
#         return self.regex_filter.sub(' ', text)
#
#     def handle(self, output: Response):
#         text = self.text_cleanup(output.response_text)
#         module_logger.info("CLEANED: " + text)
#
#         message_bytes = text.encode('ascii')
#         text_hash = hashlib.sha256(message_bytes)
#         filename = f"{self.TEMP_PATH}/{text_hash.hexdigest()}.mp3"
#         if not os.path.isfile(filename):
#             module_logger.info("GENERATING: " + filename)
#             tts = gTTS(text=text, lang='en')
#             tts.save(filename)
#         else:
#             module_logger.info("CACHE MATCH: " + filename)
#         # playsound.playsound(filename)


class CoquiTTSStreamAdapter(Stream):

    def __init__(self):
        self.TEMP_PATH: str = "tts_temp_cq"
        if not os.path.isdir(self.TEMP_PATH):
            os.mkdir(self.TEMP_PATH)
        self.regex_filter = re.compile(r'[^\w.,?!]+')
        self.tts = TTS('tts_models/en/ljspeech/tacotron2-DCA')

    def text_cleanup(self, text: str):
        return self.regex_filter.sub(' ', text)

    def handle(self, output: Response):
        text = self.text_cleanup(output.response_text)
        module_logger.info("CLEANED: " + text)

        message_bytes = text.encode('ascii')
        text_hash = hashlib.sha256(message_bytes)
        filename = f"{self.TEMP_PATH}/{text_hash.hexdigest()}.wav"
        if not os.path.isfile(filename):
            module_logger.info("GENERATING: " + filename)
            self.tts.tts_to_file(text=text,
                                 file_path=filename)
        else:
            module_logger.info("CACHE MATCH: " + filename)

        sound = AudioSegment.silent().append(AudioSegment.from_file(file=filename, format="wav"))
        play(sound)
