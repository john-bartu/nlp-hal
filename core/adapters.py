import random
import re
from abc import ABCMeta
from operator import itemgetter

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from core.logic import LogicAdapter, Response, RegexLogicAdapter

nltk.download('stopwords', quiet=True)
stop_words = nltk.corpus.stopwords.words('english')


class LowConfidenceAdapter(LogicAdapter, metaclass=ABCMeta):

    def __init__(self, confidence: float, response: str | list[str]) -> None:
        super().__init__()
        self.confidence = confidence

        if isinstance(response, list):
            self.responses = response
        else:
            self.responses = [response]

    def can_process(self, input_text) -> bool:
        return True

    def process(self, input_text: str) -> Response:
        return Response(random.choice(self.responses), self.confidence)


class CorpusLogicAdapter(LogicAdapter):

    def __init__(self, question_answer: list[list[str]]) -> None:
        super().__init__()
        self.question_answer = question_answer

    def can_process(self, input_text) -> bool:
        return True

    def process(self, input_text: str) -> Response:
        corpus = list(map(itemgetter(0), self.question_answer))
        answer = list(map(itemgetter(1), self.question_answer))
        corpus.append(input_text)
        # tfidf_vec = TfidfVectorizer(stop_words=stop_words)
        # warning: 'where are you from' will not work, all words are in stopwords
        tfidf_vec = TfidfVectorizer()
        tfidf = tfidf_vec.fit_transform(corpus)
        similarity = cosine_similarity(tfidf[-1], tfidf)
        # print(similarity)
        idx = similarity.argsort()[0][-2]
        return Response(answer[idx], similarity[0][idx])


class BinaryConvertRegexLogicAdapter(RegexLogicAdapter):
    pattern = re.compile(r'([01]{2,})', flags=re.IGNORECASE)
    keywords = ['binary', 'bin']

    def __init__(self):
        super().__init__()

    def process(self, input_text: str) -> Response:
        binary = self.pattern.findall(input_text)[0]
        text = f'Decimal value of {binary} is {int(binary, 2)}'
        conf = self.calculate_confidence(binary, input_text)
        return Response(text, conf)
