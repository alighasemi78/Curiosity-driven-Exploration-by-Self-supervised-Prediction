from collections import Counter

from abc import ABCMeta, abstractmethod


class Reporter(metaclass=ABCMeta):
    def __init__(self, report_interval=1):
        self.counter = Counter()
        self.graph_initialized = False
        self.report_interval = report_interval
        self.t = 0

    def will_report(self, tag):
        return self.counter[tag] % (self.report_interval + 1) == 0

    def scalar(self, tag, value):
        if self.will_report(tag):
            self._scalar(tag, value, self.counter[tag])
        self.counter[tag] += 1

    def graph(self, model, input_to_model):
        if not self.graph_initialized:
            self._graph(model, input_to_model)
            self.graph_initialized = True

    @abstractmethod
    def _scalar(self, tag, value, step):
        raise NotImplementedError("Implement me")

    def _graph(self, model, input_to_model):
        raise NotImplementedError("Implement me")
