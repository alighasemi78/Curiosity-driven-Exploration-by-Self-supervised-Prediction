from torch import nn as nn

from reporters.reporter import Reporter


class NoReporter(Reporter):
    def will_report(self, tag):
        return False

    def scalar(self, tag, value):
        pass

    def graph(self, model, input_to_model):
        pass

    def _scalar(self, tag, value, step):
        pass

    def _graph(self, model, input_to_model):
        pass
