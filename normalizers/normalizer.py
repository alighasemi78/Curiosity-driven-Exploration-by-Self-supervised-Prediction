from abc import ABCMeta, abstractmethod


class Normalizer(metaclass=ABCMeta):
    @abstractmethod
    def partial_fit(self, array):
        raise NotImplementedError("Implement me")

    @abstractmethod
    def transform(self, array):
        raise NotImplementedError("Implement me")

    def partial_fit_transform(self, array):
        self.partial_fit(array)
        return self.transform(array)
