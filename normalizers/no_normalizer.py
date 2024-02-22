from normalizers.normalizer import Normalizer


class NoNormalizer(Normalizer):
    def partial_fit(self, array):
        pass

    def transform(self, array):
        return array
