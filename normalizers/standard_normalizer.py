from sklearn.preprocessing import StandardScaler

from normalizers.normalizer import Normalizer


class StandardNormalizer(Normalizer):
    def __init__(self):
        self.scaler = StandardScaler()

    def partial_fit(self, array):
        self.scaler.partial_fit(self._reshape_for_scaler(array))

    def transform(self, array):
        return self.scaler.transform(self._reshape_for_scaler(array)).reshape(
            array.shape
        )

    @staticmethod
    def _reshape_for_scaler(array):
        new_shape = (-1, *array.shape[2:]) if array.ndim > 2 else (-1, 1)
        return array.reshape(new_shape)
