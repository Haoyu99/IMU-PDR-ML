from abc import ABC, abstractmethod


class DataSequence(ABC):
    """
    An abstract interface for compiled sequence.
    一个抽象类 用于处理数据
    """

    def __init__(self, **kwargs):
        super(DataSequence, self).__init__()

    @abstractmethod
    def load(self, path):
        pass

    @abstractmethod
    def get_feature(self):
        pass

    @abstractmethod
    def get_target(self):
        pass

    # @abstractmethod
    # def get_aux(self):
    #     pass
