from typing import Iterable

from webdnn.graph.variable import Variable

WEBDNN_LICENSE = "(C) Machine Intelligence Laboratory (The University of Tokyo), MIT License"

# FIXME: DOCS
class Graph:
    def __init__(self,
                 inputs: Iterable[Variable],
                 outputs: Iterable[Variable]):
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        self.licenses = {"webdnn": WEBDNN_LICENSE}

    def __repr__(self):
        return f"""<{self.__class__.__name__} inputs={self.inputs}, outputs={self.outputs}>"""

    def __str__(self):
        return self.__repr__()
