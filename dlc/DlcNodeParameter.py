import json

from flatbuffer import NodeParameter
from dlc import DlcTensor

class DlcNodeParameter:
    def __init__(self, node_parameter: NodeParameter.NodeParameter):
        self.node_parameter = node_parameter

        self.wrapped_tensors: dict[bytes, DlcTensor.DlcTensor] = {}
        
        for i in range(self.node_parameter.WeightsLength()):
            tensor = DlcTensor.DlcTensor(self.node_parameter.Weights(i))
            self.wrapped_tensors[tensor.get_type()] = tensor

    def flatbuffer(self):
        return self.node_parameter

    def get_tensors(self):
        return self.wrapped_tensors

    def get_tensor_type(self, Type: bytes):
        return self.wrapped_tensors.get(Type)

    def to_dict(self):
        return {
            "name": self.node_parameter.Name().decode("utf-8"),
            "weights": [DlcTensor.DlcTensor(self.node_parameter.Weights(i)).to_dict() for i in range(self.node_parameter.WeightsLength())],
        }

    def __str__(self):
        return json.dumps(self.to_dict())