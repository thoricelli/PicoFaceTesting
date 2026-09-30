import json

from flatbuffer import ModelParameters
from dlc import DlcNodeParameter

class DlcModelParameters:
    def __init__(self, model_parameters: ModelParameters.ModelParameters):
        self.model_parameters = model_parameters

        self.wrapped_nodes = {}
        for i in range(self.model_parameters.NodesLength()):
            node = self.model_parameters.Nodes(i)
            self.wrapped_nodes[node.Name()] = DlcNodeParameter.DlcNodeParameter(node)

    def flatbuffer(self):
        return self.model_parameters

    def get_nodes(self):
        return self.wrapped_nodes

    def get_node(self, name: bytes):
        return self.wrapped_nodes.get(name)
    
    def to_dict(self):
        return {
            "nodes": [DlcNodeParameter.DlcNodeParameter(self.model_parameters.Nodes(i)).to_dict() for i in range(self.model_parameters.NodesLength())],
        }

    def __str__(self):
        return json.dumps(self.to_dict())