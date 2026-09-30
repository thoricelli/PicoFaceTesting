import json

from flatbuffer import Model
from dlc import DlcAttribute, DlcNode, DlcModelParameters

class DlcModel:
    def __init__(self, model: Model.Model, model_parameters: DlcModelParameters.DlcModelParameters):
        self.model = model

        # Parse model into wrapper classes
        self.wrapper_nodes: dict[bytes, DlcNode.DlcNode] = {}
        self.linked_outputs: dict[bytes, list[DlcNode.DlcNode]] = {}

        for i in range(model.NodesLength()):
            node = model.Nodes(i)
            
            self.wrapper_nodes[node.Name()] = DlcNode.DlcNode(node, self.wrapper_nodes, model_parameters.get_node(node.Name()), self.linked_outputs)

        self.attributes: dict[str, DlcAttribute.DlcAttribute] = {}

        for i in range(self.model.AttributesLength()):
            attribute = self.model.Attributes(i)
            self.attributes[attribute.Name()] = DlcAttribute.DlcAttribute(attribute)

        self.inputs = {}

        # For all nodes in the model
        for name in self.wrapper_nodes:
            node = self.wrapper_nodes[name]

            if (node.flatbuffer().Type() == b"Input"):
                input_dims_list = self.get(b"BufferInfos").get(node.flatbuffer().Name()).get(b"dims").flatbuffer()
                self.inputs[node.flatbuffer().Name()] = input_dims_list.Int32ListAsNumpy().tolist()

            for output_name in node.get_output_names():

                # Find me nodes that have the same input
                for name in self.wrapper_nodes:
                    search_node = self.wrapper_nodes[name]

                    # Look through this node's inputs.
                    for input_name in search_node.get_input_names():

                        # Not itself
                        if (node.flatbuffer().Name() == search_node.flatbuffer().Name()):
                            continue

                        # We found a node!
                        if (input_name == output_name):
                            outputs_list = self.linked_outputs.get(input_name)

                            if (outputs_list is None):
                                outputs_list = []
                                self.linked_outputs[input_name] = outputs_list

                            outputs_list.append(search_node)

        output_attributes = self.get(b"BufferInfos").attributes

        self.outputs: dict[str, list[int]] = {}

        for name in output_attributes:
            if (self.linked_outputs.get(name) is None):
                attribute = output_attributes[name]
                self.outputs[name] = attribute.get(b"dims").attribute.Int32ListAsNumpy().tolist()



    def get_inputs(self): # TODO: Make this NCHW by default?
        """Returns a dictionary of {b"node_name": [bytes] <-- input dimensions in NHWC}"""
        return self.inputs

    def get_outputs(self):
        """Returns a dictionary of {b"node_name": [bytes] <-- output dimensions in NHWC}"""
        return self.outputs

    def get(self, Name: bytes):
        return self.attributes.get(Name)

    def get_nodes_for_output(self, name):
        return self.linked_outputs[name]

    def get_nodes(self) -> dict[bytes, DlcNode.DlcNode]:
        return self.wrapper_nodes

    def flatbuffer(self):
        return self.model

    def to_dict(self):
        return {
            "nodes": [
                DlcNode.DlcNode(self.model.Nodes(i)).to_dict()
                for i in range(self.model.NodesLength())
            ],
            "attributes": [
                DlcAttribute.DlcAttribute(self.model.Attributes(i)).to_dict()
                for i in range(self.model.AttributesLength())
            ],
        }

    def __str__(self):
        return json.dumps(self.to_dict())