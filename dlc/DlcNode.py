import json

import torch as torch
import torch.nn as nn

from flatbuffer import Node
from dlc import DlcAttribute, DlcNodeParameter

class DlcNode:
    """A wrapper class around the raw flatbuffer node."""

    def __init__(self, node: Node.Node, nodes: dict[bytes, DlcNode]=None, node_parameter: DlcNodeParameter.DlcNodeParameter=None, linked_outputs: dict[bytes, list[DlcNode.DlcNode]]=None):
        self.node = node
        self.node_parameter = node_parameter

        self.nodes = nodes
        self.attributes: dict[bytes, DlcAttribute.DlcAttribute] = {}

        self.input_nodes = None
        self.output_nodes = None

        self.linked_outputs = linked_outputs

        self._input_names = set()

        for i in range(self.node.InputsLength()):
            self._input_names.add(self.node.Inputs(i))

        self._output_names = set()

        for i in range(self.node.OutputsLength()):
            self._output_names.add(self.node.Outputs(i))

        for i in range(self.node.AttributesLength()):
            attribute = self.node.Attributes(i)
            self.attributes[attribute.Name()] = DlcAttribute.DlcAttribute(attribute)

        if (nodes is not None):
            self.pytorch_node = self._build_pytorch_node()

    def forward(self, x: dict[bytes, list[torch.Tensor]]) -> bool:
        inputs: list[torch.Tensor] = []

        if (not self._matches_all_inputs(x)): return False

        # Consume all outputs.
        for name in self.get_input_names():
                tensors = x.get(name)
                inputs.append(tensors.pop())

                if (len(tensors) <= 0):
                    x.pop(name)
                
        if (self.pytorch_node is not None):
            self._apply_value_to_outputs(self.pytorch_node(inputs[0]), x)
            return True

        # Special cases
        node_type = self.flatbuffer().Type()

        if (node_type == b"Input"):
            self._apply_value_to_outputs(inputs[0], x) # Uh.
            return True

        if node_type == b"ElementwiseBinaryOp":
            output = inputs[0]
            for t in inputs[1:]:
                output = output + t
            self._apply_value_to_outputs(output, x)
            return True

    def flatbuffer(self) -> Node.Node:
        return self.node

    def get(self, Name: bytes):
        return self.attributes.get(Name)

    def get_output_nodes(self) -> list[DlcNode]:
        if (self.output_nodes is not None): return self.output_nodes

        """Returns a list of all the nodes the output of this node is connected to."""
        outputNames = self.get_output_names()
        output_nodes = []

        for outputName in outputNames:
            nodes = self.linked_outputs.get(outputName)
            if (nodes is None):
                continue
            output_nodes = output_nodes + nodes

        self.output_nodes = output_nodes
        return output_nodes

    def get_input_nodes(self) -> list[DlcNode]:
        if (self.input_nodes is not None): return self.input_nodes

        result: list[DlcNode] = []
        
        for target_name in self.nodes:
            target_node = self.nodes[target_name]

            for i in range(self.node.InputsLength()):
                source_input_node_name = self.node.Inputs(i)

                if (target_name == self.node.Name()):
                    continue

                for j in range(target_node.flatbuffer().OutputsLength()):
                    targetOutput = target_node.flatbuffer().Outputs(j)

                    if (targetOutput == source_input_node_name):
                        result.append(target_node)
                        break

        self.input_nodes = result
        return result

    def get_nhwc(self):
        shape = self.get(b"OutputDims").get(b"0").flatbuffer()
        return shape.Int32ListAsNumpy().tolist()

    def get_pytorch_node(self):
        return self.pytorch_node

    def get_output_names(self):
        return self._output_names

    def get_input_names(self):
            return self._input_names

    def _matches_all_inputs(self, x: dict[str, torch.Tensor]) -> bool:
        return self._input_names.issubset(x)

    def _build_pytorch_node(self):
        """Converts this node to a Pytorch node, and loads the weights into it."""

        if (self.node.Type() == b"Convolutional"):
            convolutional = nn.Conv2d(
                self.get_input_nodes()[0].get_nhwc()[3],
                self.get_nhwc()[3],
                kernel_size=(self.get(b"kernel_size_y").i32(), self.get(b"kernel_size").i32()),
                stride=(self.get(b"stride_y").i32(), self.get(b"stride").i32()),
                padding=(self.get(b"padding_y").i32(), self.get(b"padding").i32()),
                dilation=(self.get(b"dilation_y").i32(), self.get(b"dilation_x").i32()),
                groups=self.get(b"groups").i32()
            )

            weights = self.node_parameter.get_tensor_type(b"weights").dequantized()
            convolutional.weight.data.copy_(torch.from_numpy(weights))

            bias = self.node_parameter.get_tensor_type(b"bias")

            if (bias is not None):
                convolutional.bias.data.copy_(torch.from_numpy(bias.dequantized()))
            return convolutional

        if (self.node.Type() == b"Neuron"):
            return nn.ReLU()

        if (self.node.Type() == b"Pooling"):
            return nn.AdaptiveAvgPool2d(1)

        if self.node.Type() == b"FullyConnected":
            in_features = self.get_input_nodes()[0].get_nhwc()[-1]
            out_features = self.get_nhwc()[-1]
            conv = nn.Conv2d(in_features, out_features, kernel_size=1)
        
            weights = self.node_parameter.get_tensor_type(b"weights").dequantized()
            conv.weight.data.copy_(torch.from_numpy(weights).reshape(out_features, in_features, 1, 1))
        
            bias = self.node_parameter.get_tensor_type(b"bias")
            if bias is not None:
                conv.bias.data.copy_(torch.from_numpy(bias.dequantized()))
            return conv

        return
    
    def _apply_value_to_outputs(self, value: torch.Tensor, x: dict[bytes, list[torch.Tensor]]):
        output_names = self.get_output_names()
        output_nodes = self.get_output_nodes()

        for output_name in output_names:
            tensor_list = x.get(output_name)

            if (tensor_list is None):
                tensor_list = []
                x[output_name] = tensor_list

            if (len(output_nodes) <= 0):
                tensor_list.append(value)
                continue
                
            for i in range(len(output_nodes)):
                tensor_list.append(value)

        return x

    def to_dict(self):
        return {
            "index": self.node.Index(),
            "name": self.node.Name().decode("utf-8"),
            "type": self.node.Type().decode("utf-8"),
            "inputs": [self.node.Inputs(i).decode("utf-8") for i in range(self.node.InputsLength())],
            "outputs": [self.node.Outputs(i).decode("utf-8") for i in range(self.node.OutputsLength())],
            "attributes": [DlcAttribute.DlcAttribute(self.node.Attributes(i)).to_dict() for i in range(self.node.AttributesLength())]
        }

    def __str__(self):
        return json.dumps(self.to_dict())