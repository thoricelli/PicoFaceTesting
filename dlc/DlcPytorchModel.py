import torch.nn as nn

from dlc import DlcNode, DlcModel

class DlcPyTorchModel(nn.Module):
    def __init__(self, model: DlcModel.DlcModel):
        super().__init__()

        self.model = model

        self.layers = nn.ModuleList()

        nodes = self.model.get_nodes()

        for name in nodes:
            node = nodes[name]
            py_node = node.get_pytorch_node()
            if (py_node is not None): self.layers.append(py_node)

    def forward(self, x):
        nodes: list[DlcNode.DlcNode] = []

        all_nodes = self.model.get_nodes()

        # Find input nodes
        for name in x:
            nodes.append(all_nodes.get(name))      

        while len(nodes) > 0:
            future_nodes = []

            for output in nodes:
                if (output.forward(x)):
                    future_nodes = future_nodes + output.get_output_nodes()

            nodes = future_nodes

        return x