import sys
import torch
from torch import nn

from pd import Pd
from dlc import DlcLoader, DlcModel

class ONNXExportWrapper(nn.Module):
    def __init__(self, model: nn.Module, dlc_model: DlcModel.DlcModel):
        super().__init__()
        self.model = model
        self.dlc_model = dlc_model

    def forward(self, input_tensor):
        input_name = next(iter(self.dlc_model.get_inputs()))

        input_dict = {input_name: [input_tensor]}
        return self.model(input_dict)

def main(loader: DlcLoader, out: str):
    model = loader.get_model()

    nn_model = loader.to_nn_module()
    nn_model.eval()

    inputs = model.get_inputs()
    input_dims_bytes = inputs[list(inputs.keys())[0]]

    outputs = model.get_outputs()

    tensor_input = torch.randn(*input_dims_bytes)
    tensor_input = tensor_input.permute(0, 3, 1, 2) # To NHWC

    wrapper = ONNXExportWrapper(nn_model, model)
    wrapper.eval()

    torch.onnx.export(
        wrapper,
        tensor_input,
        out + ".onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=list(inputs.keys()),
        output_names=list(outputs.keys()),
        dynamo=False,
    )

if __name__ == "__main__":
    main(DlcLoader.DlcLoader(path=sys.argv[1]), sys.argv[2])
        