import io
import zipfile

from dlc import DlcModel, DlcModelParameters
from dlc.DlcPytorchModel import DlcPyTorchModel
from flatbuffer import Model, ModelParameters

class DlcLoader:
    def __init__(self, path=None, dlc=None):
        if (dlc is not None):
            self.dlc = dlc
        else:
            with open(path, "rb") as f:
                self.dlc = f.read()

        with zipfile.ZipFile(io.BytesIO(self.dlc), "r") as dlc_zip:
            model_params_buff = dlc_zip.read("model.params")

            params = ModelParameters.ModelParameters.GetRootAs(model_params_buff, 8)
            self.params = DlcModelParameters.DlcModelParameters(params)

            model_buff = dlc_zip.read("model")

            model = Model.Model.GetRootAs(model_buff, 8)
            self.model = DlcModel.DlcModel(model, self.params)

    def get_params(self):
        return self.params

    def get_model(self):
        return self.model

    def to_nn_module(self):
            return DlcPyTorchModel(self.model)