import json

from flatbuffer import TensorData

class DlcTensorData:
    def __init__(self, tensor_data: TensorData.TensorData):
        self.tensor_data = tensor_data

    def to_dict(self):
            bytes_arr = self.tensor_data.BytesAsNumpy()
            floats_arr = self.tensor_data.FloatsAsNumpy()
    
            return {
                "d_type": self.tensor_data.DType(),
                "bytes": bytes_arr.tolist() if hasattr(bytes_arr, "tolist") else [],
                "floats": floats_arr.tolist() if hasattr(floats_arr, "tolist") else []
            }
    
    def __str__(self):
        return json.dumps(self.to_dict())