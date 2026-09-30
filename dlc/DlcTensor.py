import json

import numpy as np

from flatbuffer import Tensor
from dlc import DlcAttribute, DlcTensorData

class DlcTensor:
    def __init__(self, Tensor: Tensor.Tensor):
        self.tensor = Tensor

        self.type = Tensor.Name()
        
        self.attributes: dict[bytes, DlcAttribute.DlcAttribute] = {}
        
        for i in range(self.tensor.AttributesLength()):
            attribute = self.tensor.Attributes(i)
            self.attributes[attribute.Name()] = DlcAttribute.DlcAttribute(attribute)

    def flatbuffer(self) -> Tensor.Tensor:
        return self.tensor

    def get_type(self):
        """b'bias' or b'weights'"""
        return self.type.rsplit(b"_", 1)[0]

    def get_kernel_height(self):
        assert(self.get_type(), b"weights")

        return self.get_shape()[0]

    def get_kernel_width(self):
        assert(self.get_type(), b"weights")

        return self.get_shape()[1]

    def get_input_channels(self):
        assert(self.get_type(), b"weights")

        return self.get_shape()[2]

    def get_output_channels(self):
        assert(self.get_type(), b"weights")

        return self.get_shape()[3]

    def get_shape(self):
        return self.tensor.ShapeAsNumpy().tolist()

    def get(self, Name: bytes):
        return self.attributes.get(Name)

    def dequantized(self):
        shape = self.get_shape()
        raw = bytes(self.tensor.Data().BytesAsNumpy().tolist())

        number_of_elements = np.prod(shape)
        bytes_per_element = len(raw) / number_of_elements
        
        dtype = {1: np.uint8, 2: np.uint16, 4: np.int32}[bytes_per_element]
        reshaped = np.frombuffer(raw, dtype=dtype).reshape(shape)

        encoding = self.get(b"DataEncoding").get(b"0").get(b"DataEncodingElement").get(b"0")
        delta, offset = encoding.get(b"delta").float32(), encoding.get(b"offset").float32()

        dequantized = (reshaped + offset) * np.float32(delta)

        if (len(shape) == 4):
            return dequantized.transpose(3, 2, 0, 1) # OIHW

        return dequantized
    
    def to_dict(self):
        return {
            "name": self.tensor.Name().decode("utf-8"),
            "shape": [self.tensor.Shape(i) for i in range(self.tensor.ShapeLength())],
            "data": DlcTensorData.DlcTensorData(self.tensor.Data()).to_dict(),
            "attributes": [DlcAttribute.DlcAttribute(self.tensor.Attributes(i)).to_dict() for i in range(self.tensor.AttributesLength())],
        }

    def __str__(self):
        return json.dumps(self.to_dict())

        
