import json

from flatbuffer import Attribute

class DlcAttribute:
    def __init__(self, attribute: Attribute.Attribute):
        self.attribute = attribute

        self.attributes: dict[bytes, DlcAttribute] = {}

        for i in range(self.attribute.AttributesLength()):
            attribute = self.attribute.Attributes(i)
            self.attributes[attribute.Name()] = DlcAttribute(attribute)

    def i32(self):
        return self.flatbuffer().Int32Value()

    def float32(self):
        return self.flatbuffer().Float32Value()

    def flatbuffer(self) -> Attribute.Attribute:
        return self.attribute

    def get(self, Name: bytes):
        return self.attributes.get(Name)

    def to_dict(self):
        return {
            "name": self.attribute.Name().decode("utf-8"),
            "type": self.attribute.Type(),
            "bool_value": self.attribute.BoolValue(),
            "int32_value": self.attribute.Int32Value(),
            "uint32_value": self.attribute.Uint32Value(),
            "float32_value": self.attribute.Float32Value(),
            "string_value": self.attribute.StringValue(),
            "byte_list": [self.attribute.ByteList(i) for i in range(self.attribute.ByteListLength())],
            "int32_list": [self.attribute.Int32List(i) for i in range(self.attribute.Int32ListLength())],
            "float32_list": [self.attribute.Float32List(i) for i in range(self.attribute.Float32ListLength())],
            "attributes": [DlcAttribute(self.attribute.Attributes(i)).to_dict() for i in range(self.attribute.AttributesLength())]
        }

    def __str__(self):
        return json.dumps(self.to_dict())