from enum import Enum


class MultiValueEnum(Enum):
    def __new__(cls, value, *values):
        self = object.__new__(cls)
        self._value_ = value
        for v in values:
            self._add_value_alias_(v)
        return self