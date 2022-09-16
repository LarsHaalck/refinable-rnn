from enum import Enum


class ModelType(Enum):
    ResnetReg = 0,
    ResnetClass = 1,
    HourGlassSqueeze = 2,
    HourGlass = 3


class ModelMode(Enum):
    Single = 0,
    Recurrent = 1
