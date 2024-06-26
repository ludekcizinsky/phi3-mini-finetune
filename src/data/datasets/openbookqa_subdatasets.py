from .openbookqa_base import OpenBookQAName, OpenBookQABaseData


class OpenBookQAMainData(OpenBookQABaseData):
    NAME = OpenBookQAName.Main


class OpenBookQAAdditionalData(OpenBookQABaseData):
    NAME = OpenBookQAName.Additional
