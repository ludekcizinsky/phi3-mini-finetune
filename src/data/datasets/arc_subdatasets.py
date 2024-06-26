from .arc_base import ArcBaseData, ArcDatasetName


class ArcEasyData(ArcBaseData):
    NAME = ArcDatasetName.Easy


class ArcChallengeData(ArcBaseData):
    NAME = ArcDatasetName.Challenge
