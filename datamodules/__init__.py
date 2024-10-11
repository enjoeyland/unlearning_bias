from .base import BaseDataModule
from .flores import FLORESDataModule
from .bmlama import BMLAMADataModule
from .xnli import XNLIDataModule
from .stereoset import StereoSetDataModule
from .civil_comments import CivilCommentsDataModule
from .crows_pairs import CrowsPairsDataModule
from .combined_datamodule import CombinedDataModule
from .adult import AdultDataModule

FLORES_LANGUAGES = FLORESDataModule.SUPPORTED_LANGUAGES
BMLAMA_LANGUAGES_17 = BMLAMADataModule.SUPPORTED_LANGUAGES_17
BMLAMA_LANGUAGES_53 = BMLAMADataModule.SUPPORTED_LANGUAGES_53
XNLI_LANGUAGES = XNLIDataModule.SUPPORTED_LANGUAGES