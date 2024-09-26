from .flores import FLORESDataModule
from .bmlama import BMLAMADataModule
from .xnli import XNLIDataModule
from .stereoset import StereoSetDataModule
from .civil_comments import CivilCommentsDataModule
from .combined_datamodule import CombinedDataModule

FLORES_LANGUAGES = FLORESDataModule.SUPPORTED_LANGUAGES
BMLAMA_LANGUAGES_17 = BMLAMADataModule.SUPPORTED_LANGUAGES_17
BMLAMA_LANGUAGES_53 = BMLAMADataModule.SUPPORTED_LANGUAGES_53
XNLI_LANGUAGES = XNLIDataModule.SUPPORTED_LANGUAGES