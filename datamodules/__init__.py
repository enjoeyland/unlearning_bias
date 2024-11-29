from .base import BaseDataModule, CombinedDataModule
from .stereoset import StereoSetDataModule
from .civil_comments import CivilCommentsDataModule
from .crows_pairs import CrowsPairsDataModule
from .adult import AdultDataModule
from .compas import CompasDataModule

# 새롭게 datamodule 만들때 할일
# 1. datamodule 수정
# 2. __init__.py에서 import 추가
# 3. model.py에서 DataModule 추가
# 4. callbacks.py에서 monitor 추가