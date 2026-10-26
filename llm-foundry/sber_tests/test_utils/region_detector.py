import os
from typing import Optional, Tuple, Literal

RegionType = Literal["slurm", "cloud"]


class RegionDetector:

    DEFAULT_REGION = "SR008"
    WM_TRAIN_WORKSPACE = "wm-train"

    @staticmethod
    def get_region_for_tests() -> str:
        cluster_key = os.getenv("CLUSTER_KEY")
        workspace = os.getenv("MLS_NTB_WORKSPACE_NAME")

        if workspace == RegionDetector.WM_TRAIN_WORKSPACE and cluster_key:
            return cluster_key

        return RegionDetector.DEFAULT_REGION

    @staticmethod
    def get_current_region() -> Tuple[Optional[str], Optional[RegionType]]:

        cloud_region = os.getenv("CLUSTER_KEY")
        if cloud_region:
            return cloud_region, "cloud"

        return None, None

    @staticmethod
    def is_wm_train_workspace() -> bool:
        return os.getenv("MLS_NTB_WORKSPACE_NAME") == RegionDetector.WM_TRAIN_WORKSPACE


def get_test_submission_region() -> str:
    return RegionDetector.get_region_for_tests()
