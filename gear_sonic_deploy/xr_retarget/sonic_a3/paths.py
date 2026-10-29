"""Local assets for the XR to A3 retargeting publisher."""
from pathlib import Path
PACKAGE_DIR = Path(__file__).resolve().parent
DIST_DIR = PACKAGE_DIR.parent
REPO_ROOT = DIST_DIR.parents[1]
RETARGET_CONFIG = PACKAGE_DIR / "config/a3u_pico_retarget.yaml"
RETARGET_MJCF = DIST_DIR / "assets/mjcf/a3u_t3d0.xml"
STAND_CSV = REPO_ROOT / "gear_sonic_deploy/assets/a3_runtime/teleop_motions/BMD_0319_a3_filtered_20260319_164305__stand_Skeleton0.csv"
def resolve(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else DIST_DIR / path
