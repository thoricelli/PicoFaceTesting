import sys

import DlcRunner
from pd import Pd
from dlc import DlcLoader

def main(path, ip_camera_url: str):
    with open(path, "rb") as f:
            pd = Pd.Pd(f)

            dlc = pd.to_dlc_buffer()
            DlcRunner.main(DlcLoader.DlcLoader(dlc=dlc), ip_camera_url)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])