import sys

from pd import Pd
import DlcToOnnx
from dlc import DlcLoader

if __name__ == "__main__":
    with open(sys.argv[1], "rb") as f:
        DlcToOnnx.main(DlcLoader.DlcLoader(dlc=Pd.Pd(f).to_dlc_buffer()), sys.argv[2])