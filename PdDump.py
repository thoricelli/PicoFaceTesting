import sys

from dlc import DlcLoader
import DlcDump
from pd import Pd

if __name__ == "__main__":
    with open(sys.argv[1], "rb") as f:
        pd = Pd.Pd(f)

        loader = DlcLoader.DlcLoader(dlc=pd.to_dlc_buffer())
        DlcDump.main(loader, sys.argv[2])