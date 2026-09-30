import sys
import torch

from pd import Pd
from dlc import DlcLoader

def main(pd: Pd.Pd, out: str):
    model = DlcLoader.DlcLoader(dlc=pd.to_dlc_buffer()).to_nn_module()
    torch.save(model, out + ".pt")

if __name__ == "__main__":
    with open(sys.argv[1], "rb") as f:
        main(Pd.Pd(f), sys.argv[2])