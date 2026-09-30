import sys

import torch

from dlc import DlcLoader

def main(model, out: str):
    torch.save(model, out + ".pt")

if __name__ == "__main__":
    main(DlcLoader.DlcLoader(path=sys.argv[1]).to_nn_module())