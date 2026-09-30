import sys
from pd import Pd

def pd_to_dlc(input_path, output_path):
    pd = None

    with open(input_path, "rb") as f:
        pd = Pd.Pd(f)
        pd.to_dlc_file(output_path)  

if __name__ == "__main__":
    pd_to_dlc(sys.argv[1], sys.argv[2])