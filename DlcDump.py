import sys

from dlc import DlcLoader

def main(dlc_loader: DlcLoader.DlcLoader, output_path):
    with open(output_path + ".json", "wt") as f:
        f.write(str(dlc_loader.get_model()))

    with open(output_path + "_params.json", "wt") as f:
        f.write(str(dlc_loader.get_params()))
    

if __name__ == "__main__":
    main(DlcLoader.DlcLoader(path=sys.argv[1]), sys.argv[2])