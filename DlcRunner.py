import sys

import torch

import matplotlib.pyplot as plt
import matplotlib.axes as maxes
import numpy as np

import cv2

from dlc import DlcLoader
import ThreadedMJPEGCamera

def main(dlc_loader: DlcLoader.DlcLoader, ip_camera_url: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = dlc_loader.get_model()
    input_dims = model.get_inputs()
    output_dims = model.get_outputs()

    nn_model = dlc_loader.to_nn_module()
    nn_model.to(device)
    nn_model.eval()

    cam = ThreadedMJPEGCamera.ThreadedMJPEGCamera(ip_camera_url)

    plt.ion()
    fig, axes = plt.subplots(1, len(output_dims), figsize=(10, 4))

    output_graphs = {}

    if isinstance(axes, maxes.Axes):
        axes = [axes]

    for ax, output_name in zip(axes, output_dims):
        dims = output_dims[output_name]

        bars_amount = dims[-1]
        barh = ax.barh(range(bars_amount), np.zeros(bars_amount))
        ax.set_yticks(range(bars_amount))
        ax.set_ylabel("type")
        ax.set_title(output_name)

        ax.autoscale(enable=True, axis='x', tight=False)

        output_graphs[output_name] = {
            "bar": barh, 
            "ax": ax,
            "min_val": 0.0,
            "max_val": 0.0
        }

    while True:
        ret, frame = cam.read()

        if not ret:
            continue

        cv2.imshow("Preview", frame)

        input = next(iter(input_dims.keys()))

        n,h,w,c = input_dims[input]  

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (h, w), interpolation=cv2.INTER_LINEAR)

        tensor_input = torch.from_numpy(resized).to(device, non_blocking=True).float()
        tensor_input = (tensor_input - 128.0) / 128.0
        tensor_input = tensor_input.unsqueeze(0).unsqueeze(0).repeat(n, c, 1, 1)

        with torch.no_grad():
            out = nn_model({input: [tensor_input]})

        for output_name in output_dims:
            output_values = out[output_name][0].cpu().numpy().flatten()
            output_graph = output_graphs[output_name]

            current_max = np.max(output_values)
            if current_max > output_graph["max_val"]:
                output_graph["max_val"] = current_max

            current_min = np.min(output_values)
            if current_min < output_graph["min_val"]:
                output_graph["min_val"] = current_min

            ax = output_graph["ax"]
            ax.set_xlim(output_graph["min_val"], output_graph["max_val"])

            bars = output_graph["bar"] 

            for bar, v in zip(bars, output_values):
                bar.set_width(v)

        plt.pause(0.001)

        if not plt.fignum_exists(fig.number):
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    plt.close("all")

if __name__ == "__main__":
    main(DlcLoader.DlcLoader(path=sys.argv[1]), sys.argv[2])