# PICO DLC POC

This is a POC / test to run PICO's face tracking and eye tracking models via PyTorch and is a wrapper around the Qualcomm DeepLearning Container.  
The result of the inference was checked against `snpe-net-run`.  

This is **not** a full DLC runner, it was only made to *just* get the PICO models to run.  
And this code is... not that optimized.

How to run:
1. Extract the model you want to run from `/system/etc/pxr/avatar`
2.
```bash
pip install -r requirements.txt
python PdRunner.py /path/to/model.pd URL_MPJEG
```

Other files included:
- DlcRunner.py: `/path/to/model.dlc URL_MJPEG`
- DlcToOnnx.py: `/path/to/model.dlc /output_folder/name`
- DlcToPt.py: `/path/to/model.dlc /output_folder/name`
- DlcDump.py: `/path/to/model.dlc /output_folder/name`

- PdRunner.py: `/path/to/model.pd URL_MJPEG`
- PdToDlc.py: `/path/to/model.pd /output_folder/name`
- PdToOnnx.py: `/path/to/model.pd /output_folder/name`
- PdToPt.py: `/path/to/model.pd /output_folder/name`
- PdDump.py: `path/to/model.pd /output_folder/name`

Note:
- Currently only hard-coded to one camera feed.
- Currently only hard-coded to load weights from quantized values (which the PICO models are).
- No training, yet, I have yet to try to. Not sure if I can with de-quantized weights.