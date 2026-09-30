# Because CV2 is not properly keeping the connection alive!
# This makes the MJPEG server disconnect and close all camera's.

import threading
import time
import cv2
import numpy as np
import requests

class ThreadedMJPEGCamera:
    def __init__(self, url):
        self.url = url
        self.frame = None
        self.ret = False
        self.running = True
        self.lock = threading.Lock()
        
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        session = requests.Session()
        session.headers.update({'User-Agent': 'Mozilla/5.0'})

        while self.running:
            try:
                with session.get(self.url, stream=True, timeout=(5, 10)) as response:
                    bytes_data = b''

                    for chunk in response.raw.stream(1024, decode_content=False):
                        if not self.running:
                            break

                        bytes_data += chunk
                        
                        soi = bytes_data.find(b'\xff\xd8')
                        eoi = bytes_data.find(b'\xff\xd9')

                        if soi != -1 and eoi != -1:
                            if eoi > soi:
                                jpg = bytes_data[soi:eoi + 2]
                                bytes_data = bytes_data[eoi + 2:]

                                img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                                if img is not None:
                                    with self.lock:
                                        self.frame = img
                                        self.ret = True
                            else:
                                bytes_data = bytes_data[soi:]

            except Exception:
                with self.lock:
                    self.ret = False
                time.sleep(1)

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)