from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import cv2
import torch
import numpy as np
import zmq
import io
import base64
from glob import glob
from PIL import Image

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--server_ip', type=str)
parser.add_argument('--gpu_id', type=int, default=0)
args = parser.parse_args()

def main():

    torch.cuda.set_device(args.gpu_id)

    model_dir = "./experiments/siamrpn_r50_l234_dwxcorr/model.pth"
    model_config = "./experiments/siamrpn_r50_l234_dwxcorr/config.yaml"

    if os.path.isfile(model_dir):
        print("model file {} found".format(model_dir))
    else:
        print("model files not found, starting download".format(model_dir))
        os.system("gdown https://drive.google.com/uc?id=1-tEtYQdT1G9kn8HsqKNDHVqjE16F8YQH")
        os.system("mv model.pth ./experiments/siamrpn_r50_l234_dwxcorr")

    # load config
    cfg.merge_from_file(model_config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    model.load_state_dict(torch.load(model_dir,
        map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)

    #  Socket to talk to server
    context = zmq.Context()
    sub_socket = context.socket(zmq.SUB)

    # set up frame listening socket
    sub_socket.connect("tcp://{}:5556".format(args.server_ip))
    sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")

    # setup push socket
    context = zmq.Context()
    push_socket = context.socket(zmq.PUSH)
    push_socket.connect("tcp://{}:5557".format(args.server_ip))

    recvd_support = False

    while True:
        # wait for next message
        md = sub_socket.recv_json()
        if md['type'] == 'FRAME':
            msg = sub_socket.recv()
            buf = memoryview(msg)
            frame = np.frombuffer(buf, dtype=md['dtype']).reshape(md['shape'])

            if not recvd_support:
                continue

            outputs = tracker.track(frame)
            bbox = list(map(int, outputs['bbox']))
            cv2.rectangle(frame, (bbox[0], bbox[1]),
                          (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                          (0, 255, 0), 3)

            # send result
            push_socket.send_json({"bbox": bbox, "time": md['time']})
        elif md['type'] == 'SUPPORT':
            frame_raw = md['data']['img']  # base 64 png image
            frame = np.array(
                        Image.open(
                            io.BytesIO(
                                base64.b64decode(frame_raw)
                            )
                        ).convert('RGB'))[:, :, ::-1]
            Image.open(
                io.BytesIO(
                    base64.b64decode(frame_raw)
                )).save("/home/travis/output.png")
            bbox = [int(i) for i in md['data']['bbox'].split(",")]
            tracker.init(frame, bbox)
            recvd_support = True
            print('supports received, tracking will now start')
        elif md['type'] == 'LOCATION':
            center_pos = np.array(md['data'])
            tracker.update(center_pos)
        else:
            print('Invalid message type received: {}'.format(md['type']))


if __name__ == '__main__':
    main()
