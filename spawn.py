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
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--snapshot', type=str, help='model name')
parser.add_argument('--video_name', default='', type=str,
                    help='videos or image files')
args = parser.parse_args()


def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
        video_name.endswith('mp4'):
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame


def main():
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    model.load_state_dict(torch.load(args.snapshot,
        map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)

    #  Socket to talk to server
    context = zmq.Context()
    sub_socket = context.socket(zmq.SUB)

    server_ip = "192.168.1.124"

    # set up frame listening socket
    sub_socket.connect("tcp://{}:5556".format(server_ip))
    sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")

    # setup push socket
    context = zmq.Context()
    push_socket = context.socket(zmq.PUSH)
    push_socket.connect("tcp://{}:5557".format(server_ip))

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
            push_socket.send_json({"dtype": str(frame.dtype), "shape": frame.shape, "time": md['time']}, flags=zmq.SNDMORE)
            push_socket.send(frame)
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
