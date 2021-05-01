from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse
import socket

import cv2
import sys
import torch
import numpy as np
import zmq
import json
import io
import base64
import uuid
import threading
from glob import glob
from PIL import Image
from zmq.utils.monitor import recv_monitor_message

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
        os.system(
            "gdown https://drive.google.com/uc?id=1-tEtYQdT1G9kn8HsqKNDHVqjE16F8YQH")
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

    # create an unique identifier
    worker_id = uuid.uuid4()

    # build tracker
    tracker = build_tracker(model)

    # Socket to talk to server
    context = zmq.Context()
    sub_socket = context.socket(zmq.SUB)

    # set up frame listening socket
    sub_socket.connect("tcp://{}:5556".format(args.server_ip))
    sub_socket.setsockopt_string(zmq.SUBSCRIBE, "frame_")
    sub_socket.setsockopt_string(zmq.SUBSCRIBE, str(worker_id))

    # setup push socket
    context = zmq.Context()
    push_socket = context.socket(zmq.PUSH)
    push_socket.connect("tcp://{}:5557".format(args.server_ip))

    # event monitoring
    # used to register worker once connection is established
    EVENT_MAP = {}
    for name in dir(zmq):
        if name.startswith('EVENT_'):
            value = getattr(zmq, name)
            EVENT_MAP[value] = name

    # monitor thread function
    def event_monitor(monitor):
        while monitor.poll():
            evt = recv_monitor_message(monitor)
            evt.update({'description': EVENT_MAP[evt['event']]})
            if evt['event'] == zmq.EVENT_HANDSHAKE_SUCCEEDED:
                push_socket.send_json(
                    {"type": "REGISTER", "id": str(worker_id)})
            if evt['event'] == zmq.EVENT_MONITOR_STOPPED:
                break
        monitor.close()

    # register monitor
    monitor = sub_socket.get_monitor_socket()

    t = threading.Thread(target=event_monitor, args=(monitor,))
    t.start()

    support = None

    try:
        while True:
            # wait for next message
            _ = sub_socket.recv()
            md = sub_socket.recv_json()
            if md['type'] == 'FRAME':
                msg = sub_socket.recv()
                buf = memoryview(msg)
                frame = np.frombuffer(
                    buf, dtype=md['dtype']).reshape(md['shape'])

                if support is None:
                    continue

                outputs = tracker.track(frame)
                bbox = list(map(int, outputs['bbox']))

                # send result
                push_socket.send_json(
                    {
                        "type": "TRACK",
                        "bbox": bbox,
                        "score": outputs['best_score'].tolist(),
                        "time": md['time'],
                        "id": str(worker_id)
                    })
                print('message: {}'.format(md['time']), end='\r')
            elif md['type'] == 'SUPPORT':
                frame_raw = md['data']['img']  # base 64 png image
                frame = np.array(
                    Image.open(
                        io.BytesIO(
                            base64.b64decode(frame_raw)
                        )
                    ).convert('RGB'))[:, :, ::-1]
                bbox = [int(float(i)) for i in md['data']['bbox'].split(",")]
                tracker.init(frame, bbox)
                support = (frame, bbox)
                print('Support received, tracking will now start')
            elif md['type'] == 'LOCATION':
                # make sure tracker has been initalized
                if support is not None:
                    center_pos = np.array(md['data'])
                    tracker.update(center_pos)
            elif md['type'] == 'PING':
                push_socket.send_json({"type": "PONG", "id": str(worker_id)})
            else:
                print('Invalid message type received: {}'.format(md['type']))
    except KeyboardInterrupt:
        print('Exiting... notifying server of disconnect')
        push_socket.send_json(
            {"type": "FIN", "id": str(worker_id)})
        # wait for the server to respond or let the user forcefully close
        print("Waiting for server response. Press CTRL+C again to forcefully close")
        while True:
            _ = sub_socket.recv()
            md = sub_socket.recv_json()
            if md['type'] == "FIN":
                print('Server responded, now exiting')
                exit(0)
            elif md['type'] == "FRAME":
                # we have to accept the incoming frame to properly accept future messages
                msg = sub_socket.recv()


if __name__ == '__main__':
    main()
