import os
import sys
import time
import random
import numpy as np
import subprocess
from PIL import Image
from functools import partial
from multiprocessing import Pool, Manager
from abc import ABCMeta, abstractmethod
import pyqrcode
from pyzbar.pyzbar import decode as qr_decode

from util import get_qr_packet_size, get_qr_array_size
import qr

def main():
    if not os.path.isdir(frames_dir):
        os.makedirs(frames_dir)
        print('generating initial frames')
        subprocess.call([
            'ffmpeg',
            '-i', 'rms.webm',
            '-vf', 'scale=800:600',
            os.path.join(frames_dir, 'image-%04d.png')
            ])

    subprocess.call(['rm', 'temp.webm'])
    if os.path.isdir(encoded_dir):
        subprocess.call(['rm', '-r', 'encoded_frames'])
    if os.path.isdir(decoded_dir):
        subprocess.call(['rm', '-r', 'decoded_frames'])

    os.makedirs(encoded_dir, exist_ok=True)
    os.makedirs(decoded_dir, exist_ok=True)

    qr = QR(max_code_size=600, version=30, depth=1, color_space='RGB',
            channels=[])

    qr_sync = QR(max_code_size=200, version=5, tlx=0, tly=600, depth=1,
                 color_space='RGB', channels=[])

    names = sorted(os.listdir(frames_dir))
    names = [x for x in names if x.endswith('.png')][:30]
    packets_0 = [[get_unit_packet() for _ in range(qr.capacity)]
                 for _ in names]

    # encode data
    indirs = [os.path.join(frames_dir, x) for x in names]
    outdirs = [os.path.join(encoded_dir, x) for x in names]
    inputs = list(zip(indirs, outdirs, packets_0))
    worker = partial(_encode_one, qr)
    _multiprocess(worker, inputs, info='encoding frames')

    # encode synchronization info
    packet_ids = [[str(x)] for x in range(len(names))]
    packets_0 = {i[0]: x for i, x in zip(packet_ids, packets_0)}
    dirs = [os.path.join(encoded_dir, x) for x in names]
    inputs = list(zip(dirs, dirs, packet_ids))
    worker = partial(_encode_one, qr_sync)
    _multiprocess(worker, inputs, info='encoding sync')

    # video encoding and decoding
    print('ffmpeg encoding')
    subprocess.call([
       'ffmpeg', '-i',
       os.path.join(encoded_dir, 'image-%04d.png'),
       '-c:v', 'libvpx',
       'temp.webm'
       ])

    print('ffmpeg decoding')
    subprocess.call([
        'ffmpeg', '-i',
        'temp.webm',
        '-vf', 'scale=800:600',
        os.path.join(decoded_dir, 'image-%04d.png')
        ])

    # decode data
    indirs = [os.path.join(decoded_dir, f) for f in names]
    worker = partial(_decode_one, qr)
    packets_1 = _multiprocess(worker, indirs, info='decoding frames')

    # decode sync
    indirs = [os.path.join(decoded_dir, f) for f in names]
    worker = partial(_decode_one, qr_sync)
    packet_ids = _multiprocess(worker, indirs, info='decoding frames')
    packets_1 = {i[0]: x for i, x in zip(packet_ids, packets_1)}

    acc = 0
    throughput = 0
    sum_len = 0
    for k in packets_0:
        if k not in packets_1:
            continue
        p0 = packets_0[k]
        p1 = packets_1[k]
        ok = [x for x, y in zip(p0, p1) if x == y]
        acc += len(ok)
        sum_len += len(p0)
        throughput += len(ok) * unit_packet_size / 8000

    print()
    print('avg. packet recovery rate', acc / sum_len)
    print('avg. through put (kB per frame)',
          throughput / len(packets_0) * 3)
main()
