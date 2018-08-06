import os
import sys
import time
import json
import random
import numpy as np
import itertools
import subprocess
import textwrap
from PIL import Image
from functools import partial
from multiprocessing import Pool, Manager
import pyqrcode
from pyzbar.pyzbar import decode as qr_decode
import pickle

from util import get_qr_packet_size, get_qr_array_size
from main import QRCode, BlockCode, SimpleCode
video_dir = 'networkendend.mp4'
frames_dir = 'data/decoded_frames'
decoded_dir = 'data/decoded_frames'

# putting these in global because I don't want to pass them around
frame_size = (600, 800)
frame_rate = 24

# fixed-size packets
unit_packet_size = 100
def main():
    # if os.path.isdir(decoded_dir):
    #     subprocess.call(['rm', '-r', decoded_dir])
    # os.makedirs(decoded_dir, exist_ok=True)
    # print('ffmpeg decoding')
    # subprocess.call([
    #     'ffmpeg', '-i',
    #     'rgb_4_1_.4_skype.mp4',#change this
    #     '-vf', 'scale=900:600',
    #     os.path.join(decoded_dir, 'image-%04d.png')
    #     ],
    #     stdout=None, stderr=subprocess.STDOUT)

    # frame = Image.open('data/decoded_frames/image-0001.png')
    # frame.crop((147.7,63.5,600.7,600)).show()
    # sys.exit()
    test_end_end()
    test_rate()

def test_end_end():
    packets_0 = pickle.load(open("rgb_4_1_.2.p", "rb")) #CHANGE THIS FOR EACH RUN
    count = 0;
    total = 0;
    d = dict()
    for path in os.listdir(frames_dir):
        if path.endswith(".png"):
            pid = _decode_sync(os.path.join(frames_dir, path))
            packet_received = _decode_packet(os.path.join(frames_dir, path), id = pid)
            d[pid] = packet_received
    for key, value in d.items():
        for actual, decoded in zip(packets_0.get(key), value):
            if(actual==decoded):
                count = count+1
            total = total+1
    print("number of correctly decoded base packets", count)
    print("over number of total seen base packets", total)
    totalpackets = len(packets_0.get('data/frames/image-0001.png'))*766
    print("over number of total base packets in video", totalpackets)


def test_rate():
    x = set([])
    count = 0
    for path in os.listdir(frames_dir):
        if path.endswith(".png"):
            pid = _decode_sync(os.path.join(frames_dir, path))
            x.add(pid)
            count = count+1
    print(len(x), "/766")
    print("number of frames processed:", count)

#for basicblock
def _decode_sync(input_path):
    frame = Image.open(input_path)
    # syncframe = frame.crop((735,80,910,260)) #blocksize 8
    # syncframe = frame.crop((600,70,735,225)) #blocksize4rgb
    syncframe = frame.crop((600,60,730,230))
    q = Image.fromarray(np.uint8(syncframe), 'RGB')
    m = qr_decode(q)
    return m[0].data.decode('utf-8')

def _decode_packet(input_path, id=''):
    frame = Image.open(input_path)
    # dataframe = frame.crop((182,75,735,700)) for blocksize 8
    # dataframe = frame.crop((149,64,600,600)) #blocksize4rgb
    dataframe= frame.crop((147.7,63.5,600.7,600))
    dataframe = dataframe.resize((600,600))
    q = Image.fromarray(np.uint8(dataframe), 'RGB')
    emb = SimpleCode( #change this
            block_size=4,
            depth=1, color_space='RGB', channels=[], alpha=.2)
    return emb.decode(q, true_frame=Image.open(id).crop((0,0,600,600)))
# for qr
# def _decode_sync(input_path):
#     frame = Image.open(input_path)
#     # frame = frame.crop((555,0,735,200)) #crop with extra space at boundaries for networkqrsync.mp4
#     syncframe = frame.crop((632,60,780,225)) #syncqr
#     q = Image.fromarray(np.uint8(syncframe), 'RGB')
#     m = qr_decode(q)
#     return m[0].data.decode('utf-8')
#
# def _decode_packet(input_path):
#     frame = Image.open(input_path)
#     packetframe = frame.crop((150,60,580,535)) #packetqr
#     q = Image.fromarray(np.uint8(packetframe), 'RGB')
#     m = qr_decode(q)
#     return m[0].data.decode('utf-8')

main()
