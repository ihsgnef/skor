import os
import sys
import time
import math
import numpy as np
import subprocess
from PIL import Image
from multiprocessing import Pool, Manager
from functools import partial
from abc import ABCMeta, abstractmethod
from random import random

import pyqrcode
from pyzbar.pyzbar import decode

frames_dir = 'frames'
encoded_dir = 'encoded_frames'
decoded_dir = 'decoded_frames'

# putting these in global because I don't want to pass them around
frame_size = (600, 800)
frame_rate = 24
block_size = 12 #assume square for now


class Embedding(metaclass=ABCMeta):

    '''A embedding supports encoding and decoding data into and from a
    frame.
    '''

    @abstractmethod
    def encode(frame, m, *args, **kwargs):
        '''Encode given message into the frame

        Args:
            frame: a PIL.Image (RGB) object
            m: message to be encoded

        Returns:
            frame: the frame with data encoded
        '''
        pass

    @abstractmethod
    def decode(frame, *args, **kwargs):
        '''Decode data from a frame

        Args:
            frame: a PIL.Image (RGB) object

        Return:
            message: the message decoded from the frame, None if
                decoding failed
        '''
        pass


def get_max_msg(n_channels=1):
    '''Get a random message of max size for the current QR parameters
    '''
    block_length = math.sqrt(block_size)
    size = ((frame_size[0]//block_length)*(frame_size[1]//block_length))
    m = np.random.randint(2, size=(int(size),))
    m = ''.join(str(x) for x in m)
    print ("random message:", m)
    return m

class COLOR_BLOCK_1(Embedding):

    def __init__(self, block_size):
        self.block_size = block_size

    def encode(self, frame, m):
        '''Encode message in a frame(currently empty frame)

        Args:
            frame: PIL image object
            m: message to be encoded
        '''
        block_length = math.sqrt(block_size)
        blocks_per_colomn = int(frame_size[0]//block_length)
        cl = (len(m))//blocks_per_colomn
        blocks_per_row = int(frame_size[1]//block_length)
        rl = (len(m))//blocks_per_row #would I need to add anything to len(m)?
        array = np.array([int(x) for x in m])
        array.resize((rl, cl), refcheck=False)
        array = np.uint8(array * 255)
        array = np.repeat(array, int(block_length)*np.ones(array.shape[0], np.int), 0)
        array = np.repeat(array, int(block_length)*np.ones(array.shape[1], np.int), 1)
        img = Image.fromarray(array)
        imgrb = Image.merge('RGB', (img, img, img))
        frame.paste(img)
        # frame.show()
        return frame

    def decode(self, frame):
        block_length = int(math.sqrt(block_size))
        f = frame.crop((0, 0, int(((frame_size[1])//block_length)*block_length), int((frame_size[0]//block_length)*block_length)))
        ms = []
        for y in range(int((frame_size[0])//block_length)):
            for x in range(int((frame_size[1])//block_length)):
                # get block average
                sum = 0
                for i in range(x * block_length, (x+1) * block_length):
                    for j in range(y * block_length, (y+1) * block_length):
                        temp_px = frame.getpixel((i, j))
                        sum += (temp_px[0] + temp_px[1] + temp_px[2])/3
                avg = sum / (block_length**2)
                if avg > 127.5:
                    ms.append("1")
                else:
                    ms.append("0")
        return ''.join(ms)

class COLOR_BLOCK_3(Embedding):

    def __init__(self, block_size):
        self.block_size = block_size

    def encode(self, frame, m):
        '''Encode message in a frame(currently empty frame)

        Args:
            frame: PIL image object
            m: message to be encoded
        '''
        block_length = math.sqrt(block_size)
        blocks_per_colomn = int(frame_size[0]//block_length)
        cl = (len(m))//blocks_per_colomn
        blocks_per_row = int(frame_size[1]//block_length)
        rl = (len(m))//blocks_per_row #would I need to add anything to len(m)?
        array = np.array([int(x) for x in m])
        array.resize((rl, cl), refcheck=False)
        array = np.uint8(array * 255)
        array = np.repeat(array, int(block_length)*np.ones(array.shape[0], np.int), 0)
        array = np.repeat(array, int(block_length)*np.ones(array.shape[1], np.int), 1)
        img = Image.fromarray(array)
        imgrb = Image.merge('RGB', (img, img, img))
        frame.paste(img)
        # frame.show()
        return frame

    def decode(self, frame):
        block_length = int(math.sqrt(block_size))
        f = frame.crop((0, 0, int(((frame_size[1])//block_length)*block_length), int((frame_size[0]//block_length)*block_length)))
        ms = []
        for y in range(int((frame_size[0])//block_length)):
            for x in range(int((frame_size[1])//block_length)):
                # get block average
                sum = 0
                for i in range(x * block_length, (x+1) * block_length):
                    for j in range(y * block_length, (y+1) * block_length):
                        temp_px = frame.getpixel((i, j))
                        sum += (temp_px[0] + temp_px[1] + temp_px[2])/3
                avg = sum / (block_length**2)
                if avg > 127.5:
                    ms.append("1")
                else:
                    ms.append("0")
        return ''.join(ms)

def basic_test():
    frame = Image.open('test_frame2.png')
    mapper = COLOR_BLOCK_1(block_size)
    print ('COLOR_BLOCK_1 with block size of ', block_size)
    m0=get_max_msg()
    f = mapper.encode(frame, m0)
    m1 = mapper.decode(f)
    print("decoded:", m1)
    print(m0 == m1)

def _encode_one(mapper, inputs):
    (input_path, output_path), queue = inputs
    frame = Image.open(input_path)
    m = get_max_msg(mapper)
    f = mapper.encode(frame, m)
    f.save(output_path, "PNG")
    queue.put(0)
    return m


def encode_all(mapper, input_dir, output_dir):
    fnames = sorted(os.listdir(input_dir))[:100]
    fnames = [x for x in fnames if x.endswith('.png')]
    inputs = [os.path.join(input_dir, x) for x in fnames]
    outputs = [os.path.join(output_dir, x) for x in fnames]
    worker = partial(_encode_one, mapper)
    inputs = list(zip(inputs, outputs))
    ms = _multiprocess(worker, inputs, info='encoding frames')
    ms = {f: m for f, m in zip(fnames, ms)}
    return ms


def _decode_one(mapper, inputs):
    input_path, queue = inputs
    frame = Image.open(input_path)
    m = mapper.decode(frame)
    queue.put(0)
    return m


def decode_all(mapper, input_dir):
    fnames = sorted(os.listdir(input_dir))
    fnames = [x for x in fnames if x.endswith('.png')]
    worker = partial(_decode_one, mapper)
    inputs = [os.path.join(input_dir, f) for f in fnames]
    ms = _multiprocess(worker, inputs, info='decoding frames')
    ms = {f: m for f, m in zip(fnames, ms)}
    return ms


def _multiprocess(worker, inputs, info=''):
    total_size = len(inputs)
    output = '\r[{0}] done: {1}/{2}'
    pool = Pool(8)
    manager = Manager()
    queue = manager.Queue()
    inputs = [(x, queue) for x in inputs]
    result = pool.map_async(worker, inputs)
    # monitor loop
    while not result.ready():
        size = queue.qsize()
        sys.stderr.write(output.format(info, size, total_size))
        time.sleep(0.1)
    sys.stderr.write('\n')
    return result.get()


def main():
    if not os.path.isdir(frames_dir):
        os.makedirs(frames_dir)
        print('generating initial frames')
        subprocess.call([
            'ffmpeg', '-i',
            'rms.webm',
            '-vf', 'scale=800:600',
            os.path.join(frames_dir, 'image-%04d.png')
            ])
    os.makedirs(encoded_dir, exist_ok=True)
    os.makedirs(decoded_dir, exist_ok=True)
    mapper = COLOR_BLOCK_1()
    ms0 = encode_all(mapper, frames_dir, encoded_dir)

    print('ffmpeg encoding')
    subprocess.call(["rm", "temp.webm"])
    subprocess.call([
       'ffmpeg', '-i',
       os.path.join(encoded_dir, 'image-%04d.png'),
       "-c:v", "libvpx",
       "temp.webm"
       ])

    print('ffmpeg decoding')
    subprocess.call([
        'ffmpeg', '-i',
        'temp.webm',
        '-vf', 'scale=800:600',
        os.path.join(decoded_dir, 'image-%04d.png')
        ])

    ms1 = decode_all(mapper, decoded_dir)

    acc = 0
    throughput = 0
    for k in ms0:
        acc += ms0[k] == ms1[k]
        if ms1[k] is not None:
            match = int(ms0[k], 2) & int(ms1[k], 2)
            throughput += '{:b}'.format(match).count('1')
    print()
    print('perfectly recovered {} of frames'.format(acc / len(ms0)))
    print('through put (bps): {}'.format(
        throughput / len(ms0) * frame_rate))


if __name__ == '__main__':
    main()
