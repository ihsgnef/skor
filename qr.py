import os
import sys
import time
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


def get_qr_array(m, qr_params):
    '''Get two dimensional QR array for a message.
        Args:
            m: the message to be encoded
    '''
    qr = pyqrcode.create(m, **qr_params)
    qr = qr.text().split("\n")[:-1]
    qr = np.array([[1 - int(z) for z in x] for x in qr])
    return qr


def get_max_qr_msg(qr_params, n_channels=1):
    '''Get a random message of max size for the current QR parameters
    '''
    # TODO
    m = np.random.randint(2, size=(2000 * n_channels,))
    m = ''.join(str(x) for x in m)
    return m


class QR_RGB_1(Embedding):

    def __init__(self, qr_size=(400, 400), error='L', version=10,
                 mode='numeric'):
        self.qr_size = qr_size
        self.qr_params = {'error': error,
                          'version': version,
                          'mode': mode}

    def encode(self, frame, m):
        '''Encode message in a frame by overlay the QR code on top.

        Args:
            frame: PIL image object
            m: message to be encoded
            size: size of QR code

        # TODO: alpha
        '''
        qr_arr = get_qr_array(m, self.qr_params)
        qr_rgb = np.repeat(qr_arr[:, :, np.newaxis], 3, axis=2)
        qr_rgb = np.uint8(qr_rgb * 255)
        qr_img = Image.fromarray(qr_rgb, "RGB")
        qr_img = qr_img.resize(self.qr_size)
        frame.paste(qr_img)
        return frame

    def decode(self, frame):
        f = frame.crop((0, 0, self.qr_size[0], self.qr_size[1]))
        self.average_blocks(f)
        data = decode(f)
        if len(data) == 0:
            return None
        else:
            return data[0].data.decode("utf-8")

    def average_blocks(self, frame):
        ''' take a frame and map each block to one color according to
        the average.  this is an in place function. The qr is assumed
        to be starting at the top right of the frame.

        Args:
            frame: PIL image object
        '''

        side_len_blocks = (self.qr_params['version'] - 1) * 4 + 21 + 8
        if self.qr_size[0] == self.qr_size[1]:

            if self.qr_size[0] % side_len_blocks == 0:
                block_size_px = int(self.qr_size[0] / side_len_blocks)
            else:
                raise ValueError("qr not properly scaled")

        else:
            raise ValueError("qr not square")

        for x in range(side_len_blocks):
            for y in range(side_len_blocks):
                # get block average
                sum = 0
                for i in range(x * block_size_px, (x+1) * block_size_px):
                    for j in range(y * block_size_px, (y+1) * block_size_px):
                        temp_px = frame.getpixel((i, j))
                        sum += (temp_px[0] + temp_px[1] + temp_px[2])/3
                avg = sum / (block_size_px**2)
                if avg > 127.5:
                    avg = 255
                else:
                    avg = 0

                # set each block
                for i in range(x * block_size_px, (x+1) * block_size_px):
                    for j in range(y * block_size_px, (y+1) * block_size_px):
                        frame.putpixel((i, j), (avg, avg, avg))

        if random() < 0.1:
            frame.show()


class QR_RGB_3(Embedding):

    def __init__(self, qr_size=(400, 400), error='L', version=10,
                 mode='numeric'):
        self.qr_size = qr_size
        self.qr_params = {'error': error,
                          'version': version,
                          'mode': mode}

    def encode(self, frame, m):
        # encode message in rgb channels separately
        ml = (len(m) + 3) // 3
        ms = [m[i * ml: i * ml + ml] for i in range(3)]
        assert len(ms) == 3
        qr_rgb = [get_qr_array(m, self.qr_params) for m in ms]
        qr_rgb = np.stack(qr_rgb, axis=2)
        qr_rgb = np.uint8(qr_rgb * 255)
        qr_img = Image.fromarray(qr_rgb, "RGB")
        qr_img = qr_img.resize(self.qr_size)
        frame.paste(qr_img)
        return frame

    def decode(self, frame):
        f = np.array(frame.crop((0, 0, self.qr_size[0], self.qr_size[1])))
        ms = []
        for i in range(3):
            qr = f[:, :, i][:, :, np.newaxis]
            qr = np.repeat(qr, 3, axis=2)
            qr = Image.fromarray(qr, "RGB")
            data = decode(qr)
            if len(data) == 0:
                return None
            else:
                ms.append(decode(qr)[0].data.decode("utf-8"))
        return ''.join(ms)


class QR_YUV(Embedding):

    def __init__(self, depth=80, qr_size=(400, 400), error='L', version=10,
                 mode='numeric'):
        self.depth = depth
        self.qr_size = qr_size
        self.qr_params = {'error': error,
                          'version': version,
                          'mode': mode}

    def encode(self, frame, m):
        qr = get_qr_array(m, self.qr_params)
        qr = np.repeat(qr[:, :, np.newaxis], 3, axis=2)
        qr = np.uint8(np.clip(qr * self.depth, 0, 255))
        qr = np.array(Image.fromarray(qr, "RGB").resize(self.qr_size))
        f = frame.crop((0, 0, self.qr_size[0], self.qr_size[1]))
        f = np.array(f.convert("YCbCr"))
        yuv = np.stack([qr[:, :, 0], f[:, :, 1], f[:, :, 2]], axis=2)
        yuv = Image.fromarray(yuv, "YCbCr")
        rgb = yuv.convert("RGB")
        frame.paste(rgb)
        return frame

    def decode(self, frame):
        f = frame.convert("YCbCr")
        f = f.crop((0, 0, self.qr_size[0], self.qr_size[1]))
        f = np.array(f, dtype=np.int32)
        # FIXME this is hacky
        f[:, :, 0] = np.clip(f[:, :, 0] - self.depth + 10, 0, 255)
        f[:, :, 0] = np.clip(f[:, :, 0] * 255, 0, 255)
        f = np.uint8(f)
        yuv = Image.fromarray(f, "YCbCr")
        rgb = yuv.convert("RGB")
        data = decode(rgb)
        if len(data) == 0:
            return None
        else:
            return data[0].data.decode("utf-8")


def basic_test():
    frame = Image.open('test_frame.png')

    mappers = [QR_RGB_1, QR_RGB_3, QR_YUV]
    names = ['QR_RGB_1', 'QR_RGB_3', 'QR_YUV']

    for _mapper, name in zip(mappers, names):
        print(name)
        mapper = _mapper()
        m0 = get_max_qr_msg(mapper.qr_params)
        f = mapper.encode(frame, m0)
        m1 = mapper.decode(f)
        print(m0 == m1)


def _encode_one(mapper, inputs):
    (input_path, output_path), queue = inputs
    frame = Image.open(input_path)
    m = get_max_qr_msg(mapper.qr_params)
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

    mapper = QR_RGB_1(qr_size=(435, 435), version=20)
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
