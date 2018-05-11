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

from util import get_qr_packet_size, get_qr_array_size

frames_dir = 'frames'
encoded_dir = 'encoded_frames'
decoded_dir = 'decoded_frames'

# putting these in global because I don't want to pass them around
frame_size = (600, 800)
frame_rate = 24

# fixed-size packets
unit_packet_size = 100


def get_unit_packet():
    m = np.random.randint(0, 2, (unit_packet_size))
    m = ''.join(str(x) for x in m)
    return m


class Embedding(metaclass=ABCMeta):

    '''A embedding supports encoding and decoding data into and from a
    frame.
    '''

    @abstractmethod
    def encode(frame, packets, *args, **kwargs):
        '''Encode given message into the frame

        Args:
            frame: a PIL.Image (RGB) object
            packets: list of packets to encode

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
            packets: decoded packets, None if decoding failed
        '''
        pass


class QR(Embedding):

    def __init__(self, max_code_size=400,
                 version=10, error='L', mode='numeric',
                 depth=4, channels=[], color_space='RGB',
                 alpha=1.0):
        '''
        Args:
            max_code_size: maximum allowed code size in pixels
            version, error, mode: QR parameters
            depth: number of bits to encode in each block and channel
            channel: indices of channels to embed data, tie channels if empty
            color_space: embed in what color space
            alpha: transparency of embedding, for all channels, 1 is opaque
        '''
        self.qr_params = {'error': error, 'version': version, 'mode': mode}
        self.depth = depth
        self.color_space = color_space
        self.channels = channels
        self.alpha = alpha

        # determine pixel size of code
        self.qr_array_size = get_qr_array_size(self.qr_params)
        self.qr_block_size = max_code_size // self.qr_array_size
        self.qr_code_size = self.qr_array_size * self.qr_block_size

        # determine capacity in unit size messages
        max_packet_size = get_qr_packet_size(self.qr_params)
        self.n_channels = 1 if len(channels) == 0 else len(channels)
        self.capacity = max_packet_size // unit_packet_size
        self.capacity *= self.depth * self.n_channels

    def get_qr_array(self, m):
        '''Get two dimensional QR array for a message string.

        Args:
            m: the message to be encoded
        Returns:
            qr: 2D numpy array
        '''
        qr = pyqrcode.create(m, **self.qr_params)
        qr = qr.text().split("\n")[:-1]
        qr = np.array([[1 - int(z) for z in x] for x in qr])
        return qr

    def pack(self, packets):
        '''split packets into channels then depth'''
        if len(packets) < self.capacity:
            n = self.capacity - len(packets)
            packets += ['0' * unit_packet_size for _ in range(n)]
        row_size = len(packets) // self.n_channels
        col_size = len(packets) // (self.n_channels * self.depth)
        pcks = []
        for i in range(self.n_channels):
            row = packets[i * row_size: (i+1) * row_size]
            pcks.append([])
            for j in range(self.depth):
                col = row[j * col_size: (j+1) * col_size]
                pcks[i].append(''.join(col))
        return pcks

    def unpack(self, packets):
        '''flatten 2d packet array into a list of packets'''
        assert len(packets) == self.n_channels
        assert len(packets[0]) == self.depth
        pkts = []
        for i in range(self.n_channels):
            for j in range(self.depth):
                ps = packets[i][j]
                for st in range(0, len(ps), unit_packet_size):
                    pkts.append(ps[st: st + unit_packet_size])
        return pkts

    def encode(self, frame, packets):
        '''Encode message in a frame by overlay the QR code on top.

        Args:
            frame: PIL image object in RGB mode
            packets: list of packets to be encoded
        '''
        assert 256 % (2 ** (self.depth + 1)) == 0
        half_bucket_size = 256 // (2 ** (self.depth + 1))
        packets = self.pack(packets)

        # get QR code for each channel and depth
        channels = []
        arr_size = (self.qr_array_size, self.qr_array_size)
        for i in range(self.n_channels):
            q = np.zeros(arr_size, dtype=np.int32)
            for j in range(self.depth):
                # merge across depth
                q += self.get_qr_array(packets[i][j]) * (2 ** j)
            # convert to [0, 255)
            q = q * 2 * half_bucket_size + half_bucket_size
            # scale
            q = np.kron(q, np.ones((self.qr_block_size, self.qr_block_size)))
            assert q.shape[0] == self.qr_code_size
            channels.append(q)

        background = frame.convert(self.color_space)
        foreground = np.array(background)
        if len(self.channels) == 0:
            # put same data to all channels
            cnl = np.repeat(channels[0][:, :, np.newaxis], 3, axis=2)
            cnl = np.uint8(cnl)
            foreground[:self.qr_code_size, :self.qr_code_size, :] = cnl
        else:
            # put data to corresponding channels
            for i, c in enumerate(self.channels):
                cnl = np.uint8(channels[i])
                foreground[:self.qr_code_size, :self.qr_code_size, :] = cnl

        foreground = Image.fromarray(foreground, self.color_space)
        # blended = Image.blend(background, foreground, self.alpha)
        blended = foreground
        blended = blended.convert('RGB')
        return blended

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


class QR_YUV(Embedding):

    def __init__(self, depth=80, qr_size=(400, 400), error='L', version=10,
                 mode='numeric'):
        self.depth = depth
        self.qr_params = {'error': error,
                          'version': version,
                          'mode': mode}
        self.qr_size = qr_size

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


qr = QR()
packets = [get_unit_packet() for _ in range(qr.capacity)]
frame = Image.open('test_frame.png')
frame = qr.encode(frame, packets)
frame.show()
