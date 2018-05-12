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

frames_dir = 'frames'
encoded_dir = 'encoded_frames'
decoded_dir = 'decoded_frames'

# putting these in global because I don't want to pass them around
frame_size = (600, 800)
frame_rate = 24

# fixed-size packets
unit_packet_size = 100


def get_unit_packet():
    m = ''.join(random.choice(('0', '1')) for _ in range(unit_packet_size))
    return m


def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h // nrows, nrows, -1, ncols)
               .swapaxes(1, 2)
               .reshape(-1, nrows, ncols))


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
                 alpha=1):
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
        self.packet_size = max_packet_size // unit_packet_size
        self.packet_size *= unit_packet_size

    def get_qr_array(self, m):
        '''Get two dimensional QR array for a message string.

        Args:
            m: the message to be encoded
        Returns:
            qr: 2D numpy array
        '''
        qr = pyqrcode.create(m, **self.qr_params)
        qr = qr.text().split('\n')[:-1]
        qr = np.array([[1 - int(z) for z in x] for x in qr])
        return qr

    def pack_packets(self, packets):
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

    def unpack_packets(self, packets):
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

    def bucketize(self, qs):
        '''depth * [height, width] -> [height, width]'''
        assert len(qs) == self.depth

        if self.depth == 1:
            return qs[0] * 255

        assert 256 % (2 ** (self.depth + 1)) == 0
        half_bin_size = 256 // (2 ** (self.depth + 1))
        # convert to [0, 255)
        merged = np.zeros_like(qs[0], dtype=np.int32)
        for i, q in enumerate(qs):
            merged += q * (2 ** i)
        merged = merged * 2 * half_bin_size + half_bin_size
        return merged

    def debucketize(self, frame):
        '''[height, width] -> depth * [height, width]'''
        if self.depth == 1:
            return [frame]
        bin_size = 256 // (2 ** self.depth)
        bins = list(range(0, 255 + bin_size, bin_size))
        frame = np.digitize(frame, bins) - 1
        fs = []
        for i in range(self.depth):
            fs.append((frame % 2) * 255)
            frame //= 2
        return fs

    def encode(self, frame, packets):
        '''Encode message in a frame by overlay the QR code on top.

        Args:
            frame: PIL image object in RGB mode
            packets: list of packets to be encoded
        '''
        # get QR code for each channel and depth
        packets = self.pack_packets(packets)
        channels = []
        for i in range(self.n_channels):
            qs = []
            for j in range(self.depth):
                # merge across depth
                qs.append(self.get_qr_array(packets[i][j]))
            q = self.bucketize(qs)
            # scale
            # q = np.kron(q, np.ones((self.qr_block_size, self.qr_block_size)))
            q = np.repeat(q, self.qr_block_size*np.ones(q.shape[0], np.int), 0)
            q = np.repeat(q, self.qr_block_size*np.ones(q.shape[1], np.int), 1)
            assert q.shape[0] == self.qr_code_size
            channels.append(q)
        background = frame.convert(self.color_space)
        foreground = np.array(background)
        if len(self.channels) == 0:
            # put same data to all channels
            cnl = np.repeat(channels[0][:, :, np.newaxis], 3, axis=2)
            foreground[:self.qr_code_size, :self.qr_code_size, :] = cnl
        else:
            # put data to corresponding channels
            for i, c in enumerate(self.channels):
                cnl = channels[i]
                foreground[:self.qr_code_size, :self.qr_code_size, c] = cnl

        background = np.array(background)
        blended = foreground * self.alpha + background * (1 - self.alpha)
        blended = Image.fromarray(np.uint8(blended), self.color_space)
        blended = blended.convert('RGB')

        # foreground = Image.fromarray(np.uint8(foreground), self.color_space)
        # blended = Image.blend(background, foreground, self.alpha)
        # blended = blended.convert('RGB')
        return blended

    def decode(self, frame, true_frame=None):
        if self.alpha < 1 and true_frame is None:
            raise ValueError('True frame is required when alpha is not 1')
        frame = np.array(frame.convert(self.color_space))
        # subtract true frame
        if true_frame is not None:
            true_frame = np.array(true_frame.convert(self.color_space))
            frame = (frame - true_frame * (1 - self.alpha)) / self.alpha
            frame = np.uint8(frame)
        # crop out the code
        frame = frame[:self.qr_code_size, :self.qr_code_size, :]
        # extract encoded channels
        if len(self.channels) == 0:
            channels = [frame[:, :, 0]]
        else:
            channels = [frame[:, :, c] for c in self.channels]
        packets = []
        for c in channels:
            c = blockshaped(c, self.qr_block_size, self.qr_block_size)
            c = np.mean(c, (1, 2))
            c = c.reshape(self.qr_array_size, self.qr_array_size)
            qrs = self.debucketize(c)
            assert len(qrs) == self.depth
            packets.append([])
            for j, q in enumerate(qrs):
                h, w = q.shape
                q = np.repeat(q[:, :, np.newaxis], 3, axis=2)
                q = np.repeat(q, self.qr_block_size * np.ones(h, np.int), 0)
                q = np.repeat(q, self.qr_block_size * np.ones(w, np.int), 1)
                q = Image.fromarray(np.uint8(q), 'RGB')
                m = qr_decode(q)
                if len(m) == 0:
                    m = '0' * self.packet_size
                else:
                    m = m[0].data.decode('utf-8')
                packets[-1].append(m)
        return self.unpack_packets(packets)

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
                raise ValueError('qr not properly scaled')

        else:
            raise ValueError('qr not square')

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

        # if random() < 0.1:
        #     frame.show()


def _encode_one(mapper, inputs):
    (input_path, output_path), queue = inputs
    frame = Image.open(input_path)
    packets = [get_unit_packet() for _ in range(mapper.capacity)]
    f = mapper.encode(frame, packets)
    f.save(output_path, 'PNG')
    queue.put(0)
    return packets


def encode_all(mapper, input_dir, output_dir):
    fnames = sorted(os.listdir(input_dir))[:30]
    fnames = [x for x in fnames if x.endswith('.png')]
    inputs = [os.path.join(input_dir, x) for x in fnames]
    outputs = [os.path.join(output_dir, x) for x in fnames]
    worker = partial(_encode_one, mapper)
    inputs = list(zip(inputs, outputs))
    packets = _multiprocess(worker, inputs, info='encoding frames')
    packets = {f: m for f, m in zip(fnames, packets)}
    return packets


def _decode_one(mapper, inputs):
    input_path, queue = inputs
    frame = Image.open(input_path)
    packets = mapper.decode(frame)
    queue.put(0)
    return packets


def decode_all(mapper, input_dir):
    fnames = sorted(os.listdir(input_dir))
    fnames = [x for x in fnames if x.endswith('.png')]
    worker = partial(_decode_one, mapper)
    inputs = [os.path.join(input_dir, f) for f in fnames]
    packets = _multiprocess(worker, inputs, info='decoding frames')
    packets = {f: m for f, m in zip(fnames, packets)}
    return packets


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
            'ffmpeg',
            '-i', 'rms.webm',
            '-ss', '00:00:13',
            '-to', '00:01:50',
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

    qr = QR(max_code_size=600, version=40, depth=1, color_space='RGB',
            channels=[])
    packets_0 = encode_all(qr, frames_dir, encoded_dir)

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

    packets_1 = decode_all(qr, decoded_dir)

    acc = 0
    throughput = 0
    sum_len = 0
    for k in packets_0:
        if k not in packets_1:
            continue
        ok = [x for x, y in zip(packets_0[k], packets_1[k]) if x == y]
        acc += len(ok)
        sum_len += len(packets_0[k])
        throughput += len(ok) * unit_packet_size / 8000

    print()
    print('avg. packet recovery rate', acc / sum_len)
    print('avg. through put (kB per frame)', throughput / len(packets_0))


# qr = QR(max_code_size=600, version=10, depth=3, color_space='RGB',
#         channels=[0, 1, 2])
# packets = [get_unit_packet() for _ in range(qr.capacity)]
# original = Image.open('test_frame.png')
# encoded = qr.encode(original, packets)
# encoded.save('test_encoded.png', 'PNG')
# ps = qr.decode(Image.open('test_encoded.png'))
# recovered = [x for x, y in zip(packets, ps) if x == y]
# print('packet recovery rate', len(recovered) / len(packets))
# print('through put (kB per frame)', len(recovered) * unit_packet_size / 8000)

main()
