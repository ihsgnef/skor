import os
import sys
import time
import json
import random
import numpy as np
import itertools
import subprocess
from PIL import Image
from functools import partial
from multiprocessing import Pool, Manager
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


class BlockCode:
    '''A embedding that is based on color blocks.
    '''

    def pack_packets(self, packets):
        '''split packets into channels then depth'''
        if len(packets) < self.capacity:
            n = self.capacity - len(packets)
            packets += ['' for _ in range(n)]
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


class SimpleCode(BlockCode):

    def __init__(self, tlx=0, tly=0, code_size=(600, 600), block_size=4,
                 depth=1, channels=[], color_space='RGB', alpha=1):
        self.depth = depth
        self.color_space = color_space
        self.channels = channels
        self.n_channels = 1 if len(channels) == 0 else len(channels)
        self.alpha = alpha
        self.tlx = tlx
        self.tly = tly

        h, w = code_size
        self.block_size = block_size
        self.code_size = code_size
        self.array_size = (h // block_size, w // block_size)
        self.frame_capacity = self.array_size[0] * self.array_size[1]
        self.capacity = self.frame_capacity // unit_packet_size
        self.capacity *= self.depth * self.n_channels
        self.capacity_multiplier = 1

    def get_code_array(self, msg):
        '''convert binary string to code array'''
        assert len(msg) == self.frame_capacity
        msg = [int(x) for x in msg]
        msg = np.array(msg).reshape(self.array_size)
        return msg

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
                qs.append(self.get_code_array(packets[i][j]))
            q = self.bucketize(qs)
            # scale
            # q = np.kron(q, np.ones((self.qr_block_size, self.qr_block_size)))
            q = np.repeat(q, self.block_size*np.ones(q.shape[0], np.int), 0)
            q = np.repeat(q, self.block_size*np.ones(q.shape[1], np.int), 1)
            assert q.shape == self.code_size
            channels.append(q)
        background = frame.convert(self.color_space)
        foreground = np.array(background)
        if len(self.channels) == 0:
            # put same data to all channels
            cnl = np.repeat(channels[0][:, :, np.newaxis], 3, axis=2)
            foreground[self.tlx: self.tlx + self.code_size[0],
                       self.tly: self.tly + self.code_size[1], :] = cnl
        else:
            # put data to corresponding channels
            for i, c in enumerate(self.channels):
                cnl = channels[i]
                foreground[self.tlx: self.tlx + self.code_size[0],
                           self.tly: self.tly + self.code_size[1], c] = cnl

        background = np.array(background)
        blended = foreground * self.alpha + background * (1 - self.alpha)
        blended = Image.fromarray(np.uint8(blended), self.color_space)
        blended = blended.convert('RGB')
        return blended

    def decode(self, frame, true_frame=None):
        if self.alpha < 1 and true_frame is None:
            raise ValueError('True frame is required when alpha is not 1')
        frame = np.array(frame.convert(self.color_space))
        # subtract true frame
        if true_frame is not None and self.alpha < 1:
            true_frame = np.array(true_frame.convert(self.color_space))
            frame = (frame - true_frame * (1 - self.alpha)) / self.alpha
            frame = np.clip(frame, 0, 255)
            frame = np.uint8(frame)
        # crop out the code
        frame = frame[self.tlx: self.tlx + self.code_size[0],
                      self.tly: self.tly + self.code_size[1], :]
        # extract encoded channels
        if len(self.channels) == 0:
            channels = [frame[:, :, 0]]
        else:
            channels = [frame[:, :, c] for c in self.channels]
        packets = []
        for c in channels:
            c = blockshaped(c, self.block_size, self.block_size)
            c = np.mean(c, (1, 2))
            c = c.reshape(self.array_size)
            qrs = self.debucketize(c)
            assert len(qrs) == self.depth
            packets.append([])
            for j, q in enumerate(qrs):
                q = q / 255
                m = ''.join(str(int(x > 0.5)) for x in q.ravel())
                packets[-1].append(m)
        return self.unpack_packets(packets)


class QRCode(BlockCode):

    def __init__(self, tlx=0, tly=0, max_code_size=600,
                 depth=4, channels=[], color_space='RGB', alpha=1,
                 version=10, error='L', mode='numeric'):
        '''
        Args:
            max_code_size: maximum allowed code size in pixels
            tlx, tly: top left corner
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
        self.tlx = tlx
        self.tly = tly

        # determine pixel size of code
        self.qr_array_size = get_qr_array_size(self.qr_params)
        self.qr_block_size = max_code_size // self.qr_array_size
        self.qr_code_size = self.qr_array_size * self.qr_block_size

        # determine capacity in unit size messages
        max_packet_size = get_qr_packet_size(self.qr_params)
        self.n_channels = 1 if len(channels) == 0 else len(channels)
        self.capacity = max_packet_size // unit_packet_size
        self.capacity *= self.depth * self.n_channels
        self.frame_capacity = max_packet_size // unit_packet_size
        self.frame_capacity *= unit_packet_size
        self.capacity_multiplier = 3

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
            foreground[self.tlx: self.tlx + self.qr_code_size,
                       self.tly: self.tly + self.qr_code_size, :] = cnl
        else:
            # put data to corresponding channels
            for i, c in enumerate(self.channels):
                cnl = channels[i]
                foreground[self.tlx: self.tlx + self.qr_code_size,
                           self.tly: self.tly + self.qr_code_size, c] = cnl

        background = np.array(background)
        blended = foreground * self.alpha + background * (1 - self.alpha)
        blended = Image.fromarray(np.uint8(blended), self.color_space)
        blended = blended.convert('RGB')
        return blended

    def decode(self, frame, true_frame=None):
        if self.alpha < 1 and true_frame is None:
            raise ValueError('True frame is required when alpha is not 1')
        frame = np.array(frame.convert(self.color_space))
        # subtract true frame
        if true_frame is not None:
            true_frame = np.array(true_frame.convert(self.color_space))
            frame = (frame - true_frame * (1 - self.alpha)) / self.alpha
            frame = np.clip(frame, 0, 255)
            frame = np.uint8(frame)
        # crop out the code
        frame = frame[self.tlx: self.tlx + self.qr_code_size,
                      self.tly: self.tly + self.qr_code_size, :]
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
                    m = '0' * self.frame_capacity
                else:
                    m = m[0].data.decode('utf-8')
                packets[-1].append(m)
        return self.unpack_packets(packets)


def _encode(mapper, inputs):
    (input_path, output_path, packets), queue = inputs
    frame = Image.open(input_path)
    frame = mapper.encode(frame, packets)
    frame.save(output_path, 'PNG')
    if queue is not None:
        queue.put(0)


def _encode_sync(emb, emb_sync, inputs):
    (input_path, output_path, packets, pid), queue = inputs
    frame = Image.open(input_path)
    frame = emb.encode(frame, packets)
    frame = emb_sync.encode(frame, [pid])
    frame.save(output_path, 'PNG')
    if queue is not None:
        queue.put(0)


def _decode(mapper, inputs):
    input_path, queue = inputs
    frame = Image.open(input_path)
    packets = mapper.decode(frame)
    if queue is not None:
        queue.put(0)
    return packets


def _decode_sync(emb, emb_sync, inputs):
    input_path, queue = inputs
    frame = Image.open(input_path)
    pid = emb_sync.decode(frame)[0]
    real_frame = Image.open(pid)
    packets = emb.decode(frame, real_frame)
    if queue is not None:
        queue.put(0)
    return pid, packets


def _multiprocess(worker, inputs, info='', multi=False):
    if not multi:
        return [worker((x, None)) for x in inputs]

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
        if info != '':
            sys.stderr.write(output.format(info, size, total_size))
        time.sleep(0.1)
    if info != '':
        sys.stderr.write('\n')
    return result.get()


def main(emb, emb_sync, video=False):
    FNULL = open(os.devnull, 'w')

    if not os.path.isdir(frames_dir):
        os.makedirs(frames_dir)
        print('generating initial frames')
        subprocess.call([
            'ffmpeg',
            '-i', 'rms.webm',
            '-vf', 'scale=800:600',
            os.path.join(frames_dir, 'image-%04d.png')
            ])

    subprocess.call(['rm', 'temp.webm'],
                    stdout=FNULL, stderr=subprocess.STDOUT)
    if os.path.isdir(encoded_dir):
        subprocess.call(['rm', '-r', 'encoded_frames'])
    if os.path.isdir(decoded_dir):
        subprocess.call(['rm', '-r', 'decoded_frames'])

    os.makedirs(encoded_dir, exist_ok=True)
    os.makedirs(decoded_dir, exist_ok=True)

    names = sorted(os.listdir(frames_dir))
    names = [x for x in names if x.endswith('.png')][:30]
    in_dirs = [os.path.join(frames_dir, x) for x in names]
    packets = [[get_unit_packet() for _ in range(emb.capacity)]
               for _ in names]
    packets_0 = {i: x for i, x in zip(in_dirs, packets)}

    # encode data
    enc_dirs = [os.path.join(encoded_dir, x) for x in names]
    inputs = list(zip(in_dirs, enc_dirs, packets, in_dirs))
    worker = partial(_encode_sync, emb, emb_sync)
    _multiprocess(worker, inputs)

    if video:

        # video encoding and decoding
        print('ffmpeg encoding')
        subprocess.call([
           'ffmpeg', '-i',
           os.path.join(encoded_dir, 'image-%04d.png'),
           '-c:v', 'libvpx',
           'temp.webm'
           ],
           stdout=FNULL, stderr=subprocess.STDOUT)

        print('ffmpeg decoding')
        subprocess.call([
            'ffmpeg', '-i',
            'temp.webm',
            '-vf', 'scale=800:600',
            os.path.join(decoded_dir, 'image-%04d.png')
            ],
            stdout=FNULL, stderr=subprocess.STDOUT)

    # decode data
    if video:
        dec_dirs = [os.path.join(decoded_dir, f) for f in names]
    else:
        dec_dirs = [os.path.join(encoded_dir, f) for f in names]
    worker = partial(_decode_sync, emb, emb_sync)
    results = _multiprocess(worker, dec_dirs)
    names_decoded, packets_1 = list(map(list, zip(*results)))
    packets_1 = {i: x for i, x in zip(names_decoded, packets_1)}

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
    acc /= sum_len
    throughput = throughput / len(packets_0) * emb.capacity_multiplier
    return acc, throughput


def test_0(emb, emb_sync):
    n0 = 'test_frame.png'
    n1 = 'test_frame_encoded.png'
    ps0 = [get_unit_packet() for _ in range(emb.capacity)]
    inputs = ((n0, n1, ps0, n0), None)
    _encode_sync(emb, emb_sync, inputs)
    pid, ps1 = _decode_sync(emb, emb_sync, (n1, None))
    print(sum([x == y for x, y in zip(ps0, ps1)]) / len(ps0))


# emb = QRCode(
#         max_code_size=600,
#         depth=1, color_space='RGB', channels=[], alpha=0.4,
#         version=30)

simple_params = {
    'block_size': [2, 4, 6, 8, 10],
    'depth': [1, 2, 4],
    'channels': [[], [0], [0, 1], [0, 1, 2], [0, 2]],
    'color_space': ['RGB', 'YCbCr'],
    'alpha': [1, 0.8, 0.6, 0.4]
    }

qr_params = {
    'depth': [1, 2, 4],
    'channels': [[], [0], [0, 1], [0, 1, 2], [0, 2]],
    'color_space': ['RGB', 'YCbCr'],
    'alpha': [1, 0.8, 0.6, 0.4],
    'version': [10, 20, 30]
    }


def param_sweep(param_dict, outfile):
    results = open(outfile, 'w')
    results.write('\n\n\n')
    all_params = list(itertools.product(*param_dict.values()))
    start = time.time()
    for i, params in enumerate(all_params):
        now = time.time()
        params = dict(zip(param_dict.keys(), params))
        eta = (now - start) * (len(all_params) - i) / i if i > 0 else 0
        h = int(eta // 3600)
        m = int((eta - 3600 * h) // 60)
        print('{} / {}, {}:{}:{}'.format(i, len(all_params), h, m, 0))
        print(params)
        emb = SimpleCode(**params)
        emb_sync = QRCode(
                tlx=0, tly=600, max_code_size=200,
                depth=1, color_space='RGB', channels=[],
                version=5, mode='binary')
        acc, tp = main(emb, emb_sync, video=True)
        entry = {'params': params, 'recovery_rate': acc, 'throughput': tp}
        results.write(json.dumps(entry) + '\n')
        results.flush()
        print(acc, tp)
        print()
        print()

# acc, tp = main(emb, emb_sync, video=True)
# print()
# print('avg. packet recovery rate', acc)
# print('avg. through put (kB per frame)', tp)


param_sweep(simple_params, 'sweep_simple.txt')
# param_sweep(qr_params, 'sweep_qr.txt')
