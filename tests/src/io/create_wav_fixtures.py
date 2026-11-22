#! /usr/bin/env python3
"""Generate a fixed corpus of hand-crafted RIFF/WAVE files for the
aubio_source_wavread regression tests.

Every file is built byte by byte from the RIFF specification instead of
going through the :mod:`wave` module, so that bit widths (8/16/24/32),
non standard chunk ordering ("fact" before "fmt ", "LIST", "JUNK", ...),
odd-sized chunks with their word-alignment padding byte and truncated
files can all be exercised deterministically.

The sample values follow closed-form expressions that are mirrored in
tests/src/io/test-source_wavread_riff.c:

  mono frame n (0 <= n < NFRAMES), signed value s(n):
    8-bit   s(n) = 4*n  - 14        (stored unsigned as s(n) + 128)
    16-bit  s(n) = 1000*n  - 3500
    24-bit  s(n) = 100000*n  - 350000
    32-bit  s(n) = 1000000*n - 3500000
  stereo frame n: left = s(n), right = -s(n)

Run with ``python3 -I create_wav_fixtures.py <output_directory>``.
"""

import os
import struct
import sys

NFRAMES = 8
SR = 8000


def le32(value):
    return struct.pack('<I', value & 0xffffffff)


def le16(value):
    return struct.pack('<H', value & 0xffff)


def chunk(fourcc, payload):
    """A RIFF sub-chunk: FourCC + size + payload + padding byte when
    the payload size is odd."""
    assert isinstance(fourcc, bytes) and len(fourcc) == 4
    out = fourcc + le32(len(payload)) + payload
    if len(payload) & 1:
        out += b'\x00'  # RIFF word-alignment padding, not counted in size
    return out


def fmt_payload(bits, channels, samplerate=SR, ext_bytes=b''):
    blockalign = channels * bits // 8
    byterate = samplerate * blockalign
    body = le16(1)                       # wFormatTag = PCM
    body += le16(channels)               # nChannels
    body += le32(samplerate)             # nSamplesPerSec
    body += le32(byterate)               # nAvgBytesPerSec
    body += le16(blockalign)             # nBlockAlign
    body += le16(bits)                   # wBitsPerSample
    body += ext_bytes
    return body


def signed_values(bits):
    if bits == 8:
        return [4 * n - 14 for n in range(NFRAMES)]
    if bits == 16:
        return [1000 * n - 3500 for n in range(NFRAMES)]
    if bits == 24:
        return [100000 * n - 350000 for n in range(NFRAMES)]
    if bits == 32:
        return [1000000 * n - 3500000 for n in range(NFRAMES)]
    raise ValueError(bits)


def encode_sample(value, bits):
    if bits == 8:
        # 8-bit PCM is unsigned, centred on 128
        return struct.pack('<B', (value + 128) & 0xff)
    if bits == 16:
        return struct.pack('<h', value)
    if bits == 24:
        return struct.pack('<I', value & 0xffffff)[0:3]
    if bits == 32:
        return struct.pack('<i', value)
    raise ValueError(bits)


def data_payload(bits, channels):
    values = signed_values(bits)
    out = b''
    for n in range(NFRAMES):
        frame = [values[n]]
        if channels == 2:
            frame.append(-values[n])
        for value in frame:
            out += encode_sample(value, bits)
    return out


def fact_payload():
    # dwSampleLength, 4 bytes
    return le32(NFRAMES)


def riff_wave(body):
    return b'RIFF' + le32(4 + len(body)) + b'WAVE' + body


def layout_chunks(layout, bits, channels):
    """Return the ordered list of sub-chunks for a named layout."""
    data = chunk(b'data', data_payload(bits, channels))
    if layout == 'L1':
        # canonical order: fmt then data
        return [chunk(b'fmt ', fmt_payload(bits, channels)), data]
    if layout == 'L2':
        # fact chunk before fmt : used to make the old parser fail
        return [chunk(b'fact', fact_payload()),
                chunk(b'fmt ', fmt_payload(bits, channels)), data]
    if layout == 'L3':
        # even-sized JUNK and LIST chunks between fmt and data
        junk = chunk(b'JUNK', b'\xab' * 8)
        info = chunk(b'LIST', b'INFO' + b'name' + le32(4) + b'aubio' + b'\x00')
        return [chunk(b'fmt ', fmt_payload(bits, channels)), junk, info, data]
    if layout == 'L4':
        # two odd-sized chunks (5 and 3 bytes) around fmt, each
        # followed by a RIFF padding byte
        odd0 = chunk(b'aub0', b'\x01\x02\x03\x04\x05')
        odd1 = chunk(b'aub1', b'\x06\x07\x08')
        return [odd0, chunk(b'fmt ', fmt_payload(bits, channels)), odd1, data]
    if layout == 'L5':
        # 18-byte fmt chunk (cbSize extension) and a fact after data
        fmt18 = chunk(b'fmt ', fmt_payload(bits, channels, ext_bytes=le16(0)))
        return [fmt18, data, chunk(b'fact', fact_payload())]
    raise ValueError(layout)


def write_wav(path, bits, channels, layout):
    body = b''.join(layout_chunks(layout, bits, channels))
    with open(path, 'wb') as fid:
        fid.write(riff_wave(body))


def write_hardcoded_odd_pad(path):
    """Fully fixed byte layout used for the write-hardcoded RIFF
    padding offset assertions in the C test.

        0  "RIFF"
        4  riff size
        8  "WAVE"
       12  "aub0"                      odd chunk, 5 data bytes
       16  size = 5
       20  5 data bytes  (20..24)
       25  padding byte
       26  "fmt "
       30  size = 16
       34  16 PCM WAVEFORMAT bytes (34..49)
       50  "aub1"                      odd chunk, 3 data bytes
       54  size = 3
       58  3 data bytes (58..60)
       61  padding byte
       62  "data"
       66  size = 4                    (2 frames, 16-bit mono)
       70  4 sample bytes (70..73)
    """
    fmt = fmt_payload(16, 1)
    assert len(fmt) == 16
    body = b''
    body += b'aub0' + le32(5) + b'\x01\x02\x03\x04\x05' + b'\x00'
    body += b'fmt ' + le32(16) + fmt
    body += b'aub1' + le32(3) + b'\x06\x07\x08' + b'\x00'
    body += b'data' + le32(4)
    # 16-bit samples: -1234 (frame 0) and 4321 (frame 1)
    body += struct.pack('<hh', -1234, 4321)
    blob = riff_wave(body)
    assert len(blob) == 74, len(blob)
    with open(path, 'wb') as fid:
        fid.write(blob)


def write_odd_data_chunk(directory):
    """8-bit mono data chunk with an odd data size (5 frames): legal
    because blockalign is 1, and therefore followed by a RIFF padding
    byte before the trailing 'fact' chunk."""
    values = signed_values(8)[:5]
    samples = b''.join(encode_sample(v, 8) for v in values)
    assert len(samples) == 5  # odd, so one padding byte is required
    body = chunk(b'fmt ', fmt_payload(8, 1))
    body += chunk(b'data', samples)       # padding byte emitted by chunk()
    body += chunk(b'fact', fact_payload())
    with open(os.path.join(directory, 'odd_data_8bit.wav'), 'wb') as f:
        f.write(riff_wave(body))


def write_truncated(directory):
    """Files that must be rejected deterministically without crashing."""
    # 16-bit mono data chunk announcing 8 frames but containing only 3
    good_body = b''.join(layout_chunks('L1', 16, 1))
    full = riff_wave(good_body)
    # 12-byte RIFF header, fmt chunk, 8-byte data chunk header
    data_payload_start = 12 + len(chunk(b'fmt ', fmt_payload(16, 1))) + 8
    cut = data_payload_start + 3 * 2  # three complete frames then EOF
    with open(os.path.join(directory, 'trunc_data_16bit.wav'), 'wb') as f:
        f.write(full[:cut])

    # 24-bit mono, one byte missing from the last frame
    good_body = b''.join(layout_chunks('L1', 24, 1))
    full = riff_wave(good_body)
    with open(os.path.join(directory, 'trunc_data_24bit.wav'), 'wb') as f:
        f.write(full[:-1])

    # truncated fmt chunk: header + id + size, only 6 of 16 bytes
    broken = b'RIFF' + le32(0) + b'WAVE' + b'fmt ' + le32(16) + b'\x00' * 6
    with open(os.path.join(directory, 'trunc_fmt.wav'), 'wb') as f:
        f.write(broken)

    # odd-sized chunk at EOF, no padding byte and no following chunk
    broken = (b'RIFF' + le32(0) + b'WAVE' + b'aub0' + le32(5)
              + b'\x01\x02\x03\x04\x05')
    with open(os.path.join(directory, 'trunc_odd_eof.wav'), 'wb') as f:
        f.write(broken)

    # RIFF header with no WAVE form type at all
    with open(os.path.join(directory, 'trunc_riff_nowave.wav'), 'wb') as f:
        f.write(b'RIFF' + le32(16))

    # does not even start with RIFF
    with open(os.path.join(directory, 'bad_notriff.wav'), 'wb') as f:
        f.write(b'JUNK' + le32(16) + b'WAVE')

    # RIFF but wrong form type
    with open(os.path.join(directory, 'bad_notwave.wav'), 'wb') as f:
        f.write(b'RIFF' + le32(4) + b'AVI ')

    # data size not a multiple of blockalign
    body = chunk(b'fmt ', fmt_payload(16, 1))
    body += b'data' + le32(3) + b'\x00\x00\x00'
    with open(os.path.join(directory, 'bad_blockalign.wav'), 'wb') as f:
        f.write(riff_wave(body))

    # unsupported 12-bit PCM
    body = chunk(b'fmt ', fmt_payload(12, 1))
    body += chunk(b'data', b'\x00\x00')
    with open(os.path.join(directory, 'bad_bits.wav'), 'wb') as f:
        f.write(riff_wave(body))

    # fmt + fact but no data chunk
    body = chunk(b'fmt ', fmt_payload(16, 1)) + chunk(b'fact', fact_payload())
    with open(os.path.join(directory, 'bad_nodata.wav'), 'wb') as f:
        f.write(riff_wave(body))


def main(directory):
    os.makedirs(directory, exist_ok=True)
    layouts = ['L1', 'L2', 'L3', 'L4', 'L5']
    for bits in (8, 16, 24, 32):
        for layout in layouts:
            name = 'pcm%dbit_mono_%s.wav' % (bits, layout)
            write_wav(os.path.join(directory, name), bits, 1, layout)
    for bits in (8, 16, 24, 32):
        name = 'pcm%dbit_stereo_L1.wav' % bits
        write_wav(os.path.join(directory, name), bits, 2, 'L1')
    for bits in (16, 24):
        name = 'pcm%dbit_stereo_L4.wav' % bits
        write_wav(os.path.join(directory, name), bits, 2, 'L4')
    write_hardcoded_odd_pad(os.path.join(directory, 'odd_pad_offsets.wav'))
    write_odd_data_chunk(directory)
    write_truncated(directory)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.stderr.write('usage: %s <output_directory>\n' % sys.argv[0])
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
