/*
  Copyright (C) 2014 Paul Brossier <piem@aubio.org>

  This file is part of aubio

  aubio is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  aubio is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with aubio.  If not, see <http://www.gnu.org/licenses/>.

*/

#include "aubio_priv.h"

#ifdef HAVE_WAVREAD

#include <limits.h>

#include "fvec.h"
#include "fmat.h"
#include "ioutils.h"
#include "source_wavread.h"

#define AUBIO_WAVREAD_BUFSIZE 1024

/*
 * Reference: "Multimedia Programming Interface and Data
 * Specifications 1.0" (Microsoft/IBM, 1991), RIFF / WAVE section.
 *
 * The RIFF rules implemented here are:
 *
 *  - a RIFF file starts with a "RIFF" FourCC, a 32-bit little-endian
 *    chunk size and a form type ("WAVE"), followed by a sequence of
 *    sub-chunks;
 *  - each sub-chunk starts with a FourCC and a 32-bit little-endian
 *    size giving the number of *data* bytes that follow;
 *  - sub-chunks may appear in any order; unknown sub-chunks (e.g.
 *    "fact", "LIST", "JUNK", "PAD ") are skipped;
 *  - a sub-chunk with an odd data size is followed by a single
 *    padding byte so that the next sub-chunk starts on a word
 *    (16-bit / even) boundary. The padding byte is not counted in
 *    the sub-chunk size;
 *  - PCM WAVE (audio format 1) stores 8-bit samples as unsigned
 *    bytes and 16-, 24- and 32-bit samples as little-endian two's
 *    complement signed integers.
 */

struct _aubio_source_wavread_t {
  uint_t hop_size;
  uint_t samplerate;
  uint_t channels;

  // some data about the file
  char_t *path;
  uint_t input_samplerate;
  uint_t input_channels;

  // internal stuff
  FILE *fid;

  uint_t read_samples;
  uint_t blockalign;
  uint_t bitspersample;
  uint_t read_index;
  uint_t eof;

  uint_t duration;

  // number of complete data frames announced by the "data" chunk
  uint_t data_frames;
  // number of data frames already handed out
  uint_t frames_done;

  size_t seek_start;

  unsigned char *short_output;
  fmat_t *output;
};

/* assemble `length` little-endian bytes (1 <= length <= 4) into an
 * unsigned 32-bit value; unsigned arithmetic keeps every shift free
 * of undefined behaviour */
static unsigned int read_little_endian (const unsigned char *buf,
    unsigned int length)
{
  unsigned long long ret = 0;
  unsigned int i;
  for (i = 0; i < length; i++) {
    ret += (unsigned long long)buf[i] << (i * 8);
  }
  return (unsigned int)ret;
}

/* read exactly `size` bytes, failing deterministically on a short read */
static uint_t read_exact(aubio_source_wavread_t * s, void *buf, size_t size)
{
  size_t got = fread(buf, 1, size, s->fid);
  if (got != size) {
    AUBIO_ERR("source_wavread: truncated file %s (wanted %llu bytes, got %llu)\n",
        s->path, (unsigned long long)size, (unsigned long long)got);
    return AUBIO_FAIL;
  }
  return AUBIO_OK;
}

/* RIFF sub-chunks occupy their declared data size plus one padding
 * byte when that size is odd */
static unsigned long long riff_padded_size(unsigned long long size)
{
  return size + (size & 1);
}

/* skip a sub-chunk of `size` data bytes, honouring the RIFF word
 * alignment padding byte */
static uint_t skip_chunk(aubio_source_wavread_t * s, unsigned long long size)
{
  unsigned long long padded = riff_padded_size(size);
  // fseek SEEK_CUR takes a long offset: reject anything that cannot
  // be represented rather than silently truncating
  if (padded > (unsigned long long)LONG_MAX) {
    AUBIO_ERR("source_wavread: chunk too large in %s\n", s->path);
    return AUBIO_FAIL;
  }
  if (fseek(s->fid, (long)padded, SEEK_CUR) != 0) {
    AUBIO_STRERR("source_wavread: could not seek past chunk in %s (%s)\n",
        s->path, errorstr);
    return AUBIO_FAIL;
  }
  return AUBIO_OK;
}

/* decode one little-endian PCM sample of `bytes` bytes into a float
 * in [-1, 1]:
 *  - 8-bit PCM is unsigned, centred on 128;
 *  - 16/24/32-bit PCM is signed two's complement divided by 2^(n-1).
 */
static smpl_t decode_pcm_sample(const unsigned char *p, uint_t bytes)
{
  unsigned long long unsigned_val = 0;
  unsigned long long sign_bit;
  long long signed_val;
  unsigned int b;
  for (b = 0; b < bytes; b++) {
    unsigned_val |= (unsigned long long)p[b] << (8 * b);
  }
  if (bytes == 1) {
    // unsigned 8-bit: map [0, 255] onto [-1, 127/128]
    return (smpl_t)(((long long)unsigned_val - 128) / 128.0);
  }
  sign_bit = 1ULL << (8 * bytes - 1);
  if (unsigned_val & sign_bit) {
    signed_val = (long long)unsigned_val - (long long)(sign_bit << 1);
  } else {
    signed_val = (long long)unsigned_val;
  }
  return (smpl_t)((double)signed_val / (double)sign_bit);
}

aubio_source_wavread_t * new_aubio_source_wavread(const char_t * path, uint_t samplerate, uint_t hop_size) {
  aubio_source_wavread_t * s = AUBIO_NEW(aubio_source_wavread_t);
  unsigned char id[4];
  unsigned char buf[16];
  unsigned int riff_size, chunk_size;
  unsigned int format, channels = 0, sr = 0, byterate = 0;
  unsigned int blockalign = 0, bitspersample = 0, fmt_size = 0;
  unsigned int data_size = 0;
  unsigned long long file_size, data_offset = 0;
  uint_t have_fmt = 0, have_data = 0;

  if (path == NULL) {
    AUBIO_ERR("source_wavread: Aborted opening null path\n");
    goto beach;
  }
  if ((sint_t)samplerate < 0) {
    AUBIO_ERR("source_wavread: Can not open %s with samplerate %d\n", path, samplerate);
    goto beach;
  }
  if ((sint_t)hop_size <= 0) {
    AUBIO_ERR("source_wavread: Can not open %s with hop_size %d\n", path, hop_size);
    goto beach;
  }

  s->path = AUBIO_ARRAY(char_t, strnlen(path, PATH_MAX) + 1);
  strncpy(s->path, path, strnlen(path, PATH_MAX) + 1);

  s->samplerate = samplerate;
  s->hop_size = hop_size;

  s->fid = fopen((const char *)path, "rb");
  if (!s->fid) {
    AUBIO_STRERR("source_wavread: Failed opening %s (%s)\n", s->path, errorstr);
    goto beach;
  }

  // RIFF chunk identifier
  if (read_exact(s, id, 4) != AUBIO_OK) goto beach;
  if ( memcmp((const char *)id, "RIFF", 4) != 0 ) {
    AUBIO_ERR("source_wavread: Failed opening %s (could not find RIFF header)\n", s->path);
    goto beach;
  }

  // RIFF chunk size: size of the whole RIFF chunk minus 8 bytes
  if (read_exact(s, buf, 4) != AUBIO_OK) goto beach;
  riff_size = read_little_endian(buf, 4);
  (void)riff_size; // not trusted: some writers fill it incorrectly

  // RIFF form type
  if (read_exact(s, id, 4) != AUBIO_OK) goto beach;
  if ( memcmp((const char *)id, "WAVE", 4) != 0 ) {
    AUBIO_ERR("source_wavread: Failed opening %s (wrong format in RIFF header)\n", s->path);
    goto beach;
  }

  // walk the WAVE sub-chunks. RIFF does not impose any ordering on
  // them, so "fmt " need not be first and "data" need not be last.
  while (1) {
    size_t got = fread(id, 1, 4, s->fid);
    if (got == 0) {
      // clean end of file once the data chunk was found
      if (feof(s->fid) && have_data) break;
      AUBIO_ERR("source_wavread: truncated file %s (missing chunk identifier)\n",
          s->path);
      goto beach;
    } else if (got != 4) {
      AUBIO_ERR("source_wavread: truncated file %s (partial chunk identifier)\n",
          s->path);
      goto beach;
    }

    if (read_exact(s, buf, 4) != AUBIO_OK) goto beach;
    chunk_size = read_little_endian(buf, 4);

    if (memcmp((const char *)id, "fmt ", 4) == 0) {
      if (have_fmt) {
        AUBIO_ERR("source_wavread: duplicated 'fmt ' chunk in %s\n", s->path);
        goto beach;
      }
      if (chunk_size < 16) {
        AUBIO_ERR("source_wavread: 'fmt ' chunk too small (%u) in %s\n",
            chunk_size, s->path);
        goto beach;
      }
      fmt_size = chunk_size;
      // read the 16 bytes of the canonical PCM WAVEFORMAT structure
      if (read_exact(s, buf, 16) != AUBIO_OK) goto beach;
      // skip any extension bytes (fmt size 18, 40, ...) and the
      // RIFF padding byte, if any
      if (skip_chunk(s, (unsigned long long)fmt_size - 16) != AUBIO_OK) {
        goto beach;
      }

      format       = read_little_endian(buf + 0, 2);
      channels     = read_little_endian(buf + 2, 2);
      sr           = read_little_endian(buf + 4, 4);
      byterate     = read_little_endian(buf + 8, 4);
      blockalign   = read_little_endian(buf + 12, 2);
      bitspersample = read_little_endian(buf + 14, 2);

      if ( format != 1 ) {
        AUBIO_ERR("source_wavread: Failed opening %s (not encoded with PCM)\n", s->path);
        goto beach;
      }

      if ( channels == 0 ) {
        AUBIO_ERR("source_wavread: Failed opening %s (number of channels can not be 0)\n", s->path);
        goto beach;
      }

      if ( (sint_t)sr <= 0 ) {
        AUBIO_ERR("source_wavread: Failed opening %s (samplerate can not be <= 0)\n", s->path);
        goto beach;
      }

      if ( byterate == 0 ) {
        AUBIO_ERR("source_wavread: Failed opening %s (byterate can not be 0)\n", s->path);
        goto beach;
      }

      if ( bitspersample != 8 && bitspersample != 16
          && bitspersample != 24 && bitspersample != 32 ) {
        AUBIO_ERR("source_wavread: can not process %d-bit file %s\n",
            bitspersample, s->path);
        goto beach;
      }

      if ( (unsigned long long)byterate * 8ULL
          != (unsigned long long)sr * channels * bitspersample ) {
        AUBIO_ERR("source_wavread: Failed opening %s (wrong byterate)\n", s->path);
        goto beach;
      }

      if ( blockalign * 8 != channels * bitspersample ) {
        AUBIO_ERR("source_wavread: Failed opening %s (wrong blockalign)\n", s->path);
        goto beach;
      }

      have_fmt = 1;
    } else if (memcmp((const char *)id, "data", 4) == 0 && !have_data) {
      if (!have_fmt) {
        AUBIO_ERR("source_wavread: 'data' chunk found before 'fmt ' in %s\n",
            s->path);
        goto beach;
      }
      if (chunk_size % blockalign != 0) {
        AUBIO_ERR("source_wavread: data chunk size %u is not a multiple of blockalign %u in %s\n",
            chunk_size, blockalign, s->path);
        goto beach;
      }
      long where = ftell(s->fid);
      if (where < 0) {
        AUBIO_STRERR("source_wavread: could not tell position in %s (%s)\n",
            s->path, errorstr);
        goto beach;
      }
      data_offset = (unsigned long long)where;
      data_size = chunk_size;

      // deterministic truncation check: the file must actually hold
      // every byte the data chunk claims to contain
      if (fseek(s->fid, 0, SEEK_END) != 0) {
        AUBIO_STRERR("source_wavread: could not seek to end of %s (%s)\n",
            s->path, errorstr);
        goto beach;
      }
      long end_pos = ftell(s->fid);
      if (end_pos < 0) {
        AUBIO_STRERR("source_wavread: could not tell size of %s (%s)\n",
            s->path, errorstr);
        goto beach;
      }
      file_size = (unsigned long long)end_pos;
      if (file_size < data_offset + (unsigned long long)data_size) {
        AUBIO_ERR("source_wavread: truncated data chunk in %s (file has %llu bytes, needs %llu)\n",
            s->path, file_size,
            data_offset + (unsigned long long)data_size);
        goto beach;
      }
      have_data = 1;
      // keep scanning: legal RIFF files may carry chunks after data
      if (fseek(s->fid, (long)data_offset, SEEK_SET) != 0) {
        AUBIO_STRERR("source_wavread: could not seek back in %s (%s)\n",
            s->path, errorstr);
        goto beach;
      }
      if (skip_chunk(s, data_size) != AUBIO_OK) goto beach;
    } else {
      // unknown chunk ("fact", "LIST", "JUNK", "PAD ", ...) or a
      // second data chunk: skip data + RIFF padding byte
      if (skip_chunk(s, chunk_size) != AUBIO_OK) goto beach;
    }
  }

  if (!have_fmt || !have_data) {
    AUBIO_ERR("source_wavread: no %s chunk found in %s\n",
        have_fmt ? "data" : "fmt ", s->path);
    goto beach;
  }

  s->input_samplerate = sr;
  s->input_channels = channels;

  if (samplerate == 0) {
    s->samplerate = s->input_samplerate;
  } else if (samplerate != s->input_samplerate) {
    AUBIO_ERR("source_wavread: can not resample %s from %d to %dHz\n",
        s->path, s->input_samplerate, samplerate);
    goto beach;
  }

  // position the stream on the first audio sample
  if (data_offset > (unsigned long long)LONG_MAX ||
      fseek(s->fid, (long)data_offset, SEEK_SET) != 0) {
    AUBIO_STRERR("source_wavread: could not seek to data in %s (%s)\n",
        s->path, errorstr);
    goto beach;
  }
  s->seek_start = (size_t)data_offset;

  s->output = new_fmat(s->input_channels, AUBIO_WAVREAD_BUFSIZE);
  s->blockalign= blockalign;
  s->bitspersample = bitspersample;

  s->data_frames = data_size / blockalign;
  s->duration = s->data_frames;

  s->short_output = (unsigned char *)calloc(s->blockalign, AUBIO_WAVREAD_BUFSIZE);
  if (!s->short_output) {
    AUBIO_ERR("source_wavread: failed allocating read buffer for %s\n", s->path);
    goto beach;
  }
  s->read_index = 0;
  s->read_samples = 0;
  s->frames_done = 0;
  s->eof = 0;

  return s;

beach:
  //AUBIO_ERR("source_wavread: can not read %s at samplerate %dHz with a hop_size of %d\n",
  //    s->path, s->samplerate, s->hop_size);
  del_aubio_source_wavread(s);
  return NULL;
}

void aubio_source_wavread_readframe(aubio_source_wavread_t *s, uint_t *wavread_read);

void aubio_source_wavread_readframe(aubio_source_wavread_t *s, uint_t *wavread_read) {
  unsigned char *short_ptr = s->short_output;
  size_t read;
  uint_t i, j;
  uint_t bytes_per_sample = s->bitspersample / 8;
  uint_t frames_remaining = s->data_frames - s->frames_done;
  uint_t frames_to_read = AUBIO_WAVREAD_BUFSIZE;

  // never read past the declared data chunk, even if the file was
  // modified between open and read
  if (frames_to_read > frames_remaining) {
    frames_to_read = frames_remaining;
  }

  read = fread(short_ptr, s->blockalign, frames_to_read, s->fid);

  for (j = 0; j < read; j++) {
    for (i = 0; i < s->input_channels; i++) {
      s->output->data[i][j] = decode_pcm_sample(short_ptr, bytes_per_sample);
      short_ptr += bytes_per_sample;
    }
  }

  s->frames_done += (uint_t)read;
  *wavread_read = (uint_t)read;

  // a short item read means the data chunk was truncated: flag EOF so
  // the callers stop deterministically instead of spinning. A full read
  // that simply exhausts the data is not an error condition here; the
  // next readframe call returns 0 frames and the callers handle that.
  if (read < frames_to_read) {
    s->eof = 1;
  }
}

void aubio_source_wavread_do(aubio_source_wavread_t * s, fvec_t * read_data, uint_t * read){
  uint_t i, j;
  uint_t end = 0;
  uint_t total_wrote = 0;
  uint_t length = aubio_source_validate_input_length("source_wavread", s->path,
      s->hop_size, read_data->length);
  if (s->fid == NULL) {
    AUBIO_ERR("source_wavread: could not read from %s (file not opened)\n",
        s->path);
    return;
  }
  while (total_wrote < length) {
    end = MIN(s->read_samples - s->read_index, length - total_wrote);
    for (i = 0; i < end; i++) {
      read_data->data[i + total_wrote] = 0;
      for (j = 0; j < s->input_channels; j++ ) {
        read_data->data[i + total_wrote] += s->output->data[j][i + s->read_index];
      }
      read_data->data[i + total_wrote] /= (smpl_t)(s->input_channels);
    }
    total_wrote += end;
    if (total_wrote < length) {
      uint_t wavread_read = 0;
      aubio_source_wavread_readframe(s, &wavread_read);
      s->read_samples = wavread_read;
      s->read_index = 0;
      if (s->eof || wavread_read == 0) {
        break;
      }
    } else {
      s->read_index += end;
    }
  }

  aubio_source_pad_output (read_data, total_wrote);

  *read = total_wrote;
}

void aubio_source_wavread_do_multi(aubio_source_wavread_t * s, fmat_t * read_data, uint_t * read){
  uint_t i,j;
  uint_t end = 0;
  uint_t total_wrote = 0;
  uint_t length = aubio_source_validate_input_length("source_wavread", s->path,
      s->hop_size, read_data->length);
  uint_t channels = aubio_source_validate_input_channels("source_wavread",
      s->path, s->input_channels, read_data->height);
  if (s->fid == NULL) {
    AUBIO_ERR("source_wavread: could not read from %s (file not opened)\n",
        s->path);
    return;
  }
  while (total_wrote < length) {
    end = MIN(s->read_samples - s->read_index, length - total_wrote);
    for (j = 0; j < channels; j++) {
      for (i = 0; i < end; i++) {
        read_data->data[j][i + total_wrote] = s->output->data[j][i + s->read_index];
      }
    }
    total_wrote += end;
    if (total_wrote < length) {
      uint_t wavread_read = 0;
      aubio_source_wavread_readframe(s, &wavread_read);
      s->read_samples = wavread_read;
      s->read_index = 0;
      if (s->eof || wavread_read == 0) {
        break;
      }
    } else {
      s->read_index += end;
    }
  }

  aubio_source_pad_multi_output(read_data, s->input_channels, total_wrote);

  *read = total_wrote;
}

uint_t aubio_source_wavread_get_samplerate(aubio_source_wavread_t * s) {
  return s->samplerate;
}

uint_t aubio_source_wavread_get_channels(aubio_source_wavread_t * s) {
  return s->input_channels;
}

uint_t aubio_source_wavread_seek (aubio_source_wavread_t * s, uint_t pos) {
  uint_t ret = 0;
  if (s->fid == NULL) {
    AUBIO_ERR("source_wavread: could not seek %s (file not opened?)\n", s->path, pos);
    return AUBIO_FAIL;
  }
  if ((sint_t)pos < 0) {
    AUBIO_ERR("source_wavread: could not seek %s at %d (seeking position should be >= 0)\n", s->path, pos);
    return AUBIO_FAIL;
  }
  if (pos > s->data_frames) {
    AUBIO_ERR("source_wavread: could not seek %s at %d (file only holds %d frames)\n",
        s->path, pos, s->data_frames);
    return AUBIO_FAIL;
  }
  ret = fseek(s->fid, s->seek_start + (size_t)pos * s->blockalign, SEEK_SET);
  if (ret != 0) {
    AUBIO_STRERR("source_wavread: could not seek %s at %d (%s)\n", s->path, pos, errorstr);
    return AUBIO_FAIL;
  }
  // reset some values
  s->eof = 0;
  s->read_index = 0;
  s->read_samples = 0;
  s->frames_done = pos;
  return AUBIO_OK;
}

uint_t aubio_source_wavread_get_duration (const aubio_source_wavread_t * s) {
  if (s && s->duration) {
    return s->duration;
  }
  return 0;
}

uint_t aubio_source_wavread_close (aubio_source_wavread_t * s) {
  if (s->fid == NULL) {
    return AUBIO_OK;
  }
  if (fclose(s->fid)) {
    AUBIO_STRERR("source_wavread: could not close %s (%s)\n", s->path, errorstr);
    return AUBIO_FAIL;
  }
  s->fid = NULL;
  return AUBIO_OK;
}

void del_aubio_source_wavread(aubio_source_wavread_t * s) {
  AUBIO_ASSERT(s);
  aubio_source_wavread_close(s);
  if (s->short_output) AUBIO_FREE(s->short_output);
  if (s->output) del_fmat(s->output);
  if (s->path) AUBIO_FREE(s->path);
  AUBIO_FREE(s);
}

#endif /* HAVE_WAVREAD */
