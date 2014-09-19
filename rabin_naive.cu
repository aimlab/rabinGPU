#include <cuda_runtime.h>

#include "comdef.h"
#include "rabinCPU.h"


__global__ void rabin_naive(uint8_t *in,
                            uint32_t *size_array,
                            uint32_t *offset_array,
                            uint32_t total_threads,
                            struct rabinpoly_window *rabin_win,
                            uint8_t *rabin_buf,
                            uint64_t *out)
{
    int thread_idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (thread_idx >= total_threads)
        return;

    uint32_t offset = offset_array[thread_idx];
    uint32_t real_size = size_array[thread_idx];

    uint8_t *input = in + offset;
    uint8_t *rbuf = rabin_buf + thread_idx * DEFAULT_RABIN_WINDOW;

    uint64_t *out_idx = out + thread_idx * FIXED_FP;
    uint16_t *out_offset = (uint16_t *)(out + MAX_NUMBER_OF_PACKETS * FIXED_FP);// + thread_idx * FIXED_FP;
    uint8_t *out_size = (uint8_t *)(out_offset + MAX_NUMBER_OF_PACKETS * FIXED_FP) + thread_idx;
	out_offset = out_offset + thread_idx * FIXED_FP;

    /* parameters for rabin fingerprint */
    uint64_t fingerprint = 0;
    int shift = rabin_win->rp.shift;
    int winsize = rabin_win->size;
    int bufpos = -1;
    int index = 0;

    uint64_t mask = (uint64_t)((1UL << DEFAULT_RABIN_REPR_BITS) - 1);

    uint64_t p;
    uint8_t old_char;
    uint8_t new_char;

    __syncthreads();

    for (int i = 0; i < real_size; i++) {
        if (++bufpos >= winsize)
            bufpos = 0;

        old_char = rbuf[bufpos];
        new_char = input[i];
        rbuf[bufpos] = new_char;

        p = fingerprint ^ rabin_win->U[old_char];
        fingerprint = ((p << 8) | new_char) ^ rabin_win->rp.T[p >> shift];

        if ( (i >= DEFAULT_RABIN_WINDOW - 1) && !(fingerprint & mask) && (index < FIXED_FP)) {
            out_idx[index] = fingerprint;
            out_offset[index] = i;
            index++;
        }
    }

    __syncthreads();
    *out_size = index;
}