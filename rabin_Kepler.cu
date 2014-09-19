#include <cuda_runtime.h>

#include "comdef.h"
#include "rabinCPU.h"

__device__ inline void fill_buffer(uint8_t *buf, int4 value)
{
    buf[0]  = (uint8_t)value.x;
    buf[1]  = (uint8_t)(value.x >> 8);
    buf[2]  = (uint8_t)(value.x >> 16);
    buf[3]  = (uint8_t)(value.x >> 24);

    buf[4]  = (uint8_t)value.y;
    buf[5]  = (uint8_t)(value.y >> 8);
    buf[6]  = (uint8_t)(value.y >> 16);
    buf[7]  = (uint8_t)(value.y >> 24);

    buf[8]  = (uint8_t)value.z;
    buf[9]  = (uint8_t)(value.z >> 8);
    buf[10] = (uint8_t)(value.z >> 16);
    buf[11] = (uint8_t)(value.z >> 24);

    buf[12] = (uint8_t)value.w;
    buf[13] = (uint8_t)(value.w >> 8);
    buf[14] = (uint8_t)(value.w >> 16);
    buf[15] = (uint8_t)(value.w >> 24);
}

__global__ void rabin_Kepler(uint8_t *in,
                             uint32_t *size_array,
                             uint32_t *offset_array,
                             uint32_t *packet_number_array,
                             uint32_t total_threads,
                             struct rabinpoly_window *rabin_win,
                             uint8_t *rabin_buf,
                             uint64_t *out)
{
    __shared__ uint64_t U[256];
    __shared__ uint64_t T[256];
    __shared__ uint8_t buf[MAX_THREADS_PER_BLOCK * DEFAULT_RABIN_WINDOW];
    //uint8_t buf[DEFAULT_RABIN_WINDOW];

    int thread_idx = threadIdx.x + blockDim.x * blockIdx.x;

    /* putting the U and T arrays in shared memory results in no
     * performance benefit when we do not first optimize the memory
     * access of the input 'unsigned char *in'. However, when the
     * desired optimization is ready, we see a large performance boost
     * by using shared memory for U and T. See the inner 'for' loop.
     */
    for (int i = 0; i < 256 / blockDim.x; i++) {
        U[threadIdx.x + blockDim.x * i] = rabin_win->U[threadIdx.x + blockDim.x * i];
        T[threadIdx.x + blockDim.x * i] = rabin_win->rp.T[threadIdx.x + blockDim.x * i];
    }

    /* index for shared memory buf that can avoid bank conflict */
    int id = threadIdx.x / 32 + (threadIdx.x % 32) * 4;

    /* reset shared memory. IMPORTANT! */
    for (int i = 0; i < DEFAULT_RABIN_WINDOW; i++)
        buf[id + i * blockDim.x] = 0;//buf[i] = 0;

    /* before threads return, we must ensure that the shared memory
     * initialization (such as for arrays U and T) has finished.
     * Or else the arrays may be only partially initialized if the
     * number of total packets is smaller than the dimension of a CTA
     * (MAX_THREADS_PER_BLOCK).
     */
    if (thread_idx >= total_threads) return;

    uint32_t real_size = size_array[thread_idx];
    uint32_t offset = offset_array[thread_idx];
    int4 *input = (int4 *)in + offset;

#ifdef SHARED_MEM_BUF
    uint8_t *rbuf = buf;// + threadIdx.x * DEFAULT_RABIN_WINDOW;
#else
    uint8_t *rbuf = rabin_buf + thread_idx * DEFAULT_RABIN_WINDOW;
#endif

#ifdef SORTING
    uint32_t packet_number = packet_number_array[thread_idx];
    uint64_t *out_idx = out + packet_number * FIXED_FP;
    uint16_t *out_offset = (uint16_t *)(out + MAX_NUMBER_OF_PACKETS * FIXED_FP);
    uint8_t *out_size = (uint8_t *)(out_offset + MAX_NUMBER_OF_PACKETS * FIXED_FP) + packet_number;
    out_offset = out_offset + packet_number * FIXED_FP;
#else
    uint64_t *out_idx = out + thread_idx * FIXED_FP;
    uint16_t *out_offset = (uint16_t *)(out + MAX_NUMBER_OF_PACKETS * FIXED_FP);// + thread_idx * FIXED_FP;
    uint8_t *out_size = (uint8_t *)(out_offset + MAX_NUMBER_OF_PACKETS * FIXED_FP) + thread_idx;
    out_offset = out_offset + thread_idx * FIXED_FP;
#endif

    /* parameters for rabin fingerprint */
    uint64_t fingerprint = 0;
    int shift = rabin_win->rp.shift;
    int bufpos = -1;
    int index = 0;

    uint64_t p;
    uint8_t old_char;
    uint8_t new_char;
    int4 result;

    uint32_t align_size = real_size;

    /* recalculate the aligned size, so we do not need to record the
     * aligned size using an array, avoiding host-to-device data transfer.
     */
    if (real_size % ELEM_SZ != 0) {
        align_size = real_size + (ELEM_SZ - real_size % ELEM_SZ);
    }

    __syncthreads();

    uint8_t tmp[ELEM_SZ];

    //uint64_t fp[FIXED_FP];
    //uint16_t fp_offset[FIXED_FP];

    for (uint16_t i = 0; i < DEFAULT_RABIN_WINDOW / ELEM_SZ; i++) {
//#ifdef SORTING
//        result = *(input + packet_number + i * total_threads);
//#else
        result = *(input + i);//*(input + thread_idx + i * total_threads);
//#endif
        fill_buffer(tmp, result);

        for (uint16_t j = 0; j < ELEM_SZ; j++) {
            if (++bufpos >= DEFAULT_RABIN_WINDOW)
                bufpos = 0;
#ifdef SHARED_MEM_BUF
            old_char = rbuf[id + bufpos * blockDim.x];
            new_char = tmp[j];
            rbuf[id + bufpos * blockDim.x] = new_char;
            // old_char = rbuf[bufpos];
            // new_char = tmp[j];
            // rbuf[bufpos] = new_char;

            p = fingerprint ^ U[old_char];//__ldg(&rabin_win->U[old_char]);//
            fingerprint = ((p << 8) | new_char) ^ T[p >> shift];//__ldg(&rabin_win->rp.T[p >> shift]);//
#else
            old_char = rbuf[bufpos];
            new_char = tmp[j];
            rbuf[bufpos] = new_char;

            p = fingerprint ^ rabin_win->U[old_char];
            fingerprint = ((p << 8) | new_char) ^ rabin_win->rp.T[p >> shift];
#endif
        }
    }

    if (!(fingerprint & MASK) && (index < FIXED_FP)) {
        out_idx[index] = fingerprint;
        out_offset[index] = 0;
        //fp[index] = fingerprint;
        //fp_offset[index] = 0;
        index++;
    }

    __syncthreads();

    //uint32_t count = DEFAULT_RABIN_WINDOW;

    //int s;
    //int bytes_left = real_size - count - DEFAULT_RABIN_WINDOW;
    //bytes_left = (bytes_left > 0) ? bytes_left : 0;
    //int remain = bytes_left / ELEM_SZ;
    //int last = bytes_left % ELEM_SZ;

    for (uint16_t i = DEFAULT_RABIN_WINDOW / ELEM_SZ; i < align_size / ELEM_SZ; i++) {
        if (++bufpos >= DEFAULT_RABIN_WINDOW)
            bufpos = 0;

        //if (count >= real_size - DEFAULT_RABIN_WINDOW)
        //    break;

        /* rbuf has a significant impact on performance as
         * compared to the naive version where rbuf is located at global
         * memory. The input array is byte-oriented, so optimizing read
         * access to global memory (4, 8, or 16 bytes) would achieve better
         * performance.
         */
//#ifdef SORTING
//        result = *(input + packet_number + i * total_threads);
//#else
        result = *(input + i);//*(input + thread_idx + i * total_threads);
//#endif
        fill_buffer(tmp, result);

        //s = (remain > 0) ? ELEM_SZ : last;
        //remain--;

        for (uint16_t j = 0; j < ELEM_SZ; j++, bufpos++) {
#ifdef SHARED_MEM_BUF
            old_char = rbuf[id + bufpos * blockDim.x];
            new_char = tmp[j];
            rbuf[id + bufpos * blockDim.x] = new_char;
            // old_char = rbuf[bufpos];
            // new_char = tmp[j];
            // rbuf[bufpos] = new_char;

            p = fingerprint ^ U[old_char];//__ldg(&rabin_win->U[old_char]);//
            fingerprint = ((p << 8) | new_char) ^ T[p >> shift];//__ldg(&rabin_win->rp.T[p >> shift]);//
#else
            old_char = rbuf[bufpos];
            new_char = tmp[j];
            rbuf[bufpos] = new_char;

            p = fingerprint ^ rabin_win->U[old_char];
            fingerprint = ((p << 8) | new_char) ^ rabin_win->rp.T[p >> shift];
#endif

            if (!(fingerprint & MASK) && (index < FIXED_FP)) {
                out_idx[index] = fingerprint;
                out_offset[index] = i << 8 | j;
                //fp[index] = fingerprint;
                //fp_offset[index] = (i << 8) | j;
                index++;
            }
        }
        //count = count + s;
        bufpos--;
    }

    __syncthreads();

    // for (int i = 0; i < index; i++) {
    //     out_idx[i] = fp[i];
    //     out_offset[i] = fp_offset[i];
    // }

    *out_size = index;
}

__global__ void rabin_Kepler1(uint8_t *in,
                             uint32_t *size_array,
                             uint32_t *offset_array,
                             uint32_t *packet_number_array,
                             uint32_t total_threads,
                             struct rabinpoly_window *rabin_win,
                             uint8_t *rabin_buf,
                             uint64_t *out)
{
    __shared__ uint64_t U[256];
    __shared__ uint64_t T[256];
    __shared__ uint8_t buf[MAX_THREADS_PER_BLOCK * DEFAULT_RABIN_WINDOW];
    //uint8_t buf[DEFAULT_RABIN_WINDOW];

    int thread_idx = threadIdx.x + blockDim.x * blockIdx.x;

    /* putting the U and T arrays in shared memory results in no
     * performance benefit when we do not first optimize the memory
     * access of the input 'unsigned char *in'. However, when the
     * desired optimization is ready, we see a large performance boost
     * by using shared memory for U and T. See the inner 'for' loop.
     */
    for (int i = 0; i < 256 / blockDim.x; i++) {
        U[threadIdx.x + blockDim.x * i] = rabin_win->U[threadIdx.x + blockDim.x * i];
        T[threadIdx.x + blockDim.x * i] = rabin_win->rp.T[threadIdx.x + blockDim.x * i];
    }

    /* index for shared memory buf that can avoid bank conflict */
    int id = threadIdx.x / 32 + (threadIdx.x % 32) * 4;

    /* reset shared memory. IMPORTANT! */
    for (int i = 0; i < DEFAULT_RABIN_WINDOW; i++)
        buf[id + i * blockDim.x] = 0;//buf[i] = 0;

    //return;

    /* before threads return, we must ensure that the shared memory
     * initialization (such as for arrays U and T) has finished.
     * Or else the arrays may be only partially initialized if the
     * number of total packets is smaller than the dimension of a CTA
     * (MAX_THREADS_PER_BLOCK).
     */
    if (thread_idx >= total_threads) return;

    uint32_t real_size = size_array[thread_idx];
    uint32_t offset = offset_array[thread_idx];
    int4 *input = (int4 *)in + offset;

#ifdef SHARED_MEM_BUF
    uint8_t *rbuf = buf;// + threadIdx.x * DEFAULT_RABIN_WINDOW;
#else
    uint8_t *rbuf = rabin_buf + thread_idx * DEFAULT_RABIN_WINDOW;
#endif

//#ifdef SORTING
    uint32_t packet_number = packet_number_array[thread_idx];
    uint64_t *out_idx = out + packet_number * FIXED_FP;
    uint16_t *out_offset = (uint16_t *)(out + MAX_NUMBER_OF_PACKETS * FIXED_FP);
    uint8_t *out_size = (uint8_t *)(out_offset + MAX_NUMBER_OF_PACKETS * FIXED_FP) + packet_number;
    out_offset = out_offset + packet_number * FIXED_FP;
//#else
    // uint64_t *out_idx = out + thread_idx * FIXED_FP;
    // uint16_t *out_offset = (uint16_t *)(out + MAX_NUMBER_OF_PACKETS * FIXED_FP);// + thread_idx * FIXED_FP;
    // uint8_t *out_size = (uint8_t *)(out_offset + MAX_NUMBER_OF_PACKETS * FIXED_FP) + thread_idx;
    // out_offset = out_offset + thread_idx * FIXED_FP;
//#endif

    /* parameters for rabin fingerprint */
    uint64_t fingerprint = 0;
    int shift = rabin_win->rp.shift;
    int bufpos = -1;
    int index = 0;

    uint64_t p;
    uint8_t old_char;
    uint8_t new_char;
    int4 result;

    uint32_t align_size = real_size;

    /* recalculate the aligned size, so we do not need to record the
     * aligned size using an array, avoiding host-to-device data transfer.
     */
    if (real_size % ELEM_SZ != 0) {
        align_size = real_size + (ELEM_SZ - real_size % ELEM_SZ);
    }

    __syncthreads();

    uint8_t tmp[ELEM_SZ];

    uint32_t count = 0;//DEFAULT_RABIN_WINDOW;

    int s;
    int bytes_left = real_size - count;
    bytes_left = (bytes_left > 0) ? bytes_left : 0;
    int remain = bytes_left / ELEM_SZ;
    int last = bytes_left % ELEM_SZ;

    int n;

    for (uint16_t i = 0; i < align_size / ELEM_SZ; i++) {
        if (++bufpos >= DEFAULT_RABIN_WINDOW)
            bufpos = 0;

        if (count >= real_size - DEFAULT_RABIN_WINDOW)
            break;

        /* rbuf has a significant impact on performance as
         * compared to the naive version where rbuf is located at global
         * memory. The input array is byte-oriented, so optimizing read
         * access to global memory (4, 8, or 16 bytes) would achieve better
         * performance.
         */
        result = *(input + i);
        fill_buffer(tmp, result);

        s = (remain > 0) ? ELEM_SZ : last;
        remain--;

        if (remain > 0)
            n = ELEM_SZ;
        else
            n = s;

        for (uint16_t j = 0; j < n; j++, bufpos++) {
            old_char = rbuf[id + bufpos * blockDim.x];
            new_char = tmp[j];
            rbuf[id + bufpos * blockDim.x] = new_char;

            p = fingerprint ^ U[old_char];
            fingerprint = ((p << 8) | new_char) ^ T[p >> shift];

            if (++count >= DEFAULT_RABIN_WINDOW) {
                if (!(fingerprint & MASK) && (index < FIXED_FP)) {
                    out_idx[index] = fingerprint;
                    out_offset[index] = i << 8 | j;
                    index++;
                }
            }
        }
        //count = count + s;
        bufpos--;
    }

    __syncthreads();
    *out_size = index;
}

