#ifndef COMDEF_H
#define COMDEF_H

//#define NDEBUG

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define TRUE   1   /* Boolean true value. */
#define FALSE  0   /* Boolean false value. */

#ifdef NULL
#undef NULL
#endif

#define NULL 0

/* DO NOT CHANGE THIS VALUE, optimized for shared memory access */
#define MAX_THREADS_PER_BLOCK 128

#define ONE_K   (1UL << 10)
#define ONE_MEG (1UL << 20)

/* SHOULD be divisible by ELEM_SZ */
#define DEFAULT_RABIN_WINDOW 32

/* 6 yields on average 21.6 fingerprints, 7 - 10.8, 8 - 5.4, and 9 yields 2.7,
 * if 0, all (MTU+1-DEFAULT_RABIN_WINDOW) fingerprints will be considered */
#define DEFAULT_RABIN_REPR_BITS 8

/* FIXME: MAX_NUMBER_OF_PACKETS and RABINPOLY_BUFFER
 * should be dynamically determined. Since we use the shared memory of GPU,
 * this buffer will not be used.
 */
#define MAX_NUMBER_OF_PACKETS (1024 * 16)
#define RABINPOLY_BUFFER (DEFAULT_RABIN_WINDOW * MAX_NUMBER_OF_PACKETS)

#define MAX_PINNED_MEM_RABINWIN (sizeof(struct rabinpoly_window))
#define MAX_PINNED_MEM_PAYLOAD  (MAX_NUMBER_OF_PACKETS * DEFAULT_MTU)
//#define MAX_PINNED_MEM_METADATA (512 * ONE_K)
#define MAX_PINNED_MEM_METADATA (4 * sizeof(uint32_t) * MAX_NUMBER_OF_PACKETS)

#define FIXED_FP 12

/* 8 = sizeof(uint64_t) for fingerprint, 2 = sizeof(short) for offset,
 * 1 = sizeof(char) for the number of fingerprints
 */
#define MAX_PINNED_MEM_OUTSIZE ((8 * MAX_NUMBER_OF_PACKETS * FIXED_FP) + (2 * MAX_NUMBER_OF_PACKETS * FIXED_FP) + (1 * MAX_NUMBER_OF_PACKETS))

#define MAX_PINNED_MEM_INSIZE (MAX_PINNED_MEM_PAYLOAD + MAX_PINNED_MEM_METADATA)

#define ELEM_SZ (sizeof(int4))

#define NSTREAMS 4

#define DEFAULT_MTU 1500

#define MASK ((1UL << DEFAULT_RABIN_REPR_BITS) - 1)

#define NUM_BUCKETS 512

#endif  /* COMDEF_H */
