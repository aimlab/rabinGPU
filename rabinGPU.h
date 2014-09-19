#ifndef RABINGPU_H
#define RABINGPU_H

#include <cuda_runtime.h>
#include "pkt_reader.h"
#include "rabinCPU.h"

//#define DEBUG 1

#if DEBUG
    #define LOG_MSG printf
#else
    #define LOG_MSG(...)
#endif

#define CUDA_EVENT_TIME_BEGIN       \
    {                               \
        cudaEventCreate(&start);    \
        cudaEventCreate(&stop);     \
        cudaEventRecord(start, 0);  \
    }

#define CUDA_EVENT_TIME_END	         \
    {                                \
        cudaEventRecord(stop, 0);    \
        cudaEventSynchronize(stop);  \
        cudaEventElapsedTime(&executetime, start, stop); \
    }

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err) __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line)
{
    if (cudaSuccess != err) {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
                file, line, (int)err, cudaGetErrorString(err));
        exit(-1);
    }
}

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg) __getLastCudaError (msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file, const int line)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
                file, line, errorMessage, (int)err, cudaGetErrorString(err));
        exit(-1);
    }
}

/* Declarations for rabin_init.cpp */
void indexed_buffer_init(struct pktreader_s *preader);
void indexed_buffer_finalize(struct pktreader_s *preader);

/* Declarations for rabin_launch.cu */
void memcpy_host_to_device(struct pktreader_s *preader, int nstreams, float *times);
void memcpy_device_to_host(struct pktreader_s *preader, int nstreams, float *times);
void launch_kernels(struct pktreader_s *preader, int nstreams, float *times);
float process_with_streams(struct pktreader_s *preader);

/* Declarations for rabin_validation.cpp */
int validate_results(struct pktreader_s *preader, int nstreams);

/* Declaration for rabin_Kepler.cu and rabin_fermi.cu  */
__global__ void rabin_Kepler(uint8_t *in,
                             uint32_t *size_array,
                             uint32_t *offset_array,
                             uint32_t *packet_number_array,
                             uint32_t total_threads,
                             struct rabinpoly_window *rabin_win,
                             uint8_t *rabin_buf,
                             uint64_t *out);
__global__ void rabin_Kepler1(uint8_t *in,
                             uint32_t *size_array,
                             uint32_t *offset_array,
                             uint32_t *packet_number_array,
                             uint32_t total_threads,
                             struct rabinpoly_window *rabin_win,
                             uint8_t *rabin_buf,
                             uint64_t *out);
__global__ void rabin_naive(uint8_t *in,
                            uint32_t *size_array,
                            uint32_t *offset_array,
                            uint32_t total_threads,
                            struct rabinpoly_window *rabin_win,
                            uint8_t *rabin_buf,
                            uint64_t *out);

__global__ void rabin_Fermi(uint8_t *in, uint32_t *size_array, uint32_t *offset_array,
                            uint32_t total_threads, struct rabinpoly_window *rabin_win,
                            uint8_t *rabin_buf, uint64_t *out);

void texture_cache_init(struct indexed_buffer_s *ibuf);
void texture_cache_finalize();

/* Declaration for sort.cu  */
void appr_sort(struct indexed_buffer_s* ibuf, float *times);
void appr_sort_with_streams(struct indexed_buffer_s* ibuf);

__global__ void assign_bucket(uint32_t *d_input,
                              uint32_t length,
                              uint32_t maximum,
                              uint32_t minimum,
                              uint32_t *d_offset,
                              uint32_t *d_bucket_count,
                              uint32_t *d_bucket_index);

__global__ void approximate_sort(uint32_t *d_key,
                                 uint32_t *d_key_sorted,
                                 uint32_t *d_value,
                                 uint32_t *d_value_sorted,
                                 uint32_t *d_packet_number,
                                 uint32_t *d_packet_number_sorted,
                                 uint32_t length,
                                 uint32_t *d_offset,
                                 uint32_t *d_bucket_count,
                                 uint32_t *d_bucket_index);

/* Declaration for remap.cu  */
void remap(struct indexed_buffer_s *ibuf);
void remap_with_streams(struct indexed_buffer_s *ibuf);

#endif // RABINGPU_H