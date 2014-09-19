#include <cuda_runtime.h>
#include "pkt_reader.h"
#include "rabinGPU.h"

/**
 * The following three functions are mainly used for testing
 * in order to measure the execution times of host-to-device,
 * device-to-host data transfer, and kernel invocations.
 */
void memcpy_host_to_device(struct pktreader_s *preader, int nstreams, float *times)
{
    cudaEvent_t start, stop;
    float executetime;

    struct indexed_buffer_s *ibuf;
    ibuf = &(preader->ibufs[nstreams]);

#ifdef NAIVE
    uint32_t size = ibuf->total_real_size;
#else
    uint32_t size = ibuf->total_aligned_size;
#endif

    CUDA_EVENT_TIME_BEGIN
    checkCudaErrors( cudaMemcpy(ibuf->d_input,
                                ibuf->h_input,
                                MAX_PINNED_MEM_METADATA + size,
                                cudaMemcpyHostToDevice) );
    CUDA_EVENT_TIME_END
    times[4] = executetime;
}

void memcpy_device_to_host(struct pktreader_s *preader, int nstreams, float *times)
{
    cudaEvent_t start, stop;
    float executetime;

    struct indexed_buffer_s *ibuf;
    ibuf = &(preader->ibufs[nstreams]);

    CUDA_EVENT_TIME_BEGIN
    checkCudaErrors( cudaMemcpy(ibuf->h_output,
                                ibuf->d_output,
                                MAX_PINNED_MEM_OUTSIZE,
                                cudaMemcpyDeviceToHost) );

    checkCudaErrors( cudaMemcpy(ibuf->h_packet_number_ptr,
                                ibuf->d_packet_number_sorted,
                                sizeof(uint32_t) * MAX_NUMBER_OF_PACKETS,
                                cudaMemcpyDeviceToHost) );
    CUDA_EVENT_TIME_END
    times[5] = executetime;
}

void launch_kernels(struct pktreader_s *preader, int nstreams, float *times)
{
    cudaEvent_t start, stop;
    float executetime;

    struct indexed_buffer_s *ibuf;
    ibuf = &(preader->ibufs[nstreams]);

    dim3 block(ibuf->block_size);
    dim3 grid(ibuf->grid_size);

    //remap(ibuf);
    if (ibuf->minimum != ibuf->maximum)
        appr_sort(ibuf, times);

    // checkCudaErrors( cudaMemcpy(ibuf->h_size_ptr,
    //                             ibuf->d_size_sorted,
    //                             sizeof(uint32_t) * MAX_NUMBER_OF_PACKETS,
    //                             cudaMemcpyDeviceToHost) );

    // for (int i = 0; i < MAX_NUMBER_OF_PACKETS; i++)
    //     printf("%d\n", ibuf->h_size_ptr[i]);

    CUDA_EVENT_TIME_BEGIN
#ifdef NAIVE
    rabin_naive<<< grid, block >>>(ibuf->d_data_ptr,
                                   ibuf->d_size_ptr,
                                   ibuf->d_offset_ptr,
                                   ibuf->counter,
                                   ibuf->d_rabinwin,
                                   ibuf->d_buf,
                                   ibuf->d_output);
#else
    rabin_Kepler<<< grid, block >>>(ibuf->d_data_ptr,
                                    ibuf->d_size_sorted,
                                    ibuf->d_aligned_offset_sorted,
                                    ibuf->d_packet_number_sorted,
                                    ibuf->counter,
                                    ibuf->d_rabinwin,
                                    ibuf->d_buf,
                                    ibuf->d_output);
#endif
    CUDA_EVENT_TIME_END
    times[3] = executetime;
    printf("rabin kernel exec time = %.5f ms\n", executetime);
}

#if 0
void run_with_concurrent_remap(struct pktreader_s *preader, int nstreams)
{
    int num_kernels = 4;
    struct indexed_buffer_s *ibuf;
    ibuf = &(preader->ibufs[nstreams]);

    cudaStream_t *stream = (cudaStream_t *) malloc(num_kernels * sizeof (cudaStream_t));

    for (i = 0; i < num_kernels; i++)
        cudaStreamCreate(&stream[i]);

    checkCudaErrors( cudaMemcpy(ibuf->d_input,
                                ibuf->h_input,
                                MAX_PINNED_MEM_METADATA + size,
                                cudaMemcpyHostToDevice) );

    /* launch the asynchronous memory copies and map kernels */
    for (i = 0; i < num_kernels; i++)
        checkCudaErrors( cudaMemcpyAsync(ibuf->d_input + i * chunk_size,
                                         ibuf->h_input + i * chunk_size,
                                         sizeof (float) * chunk_size,
                                         cudaMemcpyHostToDevice,
                                         stream[i]) );


}
#endif

/**
 * Optimized asynchronous rabin fingerprinting with streams.
 * NSTREAMS is defined in "comdef.h".
 */
float process_with_streams(struct pktreader_s *preader)
{
    float time;
    cudaEvent_t start, stop;

    struct indexed_buffer_s *ibuf;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    for (int i = 0; i < NSTREAMS; i++) {
        ibuf = &(preader->ibufs[i]);
        //uint32_t size = ibuf->total_aligned_size + MAX_PINNED_MEM_METADATA;
        checkCudaErrors( cudaMemcpyAsync(ibuf->d_input,
                                         ibuf->h_input,
                                         MAX_PINNED_MEM_METADATA,
                                         cudaMemcpyHostToDevice,
                                         ibuf->cuda_stream) );
    }

    for (int i = 0; i < NSTREAMS; i++) {
        ibuf = &(preader->ibufs[i]);
        appr_sort_with_streams(ibuf);
    }

    for (int i = 0; i < NSTREAMS; i++) {
        ibuf = &(preader->ibufs[i]);
        //uint32_t size = ibuf->total_aligned_size + MAX_PINNED_MEM_METADATA;
        checkCudaErrors( cudaMemcpyAsync(ibuf->d_input + MAX_PINNED_MEM_METADATA,
                                         ibuf->h_input + MAX_PINNED_MEM_METADATA,
                                         ibuf->total_aligned_size,
                                         cudaMemcpyHostToDevice,
                                         ibuf->cuda_stream) );
    }

    for (int i = 0; i < NSTREAMS; i++) {
        ibuf = &(preader->ibufs[i]);
        uint32_t pkt_count = ibuf->counter;

        dim3 block(ibuf->block_size);
        dim3 grid(ibuf->grid_size);

        rabin_Kepler<<< grid, block, 0, ibuf->cuda_stream >>>(ibuf->d_data_ptr,
                                                              ibuf->d_size_sorted,
                                                              ibuf->d_aligned_offset_sorted,
                                                              ibuf->d_packet_number_sorted,
                                                              pkt_count,
                                                              ibuf->d_rabinwin,
                                                              ibuf->d_buf,
                                                              ibuf->d_output);
    }

    for (int i = 0; i < NSTREAMS; i++) {
        ibuf = &(preader->ibufs[i]);
        checkCudaErrors( cudaMemcpyAsync(ibuf->h_output,
                                         ibuf->d_output,
                                         MAX_PINNED_MEM_OUTSIZE,
                                         cudaMemcpyDeviceToHost,
                                         ibuf->cuda_stream) );

        // checkCudaErrors( cudaMemcpyAsync(ibuf->h_packet_number_ptr,
        //                                  ibuf->d_packet_number_sorted,
        //                                  sizeof(uint32_t) * MAX_NUMBER_OF_PACKETS,
        //                                  cudaMemcpyDeviceToHost,
        //                                  ibuf->cuda_stream) );
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    return time;//printf("processing %d streams with %f ms\n", NSTREAMS, time);
}