#include <string>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include "rabinGPU.h"
#include "comdef.h"
#include "pkt_reader.h"

#define CUDA_CALL(x) do { if((x) != cudaSuccess) {      \
      printf("Error at %s:%d\n",__FILE__,__LINE__);     \
      return EXIT_FAILURE;}} while(0)


void exclusive_scan_cpu(uint32_t *data, uint32_t length)
{
    // Perform inclusive scan
    for (int i = 0; i < length - 1; i++) {
        data[i + 1] = data[i] + data[i + 1];
    }

    // Set the last element to 0 because we need an exclusive scan
    data[length - 1] = 0;
}
//#define SHARED_MEM_ATOMIC
__global__ void assign_bucket(uint32_t *d_input,
                              uint32_t length,
                              uint32_t maximum,
                              uint32_t minimum,
                              uint32_t *d_offset,
                              uint32_t *d_bucket_count,
                              uint32_t *d_bucket_index)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    uint32_t bucket_index;
    //uint32_t maximum, minimum;

    if (idx >= length) return;

    uint32_t value = d_input[idx];

    //minimum = d_maxmin[0];
    //maximum = d_maxmin[1];

    // Assigning elements to buckets and incrementing the bucket counts
    // Calculate the bucket_index for each element
    bucket_index = (value - minimum) * (NUM_BUCKETS - 1) / (maximum - minimum);
    d_bucket_index[idx] = bucket_index;

    d_offset[idx] = atomicInc(&d_bucket_count[bucket_index], length);
}

__global__ void approximate_sort(uint32_t *d_key,
                                 uint32_t *d_key_sorted,
                                 uint32_t *d_value,
                                 uint32_t *d_value_sorted,
                                 uint32_t *d_packet_number,
                                 uint32_t *d_packet_number_sorted,
                                 uint32_t length,
                                 uint32_t *d_offset,
                                 uint32_t *d_bucket_count,
                                 uint32_t *d_bucket_index)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t count = 0;

    if (idx >= length) return;

    uint32_t key = d_key[idx];
    uint32_t value = d_value[idx];
    uint32_t number = d_packet_number[idx];

    uint32_t bucket_index = d_bucket_index[idx];
    uint32_t offset = d_offset[idx];

    // if (bucket_index == 0)
    //     count = d_bucket_count[NUM_BUCKETS - 1];
    // else
    //     count = d_bucket_count[bucket_index - 1];
    count = d_bucket_count[bucket_index];

    offset = offset + count;

    __syncthreads();

    // Do not write to the same vector
    d_key_sorted[offset] = key;
    d_value_sorted[offset] = value;
    d_packet_number_sorted[offset] = number;
}

void appr_sort(struct indexed_buffer_s* ibuf, float *times)
{
    cudaEvent_t start, stop;
    float executetime;

    int threads = NUM_BUCKETS;
    int blocks = MAX_NUMBER_OF_PACKETS / threads;

    CUDA_EVENT_TIME_BEGIN
    //Distribute elements into their respective buckets
    assign_bucket<<<blocks, threads, NUM_BUCKETS*sizeof(uint32_t)>>>(ibuf->d_size_ptr,
                                                                     MAX_NUMBER_OF_PACKETS,
                                                                     ibuf->maximum,
                                                                     ibuf->minimum,
                                                                     ibuf->d_sort_offset,
                                                                     ibuf->d_bucket_count_in,
                                                                     ibuf->d_bucket_index);
    CUDA_EVENT_TIME_END
    times[0] = executetime;
    printf("assign_bucket time: %.5f ms\n", executetime);

    CUDA_EVENT_TIME_BEGIN
    // checkCudaErrors( cudaMemcpy(ibuf->h_bucket_count,
    //                             ibuf->d_bucket_count,
    //                             NUM_BUCKETS * sizeof(uint32_t),
    //                             cudaMemcpyDeviceToHost) );

    // exclusive_scan_cpu(ibuf->h_bucket_count, NUM_BUCKETS);

    // checkCudaErrors( cudaMemcpy(ibuf->d_bucket_count,
    //                             ibuf->h_bucket_count,
    //                             NUM_BUCKETS * sizeof(uint32_t),
    //                             cudaMemcpyHostToDevice) );
    // for (int i = 0; i < NUM_BUCKETS; i++) {
    //     printf("%u\n", ibuf->h_bucket_count[i]);
    // }
    cudppScanDispatch(ibuf->d_bucket_count, ibuf->d_bucket_count_in, NUM_BUCKETS, 1, (CUDPPScanPlan *)(ibuf->plan));

    // checkCudaErrors( cudaMemcpy(ibuf->h_bucket_count,
    //                             ibuf->d_bucket_count,
    //                             NUM_BUCKETS * sizeof(uint32_t),
    //                             cudaMemcpyDeviceToHost) );
    // for (int i = 0; i < NUM_BUCKETS; i++) {
    //     printf("%u\n", ibuf->h_bucket_count[i]);
    // }
    CUDA_EVENT_TIME_END
    times[1] = executetime;
    printf("scan time: %.5f ms\n", executetime);

    CUDA_EVENT_TIME_BEGIN
    approximate_sort<<<blocks, threads>>>(ibuf->d_size_ptr,
                                          ibuf->d_size_sorted,
                                          ibuf->d_aligned_offset_ptr,
                                          ibuf->d_aligned_offset_sorted,
                                          ibuf->d_packet_number_ptr,
                                          ibuf->d_packet_number_sorted,
                                          MAX_NUMBER_OF_PACKETS,
                                          ibuf->d_sort_offset,
                                          ibuf->d_bucket_count,
                                          ibuf->d_bucket_index);
    CUDA_EVENT_TIME_END
    printf("approximate_sort time: %.5f ms\n", executetime);
    times[2] = executetime;
}

void appr_sort_with_streams(struct indexed_buffer_s* ibuf)
{
    // cudaEvent_t start, stop;
    // float executetime;

    int threads = NUM_BUCKETS;
    int blocks = MAX_NUMBER_OF_PACKETS / threads;

    //CUDA_EVENT_TIME_BEGIN
    //Distribute elements into their respective buckets
    assign_bucket<<<blocks, threads, 0, ibuf->cuda_stream>>>(ibuf->d_size_ptr,
                                                             MAX_NUMBER_OF_PACKETS,
                                                             ibuf->maximum,
                                                             ibuf->minimum,
                                                             ibuf->d_sort_offset,
                                                             ibuf->d_bucket_count_in,
                                                             ibuf->d_bucket_index);
    //CUDA_EVENT_TIME_END
    //printf("assign_bucket time: %.5f ms\n", executetime);

    //CUDA_EVENT_TIME_BEGIN
    // checkCudaErrors( cudaMemcpyAsync(ibuf->h_bucket_count,
    //                                  ibuf->d_bucket_count,
    //                                  NUM_BUCKETS * sizeof(uint32_t),
    //                                  cudaMemcpyDeviceToHost,
    //                                  ibuf->cuda_stream) );

    // exclusive_scan_cpu(ibuf->h_bucket_count, NUM_BUCKETS);

    // checkCudaErrors( cudaMemcpyAsync(ibuf->d_bucket_count,
    //                                  ibuf->h_bucket_count,
    //                                  NUM_BUCKETS * sizeof(uint32_t),
    //                                  cudaMemcpyHostToDevice,
    //                                  ibuf->cuda_stream) );
    //CUDA_EVENT_TIME_END
    //printf("scan time: %.5f ms\n", executetime);

    cudppScanDispatch(ibuf->d_bucket_count, ibuf->d_bucket_count_in, NUM_BUCKETS, 1, (CUDPPScanPlan *)(ibuf->plan));


    //CUDA_EVENT_TIME_BEGIN
    approximate_sort<<<blocks, threads, 0, ibuf->cuda_stream>>>(ibuf->d_size_ptr,
                                                                ibuf->d_size_sorted,
                                                                ibuf->d_aligned_offset_ptr,
                                                                ibuf->d_aligned_offset_sorted,
                                                                ibuf->d_packet_number_ptr,
                                                                ibuf->d_packet_number_sorted,
                                                                MAX_NUMBER_OF_PACKETS,
                                                                ibuf->d_sort_offset,
                                                                ibuf->d_bucket_count,
                                                                ibuf->d_bucket_index);
    //CUDA_EVENT_TIME_END
    //printf("approximate_sort time: %.5f ms\n", executetime);
}

#if 0
int main(int argc, char **argv)
{
    std::string distr[6]={"uniform","gaussian","zero","bucket","staggered","sorted"};

    const uint N = 4096 * 8;
    int numBuckets = 512;

    cudaEvent_t start, stop;
    float executetime;

    uint *h_vector;
    uint *h_vector_out;
    uint *d_vector;
    uint *d_vector_out;
    uint *d_bucket_index;
    uint *h_bucket_index;

    uint *h_bucket_count;

    size_t value  = 8 * 1024 * 1024;
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, value);

    uint length = N;

    int threads = numBuckets;
    int blocks = N / threads;

    //array showing the number of elements in each bucket
    uint *d_bucket_count;
    //array showing what bucket every element is in
    uint *d_offset;
    //find max and min with thrust
    uint maximum, minimum;

    h_vector = (uint *)malloc(length * sizeof(uint));
    h_vector_out = (uint *)malloc(2 * sizeof(uint));
    //distribution<uint>(h_vector, length, distr[0]);

    for (int i = 0; i < length; i++) {
        h_vector[i] = (uint)(rand() % 1460);
        printf("%u\n", h_vector[i]);
    }
    h_vector[0] = 0;
    h_vector[N-1] = 1459;

    CUDA_CALL(cudaMalloc(&d_vector, length * sizeof(uint)));
    CUDA_CALL(cudaMalloc(&d_vector_out, length * sizeof(uint)));
    //Allocate memory to store bucket assignments
    CUDA_CALL(cudaMalloc(&d_offset, length * sizeof(uint)));
    CUDA_CALL(cudaMalloc(&d_bucket_index, length * sizeof(uint)));
    //Allocate memory to store bucket counts
    size_t totalBucketSize = numBuckets * sizeof(uint);
    CUDA_CALL(cudaMalloc((void **)&d_bucket_count, totalBucketSize));

    CUDA_CALL(cudaMallocHost((void **)&h_bucket_count, totalBucketSize));
    CUDA_CALL(cudaMallocHost((void **)&h_bucket_index, length * sizeof(uint)));

    //Set the bucket count vector to all zeros
    cudaMemset(d_bucket_count, 0, totalBucketSize);

    CUDA_CALL(cudaMemcpy(d_vector, h_vector, length * sizeof(uint), cudaMemcpyHostToDevice));

    // CUDA_EVENT_TIME_BEGIN
    // compute_reduction(d_vector, d_vector_out, length);
    // CUDA_EVENT_TIME_END
    // printf("custom minmax time: %.5f ms\n", executetime);

    // CUDA_EVENT_TIME_BEGIN
    // thrust::device_ptr<uint>dev_ptr(d_vector);
    // thrust::pair< thrust::device_ptr<uint>, thrust::device_ptr<uint> > result = thrust::minmax_element(dev_ptr, dev_ptr + length);
    // CUDA_EVENT_TIME_END
    // printf("minmax_element time: %.5f ms\n", executetime);

    minimum = 0;//*result.first;
    maximum = 1459;//*result.second;

    // CUDA_CALL(cudaMemcpy(h_vector_out, d_vector_out, 2 * sizeof(uint), cudaMemcpyDeviceToHost));

    // if (minimum == h_vector_out[0] && maximum == h_vector_out[1])
    //     printf("%s\n", "max-min kernel runs correctly");

    //if the max and the min are the same, then we are done
    if (maximum == minimum)
        return maximum;

    //Calculate max-min
    double range = maximum - minimum;
    //Calculate the slope, i.e numBuckets/range
    double slope = (numBuckets - 1)/range;

    printf("min: %u, max: %u, slope: %f\n", minimum, maximum, slope);

    CUDA_EVENT_TIME_BEGIN
    //Distribute elements into their respective buckets
    assign_bucket<<<blocks, threads, numBuckets*sizeof(uint)>>>(d_vector,
                                                                length,
                                                                numBuckets,
                                                                maximum,
                                                                minimum,
                                                                d_offset,
                                                                d_bucket_count,
                                                                d_bucket_index);
    CUDA_EVENT_TIME_END
    printf("assign_bucket time: %.5f ms\n", executetime);

    printf("\n\n\n\n");
    //print_index<<<blocks, threads>>>(d_bucket_index, length);
    printf("\n\n\n\n");

    CUDA_EVENT_TIME_BEGIN
    CUDA_CALL(cudaMemcpy(h_bucket_count, d_bucket_count, totalBucketSize, cudaMemcpyDeviceToHost));
    exclusive_scan_cpu(h_bucket_count, numBuckets);
    CUDA_CALL(cudaMemcpy(d_bucket_count, h_bucket_count, totalBucketSize, cudaMemcpyHostToDevice));
    CUDA_EVENT_TIME_END
    printf("scan time: %.5f ms\n", executetime);

    CUDA_CALL(cudaMemcpy(h_bucket_index, d_bucket_index, length * sizeof(uint), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(h_vector, d_vector, length * sizeof(uint), cudaMemcpyDeviceToHost));

    for (int i = 0; i < length; i++) {
        //printf("value : index --- %u : %u\n", h_vector[i], h_bucket_index[i]);
    }
    printf("\n\n\n\n");

    CUDA_CALL(cudaMemcpy(d_bucket_index, h_bucket_index, length * sizeof(uint), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_vector, h_vector, length * sizeof(uint), cudaMemcpyHostToDevice));

    CUDA_EVENT_TIME_BEGIN
    approximate_sort<<<blocks, threads>>>(d_vector, d_vector_out, length, numBuckets, d_offset, d_bucket_count, d_bucket_index);
    CUDA_EVENT_TIME_END
    printf("approximate_sort time: %.5f ms\n", executetime);

    CUDA_CALL(cudaMemcpy(h_vector, d_vector_out, length * sizeof(uint), cudaMemcpyDeviceToHost));
    for (int i = 0; i < length; i++) {
        printf("%u\n", h_vector[i]);
    }

    free(h_vector);
    free(h_vector_out);

    CUDA_CALL(cudaFreeHost(h_bucket_count));
    CUDA_CALL(cudaFreeHost(h_bucket_index));

    CUDA_CALL(cudaFree(d_vector));
    CUDA_CALL(cudaFree(d_offset));
    CUDA_CALL(cudaFree(d_bucket_count));
    CUDA_CALL(cudaFree(d_bucket_index));
    CUDA_CALL(cudaFree(d_vector_out));

    return 0;
}
#endif