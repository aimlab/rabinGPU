#include "rabinGPU.h"
#include "scan_globals.h"

void indexed_buffer_init(struct pktreader_s *preader)
{
    struct indexed_buffer_s *ibuf;
    uint32_t max_size = 0;
    uint32_t num_elem = 0;
    int err;

    max_size = DEFAULT_MTU + (ELEM_SZ - DEFAULT_MTU % ELEM_SZ);
    num_elem = max_size * MAX_NUMBER_OF_PACKETS / ELEM_SZ;

    CUDPPConfiguration config;

    config.op = CUDPP_ADD;
    config.datatype = CUDPP_INT;

    CUDPPOption direction = CUDPP_OPTION_FORWARD;
    CUDPPOption inclusivity = CUDPP_OPTION_EXCLUSIVE;

    config.options = direction | inclusivity;

    for (int i = 0; i < NSTREAMS; i++) {
        ibuf = &(preader->ibufs[i]);

        ibuf->counter = 0;
        ibuf->total_aligned_size = 0;
        ibuf->max_size = 0;

        ibuf->maximum = 0;
        ibuf->minimum = UINT_MAX;

        ibuf->plan = new CUDPPScanPlan(config, NUM_BUCKETS, 1, 0);

        checkCudaErrors( cudaMalloc((void **)&(ibuf->d_input), MAX_PINNED_MEM_INSIZE) );
        checkCudaErrors( cudaMalloc((void **)&(ibuf->d_output), MAX_PINNED_MEM_OUTSIZE) );
        checkCudaErrors( cudaMalloc((void **)&(ibuf->d_buf), RABINPOLY_BUFFER) );

        checkCudaErrors( cudaMalloc((void **)&(ibuf->d_size_sorted), MAX_NUMBER_OF_PACKETS * sizeof(uint32_t)) );
        checkCudaErrors( cudaMalloc((void **)&(ibuf->d_aligned_offset_sorted), MAX_NUMBER_OF_PACKETS * sizeof(uint32_t)) );
        checkCudaErrors( cudaMalloc((void **)&(ibuf->d_rabinwin), MAX_PINNED_MEM_RABINWIN ) );

        //checkCudaErrors( cudaMalloc((void **)&(ibuf->d_maxmin), MAX_NUMBER_OF_PACKETS * sizeof(uint32_t)) );
        checkCudaErrors( cudaMalloc((void **)&(ibuf->d_sort_offset), MAX_NUMBER_OF_PACKETS * sizeof(uint32_t)) );
        checkCudaErrors( cudaMalloc((void **)&(ibuf->d_bucket_index), MAX_NUMBER_OF_PACKETS * sizeof(uint32_t)) );
        checkCudaErrors( cudaMalloc((void **)&(ibuf->d_bucket_count), NUM_BUCKETS * sizeof(uint32_t)) );
        checkCudaErrors( cudaMalloc((void **)&(ibuf->d_bucket_count_in), NUM_BUCKETS * sizeof(uint32_t)) );

        checkCudaErrors( cudaMalloc((void **)&(ibuf->d_packet_number_sorted), MAX_NUMBER_OF_PACKETS * sizeof(uint32_t)) );

        /* d_buf must be initialized with 0 */
        checkCudaErrors( cudaMemset((void *)ibuf->d_buf, 0, RABINPOLY_BUFFER) );
        checkCudaErrors( cudaMemset((void *)ibuf->d_bucket_count, 0, NUM_BUCKETS * sizeof(uint32_t)) );
        checkCudaErrors( cudaMemset((void *)ibuf->d_bucket_count_in, 0, NUM_BUCKETS * sizeof(uint32_t)) );

        checkCudaErrors( cudaMallocHost((void **)&(ibuf->h_input), MAX_PINNED_MEM_INSIZE) );
        checkCudaErrors( cudaMallocHost((void **)&(ibuf->h_output), MAX_PINNED_MEM_OUTSIZE) );
        checkCudaErrors( cudaMallocHost((void **)&(ibuf->h_rabinwin), MAX_PINNED_MEM_RABINWIN) );
        checkCudaErrors( cudaMallocHost((void **)&(ibuf->h_bucket_count), NUM_BUCKETS * sizeof(uint32_t)) );

        ibuf->h_data_ptr = (uint8_t *)(ibuf->h_input) + MAX_PINNED_MEM_METADATA;
        ibuf->h_size_ptr = (uint32_t *)(ibuf->h_input);
        ibuf->h_offset_ptr = (uint32_t *)((uint8_t *)(ibuf->h_input) + MAX_PINNED_MEM_METADATA / 4);
        ibuf->h_packet_number_ptr = (uint32_t *)((uint8_t *)(ibuf->h_input) + MAX_PINNED_MEM_METADATA * 2 / 4);
        ibuf->h_aligned_offset_ptr = (uint32_t *)((uint8_t *)(ibuf->h_input) + MAX_PINNED_MEM_METADATA * 3 / 4);

        ibuf->d_data_ptr = (uint8_t *)(ibuf->d_input) + MAX_PINNED_MEM_METADATA;
        ibuf->d_size_ptr = (uint32_t *)(ibuf->d_input);
        ibuf->d_offset_ptr = (uint32_t *)((uint8_t *)(ibuf->d_input) + MAX_PINNED_MEM_METADATA / 4);
        ibuf->d_packet_number_ptr = (uint32_t *)((uint8_t *)(ibuf->d_input) + MAX_PINNED_MEM_METADATA * 2 / 4);
        ibuf->d_aligned_offset_ptr = (uint32_t *)((uint8_t *)(ibuf->d_input) + MAX_PINNED_MEM_METADATA * 3 / 4);

        checkCudaErrors( cudaStreamCreate(&(ibuf->cuda_stream)) );
        checkCudaErrors( cudaEventCreate(&(ibuf->cuda_event)) );

        ibuf->plan->m_stream = &(ibuf->cuda_stream);

        if ((err = rabinpoly_init(ibuf->h_rabinwin, FINGERPRINT_PT, DEFAULT_RABIN_WINDOW))) {
            fprintf(stderr, "Error encountered in rabinpoly_init %s\n", strerror(err));
            exit(1);
        }

        checkCudaErrors( cudaMemcpy(ibuf->d_rabinwin, ibuf->h_rabinwin, MAX_PINNED_MEM_RABINWIN, cudaMemcpyHostToDevice) );
    }
}

void indexed_buffer_finalize(struct pktreader_s *preader)
{
    struct indexed_buffer_s *ibuf;

    for (int i = 0; i < NSTREAMS; i++) {
        ibuf = &(preader->ibufs[i]);

        checkCudaErrors( cudaFree(ibuf->d_input) );
        checkCudaErrors( cudaFree(ibuf->d_output) );
        checkCudaErrors( cudaFree(ibuf->d_buf) );

        checkCudaErrors( cudaFree(ibuf->d_size_sorted) );
        checkCudaErrors( cudaFree(ibuf->d_aligned_offset_sorted) );
        checkCudaErrors( cudaFree(ibuf->d_rabinwin) );

        //checkCudaErrors( cudaFree(ibuf->d_maxmin) );
        checkCudaErrors( cudaFree(ibuf->d_sort_offset) );
        checkCudaErrors( cudaFree(ibuf->d_bucket_index) );
        checkCudaErrors( cudaFree(ibuf->d_bucket_count) );
        checkCudaErrors( cudaFree(ibuf->d_bucket_count_in) );

        checkCudaErrors( cudaFree(ibuf->d_packet_number_sorted));

        checkCudaErrors( cudaFreeHost(ibuf->h_input) );
        checkCudaErrors( cudaFreeHost(ibuf->h_output) );
        checkCudaErrors( cudaFreeHost(ibuf->h_rabinwin) );
        checkCudaErrors( cudaFreeHost(ibuf->h_bucket_count) );

        checkCudaErrors( cudaStreamDestroy(ibuf->cuda_stream) );
        checkCudaErrors( cudaEventDestroy(ibuf->cuda_event) );

        delete static_cast<CUDPPScanPlan*>(ibuf->plan);
    }
}