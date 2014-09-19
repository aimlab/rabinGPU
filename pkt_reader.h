#ifndef PACKET_READER_H_
#define PACKET_READER_H_

#include <cuda_runtime.h>
#include <vector>
#include "comdef.h"
#include "scan_globals.h"

#define MAX_PACKETS MAX_NUMBER_OF_PACKETS  // max number of packets
#define MAX_PACKET_SIZE DEFAULT_MTU        // packet max size

typedef struct packet_s {
	uint8_t *buffer;
	int size;
} packet_t;

typedef struct indexed_buffer_s {
	packet_t packets[MAX_PACKETS];
    uint32_t counter;
    uint32_t total_real_size;
    uint32_t total_aligned_size;

    uint8_t *d_input;
    uint8_t *h_input;

    uint8_t *d_buf;

    /* pointers to d_input */
    uint8_t *d_data_ptr;
    uint8_t *h_data_ptr;

    uint32_t *d_size_ptr;
    uint32_t *h_size_ptr;

    uint32_t *d_offset_ptr;
    uint32_t *h_offset_ptr;

    /* sorted size and offset array */
    uint32_t *d_size_sorted;
    uint32_t *d_aligned_offset_sorted;

    uint64_t *d_output;
    uint64_t *h_output;

    struct rabinpoly_window *d_rabinwin;
    struct rabinpoly_window *h_rabinwin;

    uint32_t *d_packet_number_ptr;
    uint32_t *d_packet_number_sorted;
    uint32_t *h_packet_number_ptr;

    uint32_t *d_aligned_offset_ptr;
    uint32_t *h_aligned_offset_ptr;

    /* Used in approximate sorting */
    //uint32_t *d_maxmin;
    uint32_t *d_bucket_index;
    uint32_t *d_bucket_count;
    uint32_t *d_bucket_count_in;
    uint32_t *h_bucket_count;
    uint32_t *d_sort_offset;

    uint32_t maximum;
    uint32_t minimum;

    uint32_t size;
    uint32_t max_size;
    uint32_t num_elem;

    uint32_t block_size;
    uint32_t grid_size;

    cudaEvent_t cuda_event;
    cudaStream_t cuda_stream;
    CUDPPPlan *plan;
} indexed_buffer_t;

typedef struct pktreader_s {
    packet_t packets[MAX_PACKETS];
    indexed_buffer_t ibufs[NSTREAMS];

    /* s_vector contains generated packet size distribution */
    std::vector<unsigned int> s_vector;

    uint32_t packet_counter;
    uint32_t total_size;
} pktreader_t;

bool pktreader_init(pktreader_t *preader);
void pktreader_finalize(pktreader_t *preader);
bool pktreader_generate_packets(pktreader_t *preader);

#endif /* PACKET_READER_H_ */
