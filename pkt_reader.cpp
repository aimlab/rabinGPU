#include <stdio.h>
#include "pkt_reader.h"

bool pktreader_init(pktreader_t *preader)
{
    return 0;
}

static inline void store_packet_to_buffer(struct indexed_buffer_s *ibuf,
										  uint8_t *payload, uint32_t size)
{
	int index = 0;

	uint32_t aligned_size = size;
	uint32_t total_aligned_size = ibuf->total_aligned_size;
	uint32_t counter = ibuf->counter;

#ifdef NAIVE
    memcpy(ibuf->h_data_ptr + ibuf->total_real_size, payload, size);
#else
    memcpy(ibuf->h_data_ptr + total_aligned_size, payload, size);
#endif


	ibuf->h_size_ptr[counter] = size;
	ibuf->h_offset_ptr[counter] = ibuf->total_real_size;
	ibuf->h_packet_number_ptr[counter] = counter;
    ibuf->h_aligned_offset_ptr[counter] = total_aligned_size / ELEM_SZ;

	if (size % ELEM_SZ != 0) {
        aligned_size = size + (ELEM_SZ - size % ELEM_SZ);
    }

    ibuf->total_aligned_size += aligned_size;
    ibuf->total_real_size += size;
    ibuf->counter++;
    ibuf->num_elem = ibuf->total_aligned_size / ELEM_SZ;

    ibuf->minimum = size < ibuf->minimum ? size : ibuf->minimum;
    ibuf->maximum = size > ibuf->maximum ? size : ibuf->maximum;
}

bool pktreader_generate_packets(pktreader_t *preader)
{
    uint32_t count = 0;
    uint32_t payload_size = 0;
    uint32_t total_size = 0;
    uint8_t *tmp;
    std::vector<unsigned int> vector = preader->s_vector;

    struct indexed_buffer_s *ibuf;

    for (int i = 0; i < vector.size(); i++) {
        payload_size = vector[i];

        if (payload_size < DEFAULT_RABIN_WINDOW)
            payload_size += DEFAULT_RABIN_WINDOW;

        for (int j = 0; j < NSTREAMS; j++) {
            ibuf = &(preader->ibufs[j]);

            ibuf->packets[count].size = payload_size;
			tmp = ibuf->packets[count].buffer = (uint8_t *)malloc(payload_size);

			if (tmp == NULL) {
			    printf("malloc error. payload_size: %d\n", payload_size);
			    exit(0);
			}

			for (int j = 0; j < payload_size; j++) {
			    uint8_t value = rand() % 256;
			    tmp[j] = value;
			}

            store_packet_to_buffer(ibuf, (uint8_t *)tmp, payload_size);
        }
        count++;
        total_size += payload_size;
    }

    // Calculate block and grid dimensions for each stream
    for (int i = 0; i < NSTREAMS; i++) {
        ibuf = &(preader->ibufs[i]);

        if (count <= MAX_THREADS_PER_BLOCK ) {
            ibuf->block_size = MAX_THREADS_PER_BLOCK;
            ibuf->grid_size = 1;
        } else {
            ibuf->block_size = MAX_THREADS_PER_BLOCK;
            ibuf->grid_size = ((count % ibuf->block_size) == 0) ?
                              (count / ibuf->block_size) :
                              (count / ibuf->block_size) + 1;
        }
    }

    preader->packet_counter = count * NSTREAMS;
	preader->total_size = total_size * NSTREAMS;
	return TRUE;
}

void pktreader_finalize(pktreader_t *preader)
{
    struct indexed_buffer_s *ibuf;
    uint32_t count;

    for (int i = 0; i < NSTREAMS; i++) {
        ibuf = &(preader->ibufs[i]);
        count = ibuf->counter;

        if (count > 0) {
		    for (int j = 0; j < count; j++)
			    free(ibuf->packets[j].buffer);
	    }
    }
}