#include "pkt_reader.h"
#include "rabinCPU.h"
#include "rabinGPU.h"

int validate_results(struct pktreader_s *preader, int nstreams)
{
    int error = 0;
    struct indexed_buffer_s *ibuf;
    ibuf = &(preader->ibufs[nstreams]);
    uint32_t pkt_count = ibuf->counter;

    uint64_t *out_idx;
    short *out_offset;
    char *out_size;

    uint64_t mask = (uint64_t)((1UL << DEFAULT_RABIN_REPR_BITS) - 1);
    int index;
    uint64_t rabinf;
    packet_t pkt;

	for (int i = 0; i < pkt_count; i++) {
	    out_idx = ibuf->h_output + i * FIXED_FP;
	    out_offset = (short *)(ibuf->h_output + MAX_NUMBER_OF_PACKETS * FIXED_FP);
	    out_size = (char *)(out_offset + MAX_NUMBER_OF_PACKETS * FIXED_FP) + i;

	    index = 0;
        pkt = ibuf->packets[i];

		LOG_MSG("-------------------Packet %d (%d)-------------------\n", i, pkt.size);

		if ((error = rabinpoly_init(ibuf->h_rabinwin, FINGERPRINT_PT, DEFAULT_RABIN_WINDOW))) {
		    fprintf(stderr, "Error encountered in rabinpoly_init %s\n", strerror(error));
		    exit(1);
	    }

		for (int j = 0; j < pkt.size; j++) {
			rabinf = rabinpoly_slide8(ibuf->h_rabinwin, pkt.buffer[j]);

			if ( (j >= DEFAULT_RABIN_WINDOW - 1) && !(rabinf & mask) && (index < FIXED_FP) ) {
                if (rabinf == out_idx[index]) {
                    index++;
                } else {
                    LOG_MSG("%lu (CPU) : %lu (GPU) : %d (index)\t\t\tNo match! ERROR!!!\n", rabinf, out_idx[index], index);
                    error = 1;
                }
			}
		}

		if (index != *out_size) {
		    LOG_MSG("CPU size: %u, GPU size: %u\n", index, *out_size);

            out_offset = out_offset + i * FIXED_FP;

            for (int k = *out_size; k > index; k--) {
                short offset = out_offset[k - 1];
                short left = offset >> 8;
                short right = offset & 0x00ff;

                LOG_MSG("packet size: %d, offset: %x left: %d, right: %d\n", pkt.size, offset, left, right);

                if ((left * ELEM_SZ + right) >= (pkt.size - DEFAULT_RABIN_WINDOW + 1)) {
                    LOG_MSG("fingerprint from index %d is due to optimization, ignore it.\n", k);
                } else {
                    error = 1;
                    printf("strange!\n"); // Always show this message.
                }

            }
		} else {
		    LOG_MSG("Equal size: %d\n", index);
		}

        rabinpoly_deinit(ibuf->h_rabinwin);
		LOG_MSG("-------------------Packet %d (%d)-------------------\n\n\n", i, pkt.size);
	}

	return error;
}