#include <stdio.h>
#include <fcntl.h>
#include <cuda_runtime.h>

#include <vector>
#include <string>

#include "comdef.h"
#include "timer.h"
#include "pkt_reader.h"
#include "rabinGPU.h"
//#include "distribution.h" // For generating packet size distribution.

inline void zero(std::vector<unsigned int>& vec)
{
    unsigned int value = 1460;//rand() % 1460;
    unsigned int sum = 0;

    for (int i = 0; i < MAX_NUMBER_OF_PACKETS; i++) {
        if ( (i + 1) % 32 == 0)
            vec.push_back(1459);
        else
            vec.push_back(value);

        sum += value;
    }
}

inline void uniform(std::vector<unsigned int>& vec)
{
    unsigned int value;
    unsigned int sum = 0;

    for (int i = 0; i < MAX_NUMBER_OF_PACKETS; i++) {
        value = rand() % 1460;
        vec.push_back(value);
        sum += value;
    }
}

void rabin_cpu(struct pktreader_s *preader, int nstreams)
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

        //LOG_MSG("-------------------Packet %d (%d)-------------------\n", i, pkt.size);

        // if ((error = rabinpoly_init(ibuf->h_rabinwin, FINGERPRINT_PT, DEFAULT_RABIN_WINDOW))) {
        //     fprintf(stderr, "Error encountered in rabinpoly_init %s\n", strerror(error));
        //     exit(1);
        // }

        for (int j = 0; j < pkt.size; j++) {
            rabinf = rabinpoly_slide8(ibuf->h_rabinwin, pkt.buffer[j]);

            if ( (j >= DEFAULT_RABIN_WINDOW - 1) && !(rabinf & mask) && (index < FIXED_FP) ) {
                out_idx[index] = rabinf;
                out_offset[index] = j;
            }
        }

        *out_size = index;

        // rabinpoly_deinit(ibuf->h_rabinwin);
        //LOG_MSG("-------------------Packet %d (%d)-------------------\n\n\n", i, pkt.size);
    }

    return;
}

int main(int argc, char **argv)
{
    pktreader_t reader;
    mytimer_t timer;
    float times[6];
    float s_time = 0;
    float total = 0;
    float total_no_io = 0;
    std::string distr[6]={"uniform","Gaussian","zero","bucket","staggered","sorted"};

    size_t value  = 8 * 1024 * 1024;
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, value);

    //cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

    // FIXME: segmentation fault when vector size exceeds certain value e.g. 4096*64
    uniform(reader.s_vector);
    //distribution(reader.s_vector, distr[2]);
    printf("after distribution. size: %ld\n", reader.s_vector.size());

    pktreader_init(&reader);
    indexed_buffer_init(&reader);

    pktreader_generate_packets(&reader);

    for (int i = 0; i < NSTREAMS; i++) {
        printf("counter: %d, num_elem: %d, increment: %d, total_size: %d\n",
               reader.ibufs[i].counter,
               reader.ibufs[i].num_elem,
               reader.ibufs[i].total_aligned_size,
               reader.ibufs[i].total_real_size);
    }

    printf("number of packets: %d\n", reader.packet_counter);
    printf("total length: %d\n", reader.total_size);

#ifdef STREAMS

    s_time = process_with_streams(&reader);
    printf("%f Gbps\n", (reader.total_size * 8) / (1024 * 1024 * s_time));

    for (int i = 0; i < NSTREAMS; i++)
        if (validate_results(&reader, i))
            printf("error occurred!\n");

#else

    timer_start(&timer);
    memcpy_host_to_device(&reader, 0, times);
    timer_stop(&timer);
    printf("memcpy_host_to_device time consumed: %f\n", timer_elapsed(&timer));

    timer_start(&timer);
    launch_kernels(&reader, 0, times);
    timer_stop(&timer);
    printf("launch_kernels time consumed: %f\n", timer_elapsed(&timer));

    timer_start(&timer);
    memcpy_device_to_host(&reader, 0, times);
    timer_stop(&timer);
    printf("memcpy_device_to_host time consumed: %f\n", timer_elapsed(&timer));

    for (int i = 0; i < 4; i++)
        total_no_io += times[i];
    for (int i = 0; i < 6; i++)
        total += times[i];

    printf("%f Gbps (Without transfer)\n", (reader.ibufs[0].total_real_size * 8) / (1024 * 1024 * total_no_io));
    printf("%f Gbps (With transfer)\n", (reader.ibufs[0].total_real_size * 8) / (1024 * 1024 * total));

    for (int i = 0; i < 1; i++)
        if (validate_results(&reader, i))
            printf("error occurred!\n");

#endif

    // timer_start(&timer);
    // rabin_cpu(&reader, 0);
    // timer_stop(&timer);
    // printf("%f Gbps (CPU) \n", (reader.ibufs[0].total_real_size * 8) /  (1024 * 1024 * timer_elapsed(&timer)));

    indexed_buffer_finalize(&reader);
    pktreader_finalize(&reader);

    return 0;
}
