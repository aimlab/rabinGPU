// -------------------------------------------------------------
// CUDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision: 5636 $
// $Date: 2009-07-02 13:39:38 +1000 (Thu, 02 Jul 2009) $
// -------------------------------------------------------------
// This source code is distributed under the terms of license.txt
// in the root directory of this source distribution.
// -------------------------------------------------------------

/**
 * @file
 * scan_app.cu
 *
 * @brief CUDPP application-level scan routines
 */

/** \defgroup cudpp_app CUDPP Application-Level API
  * The CUDPP Application-Level API contains functions
  * that run on the host CPU and invoke GPU routines in
  * the CUDPP \link cudpp_kernel Kernel-Level API\endlink.
  * Application-Level API functions are used by
  * CUDPP \link publicInterface Public Interface\endlink
  * functions to implement CUDPP's core functionality.
  * @{
  */

/** @name Scan Functions
 * @{
 */

 #include <cstdlib>
 #include <cstdio>
 #include <assert.h>

#include "scan_globals.h"
#include "scan_kernel.cuh"
//#include "vector_kernel.cuh"

/** @brief Plan base class constructor
  *
  * @param[in]  mgr pointer to the CUDPPManager
  * @param[in]  config The configuration struct specifying algorithm and options
  * @param[in]  numElements The maximum number of elements to be processed
  * @param[in]  numRows The number of rows (for 2D operations) to be processed
  * @param[in]  rowPitch The pitch of the rows of input data, in elements
  */
CUDPPPlan::CUDPPPlan(CUDPPConfiguration config,
                     size_t numElements,
                     size_t numRows,
                     size_t rowPitch)
: m_config(config),
  m_numElements(numElements),
  m_numRows(numRows),
  m_rowPitch(rowPitch)
{
}

/** @brief Scan Plan constructor
*
* @param[in]  mgr pointer to the CUDPPManager
* @param[in]  config The configuration struct specifying algorithm and options
* @param[in]  numElements The maximum number of elements to be scanned
* @param[in]  numRows The maximum number of rows (for 2D operations) to be scanned
* @param[in]  rowPitch The pitch of the rows of input data, in elements
*/
CUDPPScanPlan::CUDPPScanPlan(CUDPPConfiguration config,
                             size_t numElements,
                             size_t numRows,
                             size_t rowPitch)
: CUDPPPlan(config, numElements, numRows, rowPitch),
  m_blockSums(0),
  m_rowPitches(0),
  m_numEltsAllocated(0),
  m_numRowsAllocated(0),
  m_numLevelsAllocated(0)
{
    allocScanStorage(this);
}

/** @brief CUDPP scan plan destructor */
CUDPPScanPlan::~CUDPPScanPlan()
{
    freeScanStorage(this);
}

/** @brief Add a uniform value to each data element of an array (vec4 version)
  *
  * This function reads one value per CTA from \a d_uniforms into shared
  * memory and adds that value to all values "owned" by the CTA in \a d_vector.
  * Each thread adds the uniform value to eight values in \a d_vector.
  *
  * @param[out] d_vector The d_vector whose values will have the uniform added
  * @param[in] d_uniforms The array of uniform values (one per CTA)
  * @param[in] numElements The number of elements in \a d_vector to process
  * @param[in] vectorRowPitch For 2D arrays, the pitch (in elements) of the
  * rows of \a d_vector.
  * @param[in] uniformRowPitch For 2D arrays, the pitch (in elements) of the
  * rows of \a d_uniforms.
  * @param[in] blockOffset an optional offset to the beginning of this block's
  * data.
  * @param[in] baseIndex an optional offset to the beginning of the array
  * within \a d_vector.
  */
template <class T, class Oper, int elementsPerThread, bool fullBlocks>
__global__ void vectorAddUniform4(T       *d_vector,
                                  const T *d_uniforms,
                                  int      numElements,
                                  int      vectorRowPitch,     // width of input array in elements
                                  int      uniformRowPitch,    // width of uniform array in elements
                                  int      blockOffset,
                                  int      baseIndex)
{
    __shared__ T uni;
    // Get this block's uniform value from the uniform array in device memory
    // We store it in shared memory so that the hardware's shared memory
    // broadcast capability can be used to share among all threads in each warp
    // in a single cycle
    if (threadIdx.x == 0)
    {
        uni = d_uniforms[blockIdx.x + __umul24(uniformRowPitch, blockIdx.y) + blockOffset];
    }

    // Compute this thread's output address
    //int width = __mul24(gridDim.x,(blockDim.x << 1));

    unsigned int address = baseIndex + __umul24(vectorRowPitch, blockIdx.y)
        + threadIdx.x + __umul24(blockIdx.x, (blockDim.x * elementsPerThread));
    numElements += __umul24(vectorRowPitch, blockIdx.y);

    __syncthreads();

    // create the operator functor
    Oper op;
#if 0
    for (int i = 0; i < elementsPerThread && address < numElements; i++)
    {
        d_vector[address] = op(d_vector[address], uni);
        address += blockDim.x;
    }
#else
    for (int i = 0; i < elementsPerThread; i++)
    {
        if (!fullBlocks && address >= numElements) return;

        d_vector[address] = op(d_vector[address], uni);
        address += blockDim.x;
    }
#endif
}

/** @brief Perform recursive scan on arbitrary size arrays
  *
  * This is the CPU-side workhorse function of the scan engine.  This function
  * invokes the CUDA kernels which perform the scan on individual blocks.
  *
  * Scans of large arrays must be split (possibly recursively) into a hierarchy of block scans,
  * where each block is scanned by a single CUDA thread block.  At each recursive level of the
  * scanArrayRecursive first invokes a kernel to scan all blocks of that level, and if the level
  * has more than one block, it calls itself recursively.  On returning from each recursive level,
  * the total sum of each block from the level below is added to all elements of the corresponding
  * block in this level.  See "Parallel Prefix Sum (Scan) in CUDA" for more information (see
  * \ref references ).
  *
  * Template parameter \a T is the datatype; \a isBackward specifies backward or forward scan;
  * \a isExclusive specifies exclusive or inclusive scan, and \a op specifies the binary associative
  * operator to be used.
  *
  * @param[out] d_out       The output array for the scan results
  * @param[in]  d_in        The input array to be scanned
  * @param[out] d_blockSums Array of arrays of per-block sums (one array per recursive level, allocated
  *                         by allocScanStorage())
  * @param[in]  numElements The number of elements in the array to scan
  * @param[in]  numRows The number of rows in the array to scan
  * @param[in]  rowPitches  Array of row pitches (one array per recursive level, allocated by
  *                         allocScanStorage())
  * @param[in]  level       The current recursive level of the scan
  */
#ifdef STREAMS
template <class T, bool isBackward, bool isExclusive, class Op>
void scanArrayRecursive(T                   *d_out,
                        const T             *d_in,
                        T                   **d_blockSums,
                        size_t              numElements,
                        size_t              numRows,
                        const size_t        *rowPitches,
                        int                 level,
                        cudaStream_t        *stream)
{
    unsigned int numBlocks =
        max(1, (unsigned int)ceil((double)numElements / ((double)SCAN_ELTS_PER_THREAD * SCAN_CTA_SIZE)));

    unsigned int sharedEltsPerBlock = SCAN_CTA_SIZE * 2;

    unsigned int sharedMemSize = sizeof(T) * sharedEltsPerBlock;

    // divide pitch by four since scan's load/store addresses are for vec4 elements
    unsigned int rowPitch = 1;
    unsigned int blockSumRowPitch = 1;

    if (numRows > 1)
    {
        rowPitch         = (unsigned int)(rowPitches[level] / 4);
        blockSumRowPitch = (unsigned int)((numBlocks > 1) ? rowPitches[level+1] / 4 : 0);
    }

    bool fullBlock = (numElements == numBlocks * SCAN_ELTS_PER_THREAD * SCAN_CTA_SIZE);

    // setup execution parameters
    dim3  grid(numBlocks, (unsigned int)numRows, 1);
    dim3  threads(SCAN_CTA_SIZE, 1, 1);

    // make sure there are no CUDA errors before we start
    CUDA_CHECK_ERROR("scanArray before kernels");

    unsigned int traitsCode = 0;
    if (numBlocks > 1) traitsCode |= 1;
    if (numRows > 1)   traitsCode |= 2;
    if (fullBlock)     traitsCode |= 4;

    switch (traitsCode)
    {
    case 0: // single block, single row, non-full block
        scan4<T, ScanTraits<T, Op, isBackward, isExclusive, false, false, false> >
               <<< grid, threads, sharedMemSize, *stream >>>
               (d_out, d_in, 0, (unsigned)numElements, rowPitch, blockSumRowPitch);
        break;
    case 1: // multiblock, single row, non-full block
        scan4< T, ScanTraits<T, Op, isBackward, isExclusive, false, true, false> >
               <<< grid, threads, sharedMemSize, *stream >>>
               (d_out, d_in, d_blockSums[level], (unsigned)numElements, rowPitch, blockSumRowPitch);
        break;
    case 2: // single block, multirow, non-full block
        scan4<T, ScanTraits<T, Op, isBackward, isExclusive, true, false, false> >
                <<< grid, threads, sharedMemSize, *stream >>>
                (d_out, d_in, 0, (unsigned)numElements, rowPitch, blockSumRowPitch);
        break;
    case 3: // multiblock, multirow, non-full block
        scan4<T, ScanTraits<T, Op, isBackward, isExclusive, true, true, false> >
                <<< grid, threads, sharedMemSize, *stream >>>
                (d_out, d_in, d_blockSums[level], (unsigned)numElements, rowPitch, blockSumRowPitch);
        break;
    case 4: // single block, single row, full block
        scan4<T, ScanTraits<T, Op, isBackward, isExclusive, false, false, true> >
               <<< grid, threads, sharedMemSize, *stream >>>
               (d_out, d_in, 0, (unsigned)numElements, rowPitch, blockSumRowPitch);
        break;
    case 5: // multiblock, single row, full block
        scan4< T, ScanTraits<T, Op, isBackward, isExclusive, false, true, true> >
               <<< grid, threads, sharedMemSize, *stream >>>
               (d_out, d_in, d_blockSums[level], (unsigned)numElements, rowPitch, blockSumRowPitch);
        break;
    case 6: // single block, multirow, full block
        scan4<T, ScanTraits<T, Op, isBackward, isExclusive, true, false, true> >
                <<< grid, threads, sharedMemSize, *stream >>>
                (d_out, d_in, 0, (unsigned)numElements, rowPitch, blockSumRowPitch);
        break;
    case 7: // multiblock, multirow, full block
        scan4<T, ScanTraits<T, Op, isBackward, isExclusive, true, true, true> >
                <<< grid, threads, sharedMemSize, *stream >>>
                (d_out, d_in, d_blockSums[level], (unsigned)numElements, rowPitch, blockSumRowPitch);
        break;
    }

    CUDA_CHECK_ERROR("prescan");

    if (numBlocks > 1)
    {
        // After scanning all the sub-blocks, we are mostly done. But
        // now we need to take all of the last values of the
        // sub-blocks and scan those. This will give us a new value
        // that must be sdded to each block to get the final results.

        scanArrayRecursive<T, isBackward, true, Op>
            ((T*)d_blockSums[level], (const T*)d_blockSums[level],
             (T**)d_blockSums, numBlocks, numRows, rowPitches, level + 1, stream); // recursive (CPU) call

        if (fullBlock)
            vectorAddUniform4<T, Op, SCAN_ELTS_PER_THREAD, true>
                <<< grid, threads, 0, *stream >>>(d_out,
                                      (T*)d_blockSums[level],
                                      (unsigned)numElements,
                                      rowPitch*4,
                                      blockSumRowPitch*4,
                                      0, 0);
        else
            vectorAddUniform4<T, Op, SCAN_ELTS_PER_THREAD, false>
                <<< grid, threads, 0, *stream >>>(d_out,
                                      (T*)d_blockSums[level],
                                      (unsigned)numElements,
                                      rowPitch*4,
                                      blockSumRowPitch*4,
                                      0, 0);

        CUDA_CHECK_ERROR("vectorAddUniform");
    }
}
#else
template <class T, bool isBackward, bool isExclusive, class Op>
void scanArrayRecursive(T                   *d_out,
                        const T             *d_in,
                        T                   **d_blockSums,
                        size_t              numElements,
                        size_t              numRows,
                        const size_t        *rowPitches,
                        int                 level,
                        cudaStream_t        *stream)
{
    unsigned int numBlocks =
        max(1, (unsigned int)ceil((double)numElements / ((double)SCAN_ELTS_PER_THREAD * SCAN_CTA_SIZE)));

    unsigned int sharedEltsPerBlock = SCAN_CTA_SIZE * 2;

    unsigned int sharedMemSize = sizeof(T) * sharedEltsPerBlock;

    // divide pitch by four since scan's load/store addresses are for vec4 elements
    unsigned int rowPitch = 1;
    unsigned int blockSumRowPitch = 1;

    if (numRows > 1)
    {
        rowPitch         = (unsigned int)(rowPitches[level] / 4);
        blockSumRowPitch = (unsigned int)((numBlocks > 1) ? rowPitches[level+1] / 4 : 0);
    }

    bool fullBlock = (numElements == numBlocks * SCAN_ELTS_PER_THREAD * SCAN_CTA_SIZE);

    // setup execution parameters
    dim3  grid(numBlocks, (unsigned int)numRows, 1);
    dim3  threads(SCAN_CTA_SIZE, 1, 1);

    // make sure there are no CUDA errors before we start
    CUDA_CHECK_ERROR("scanArray before kernels");

    unsigned int traitsCode = 0;
    if (numBlocks > 1) traitsCode |= 1;
    if (numRows > 1)   traitsCode |= 2;
    if (fullBlock)     traitsCode |= 4;

    switch (traitsCode)
    {
    case 0: // single block, single row, non-full block
        scan4<T, ScanTraits<T, Op, isBackward, isExclusive, false, false, false> >
               <<< grid, threads, sharedMemSize >>>
               (d_out, d_in, 0, (unsigned)numElements, rowPitch, blockSumRowPitch);
        break;
    case 1: // multiblock, single row, non-full block
        scan4< T, ScanTraits<T, Op, isBackward, isExclusive, false, true, false> >
               <<< grid, threads, sharedMemSize >>>
               (d_out, d_in, d_blockSums[level], (unsigned)numElements, rowPitch, blockSumRowPitch);
        break;
    case 2: // single block, multirow, non-full block
        scan4<T, ScanTraits<T, Op, isBackward, isExclusive, true, false, false> >
                <<< grid, threads, sharedMemSize >>>
                (d_out, d_in, 0, (unsigned)numElements, rowPitch, blockSumRowPitch);
        break;
    case 3: // multiblock, multirow, non-full block
        scan4<T, ScanTraits<T, Op, isBackward, isExclusive, true, true, false> >
                <<< grid, threads, sharedMemSize >>>
                (d_out, d_in, d_blockSums[level], (unsigned)numElements, rowPitch, blockSumRowPitch);
        break;
    case 4: // single block, single row, full block
        scan4<T, ScanTraits<T, Op, isBackward, isExclusive, false, false, true> >
               <<< grid, threads, sharedMemSize >>>
               (d_out, d_in, 0, (unsigned)numElements, rowPitch, blockSumRowPitch);
        break;
    case 5: // multiblock, single row, full block
        scan4< T, ScanTraits<T, Op, isBackward, isExclusive, false, true, true> >
               <<< grid, threads, sharedMemSize >>>
               (d_out, d_in, d_blockSums[level], (unsigned)numElements, rowPitch, blockSumRowPitch);
        break;
    case 6: // single block, multirow, full block
        scan4<T, ScanTraits<T, Op, isBackward, isExclusive, true, false, true> >
                <<< grid, threads, sharedMemSize >>>
                (d_out, d_in, 0, (unsigned)numElements, rowPitch, blockSumRowPitch);
        break;
    case 7: // multiblock, multirow, full block
        scan4<T, ScanTraits<T, Op, isBackward, isExclusive, true, true, true> >
                <<< grid, threads, sharedMemSize >>>
                (d_out, d_in, d_blockSums[level], (unsigned)numElements, rowPitch, blockSumRowPitch);
        break;
    }

    CUDA_CHECK_ERROR("prescan");

    if (numBlocks > 1)
    {
        // After scanning all the sub-blocks, we are mostly done. But
        // now we need to take all of the last values of the
        // sub-blocks and scan those. This will give us a new value
        // that must be sdded to each block to get the final results.

        scanArrayRecursive<T, isBackward, true, Op>
            ((T*)d_blockSums[level], (const T*)d_blockSums[level],
             (T**)d_blockSums, numBlocks, numRows, rowPitches, level + 1, stream); // recursive (CPU) call

        if (fullBlock)
            vectorAddUniform4<T, Op, SCAN_ELTS_PER_THREAD, true>
                <<< grid, threads >>>(d_out,
                                      (T*)d_blockSums[level],
                                      (unsigned)numElements,
                                      rowPitch*4,
                                      blockSumRowPitch*4,
                                      0, 0);
        else
            vectorAddUniform4<T, Op, SCAN_ELTS_PER_THREAD, false>
                <<< grid, threads >>>(d_out,
                                      (T*)d_blockSums[level],
                                      (unsigned)numElements,
                                      rowPitch*4,
                                      blockSumRowPitch*4,
                                      0, 0);

        CUDA_CHECK_ERROR("vectorAddUniform");
    }
}
#endif

// global

#ifdef __cplusplus
extern "C"
{
#endif

/** @brief Allocate intermediate arrays used by scan.
  *
  * Scans of large arrays must be split (possibly recursively) into a hierarchy
  * of block scans, where each block is scanned by a single CUDA thread block.
  * At each recursive level of the scan, we need an array in which to store the
  * total sums of all blocks in that level.  This function computes the amount
  * of storage needed and allocates it.
  *
  * @param plan Pointer to CUDPPScanPlan object containing options and number
  *             of elements, which is used to compute storage requirements, and
  *             within which intermediate storage is allocated.
  */
void allocScanStorage(CUDPPScanPlan *plan)
{
    plan->m_numEltsAllocated = plan->m_numElements;

    size_t numElts = plan->m_numElements;

    size_t level = 0;

    do
    {
        size_t numBlocks =
            max(1, (unsigned int)ceil((double)numElts / ((double)SCAN_ELTS_PER_THREAD * SCAN_CTA_SIZE)));
        if (numBlocks > 1)
        {
            level++;
        }
        numElts = numBlocks;
    } while (numElts > 1);

    size_t elementSize = 0;

    switch(plan->m_config.datatype)
    {
    case CUDPP_INT:
        plan->m_blockSums = (void**) malloc(level * sizeof(int*));
        elementSize = sizeof(int);
        break;
    case CUDPP_UINT:
        plan->m_blockSums = (void**) malloc(level * sizeof(unsigned int*));
        elementSize = sizeof(unsigned int);
        break;
    case CUDPP_FLOAT:
        plan->m_blockSums = (void**) malloc(level * sizeof(float*));
        elementSize = sizeof(float);
        break;
    case CUDPP_DOUBLE:
        plan->m_blockSums = (void**) malloc(level * sizeof(double*));
        elementSize = sizeof(double);
        break;
    case CUDPP_LONGLONG:
        plan->m_blockSums = (void**) malloc(level * sizeof(long long*));
        elementSize = sizeof(long long);
        break;
    case CUDPP_ULONGLONG:
        plan->m_blockSums = (void**) malloc(level * sizeof(unsigned long long*));
        elementSize = sizeof(unsigned long long);
        break;
    default:
        break;
    }

    plan->m_numLevelsAllocated = level;
    numElts = plan->m_numElements;
    size_t numRows = plan->m_numRows;
    plan->m_numRowsAllocated = numRows;
    plan->m_rowPitches = 0;

    if (numRows > 1)
    {
        plan->m_rowPitches = (size_t*) malloc((level + 1) * sizeof(size_t));
        plan->m_rowPitches[0] = plan->m_rowPitch;
    }

    level = 0;

    do
    {
        size_t numBlocks =
            max(1, (unsigned int)ceil((double)numElts / ((double)SCAN_ELTS_PER_THREAD * SCAN_CTA_SIZE)));
        if (numBlocks > 1)
        {
            // Use cudaMallocPitch for multi-row block sums to ensure alignment
            if (numRows > 1)
            {
                size_t dpitch;
                CUDA_SAFE_CALL( cudaMallocPitch((void**) &(plan->m_blockSums[level]),
                                                &dpitch,
                                                numBlocks * elementSize,
                                                numRows));
                plan->m_rowPitches[level+1] = dpitch / elementSize;
                level++;
            }
            else
            {
                CUDA_SAFE_CALL(cudaMalloc((void**) &(plan->m_blockSums[level++]),
                                          numBlocks * elementSize));
            }
        }
        numElts = numBlocks;
    } while (numElts > 1);

    CUDA_CHECK_ERROR("allocScanStorage");
}

/** @brief Deallocate intermediate block sums arrays in a CUDPPScanPlan object.
  *
  * These arrays must have been allocated by allocScanStorage(), which is called
  * by the constructor of cudppScanPlan().
  *
  * @param plan Pointer to CUDPPScanPlan object initialized by allocScanStorage().
  */
void freeScanStorage(CUDPPScanPlan *plan)
{
    for (unsigned int i = 0; i < plan->m_numLevelsAllocated; i++)
    {
        cudaFree(plan->m_blockSums[i]);
    }

    CUDA_CHECK_ERROR("freeScanStorage");

    free((void**)plan->m_blockSums);
    if (plan->m_numRows > 1)
        free((void*)plan->m_rowPitches);

    plan->m_blockSums = 0;
    plan->m_numEltsAllocated = 0;
    plan->m_numLevelsAllocated = 0;
}

#ifdef __cplusplus
}
#endif

template <typename T, bool isBackward, bool isExclusive>
void cudppScanDispatchOperator(void                *d_out,
                               const void          *d_in,
                               size_t              numElements,
                               size_t              numRows,
                               const CUDPPScanPlan *plan)
{
    switch(plan->m_config.op)
    {
    case CUDPP_ADD:
        scanArrayRecursive<T, isBackward, isExclusive, OperatorAdd<T> >
            ((T*)d_out, (const T*)d_in,
            (T**)plan->m_blockSums,
            numElements, numRows, plan->m_rowPitches, 0, plan->m_stream);
        break;
//    case CUDPP_MULTIPLY:
//        scanArrayRecursive<T, isBackward, isExclusive, OperatorMultiply<T> >
//            ((T*)d_out, (const T*)d_in,
//            (T**)plan->m_blockSums,
//            numElements, numRows, plan->m_rowPitches, 0);
//        break;
//    case CUDPP_MAX:
//        scanArrayRecursive<T, isBackward, isExclusive, OperatorMax<T> >
//            ((T*)d_out, (const T*)d_in,
//            (T**)plan->m_blockSums,
//            numElements, numRows, plan->m_rowPitches, 0);
//        break;
//    case CUDPP_MIN:
//        scanArrayRecursive<T, isBackward, isExclusive, OperatorMin<T> >
//            ((T*)d_out, (const T*)d_in,
//            (T**)plan->m_blockSums,
//            numElements, numRows, plan->m_rowPitches, 0);
//        break;
    default:
        break;
    }
}

template <bool isBackward, bool isExclusive>
void cudppScanDispatchType(void                *d_out,
                           const void          *d_in,
                           size_t              numElements,
                           size_t              numRows,
                           const CUDPPScanPlan *plan)
{
    switch(plan->m_config.datatype)
    {
    case CUDPP_INT:
        cudppScanDispatchOperator<int,
                                  isBackward,
                                  isExclusive>(d_out, d_in, numElements,
                                               numRows, plan);
        break;
//    case CUDPP_UINT:
//        cudppScanDispatchOperator<unsigned int,
//                                  isBackward,
//                                  isExclusive>(d_out, d_in, numElements,
//                                               numRows, plan);
//        break;
//    case CUDPP_FLOAT:
//        cudppScanDispatchOperator<float,
//                                  isBackward,
//                                  isExclusive>(d_out, d_in, numElements,
//                                               numRows, plan);
//        break;
//    case CUDPP_DOUBLE:
//        cudppScanDispatchOperator<double,
//                                  isBackward,
//                                  isExclusive>(d_out, d_in, numElements,
//                                               numRows, plan);
//        break;
//    case CUDPP_LONGLONG:
//        cudppScanDispatchOperator<long long,
//                                  isBackward,
//                                  isExclusive>(d_out, d_in, numElements,
//                                               numRows, plan);
//        break;
//    case CUDPP_ULONGLONG:
//        cudppScanDispatchOperator<unsigned long long,
//                                  isBackward,
//                                  isExclusive>(d_out, d_in, numElements,
//                                               numRows, plan);
//        break;
    default:
        break;
    }
}

#ifdef __cplusplus
extern "C"
{
#endif

/** @brief Dispatch function to perform a scan (prefix sum) on an
  * array with the specified configuration.
  *
  * This is the dispatch routine which calls scanArrayRecursive() with
  * appropriate template parameters and arguments to achieve the scan as
  * specified in \a plan.
  *
  * @param[out] d_out    The output array of scan results
  * @param[in]  d_in     The input array
  * @param[in]  numElements The number of elements to scan
  * @param[in]  numRows     The number of rows to scan in parallel
  * @param[in]  plan     Pointer to CUDPPScanPlan object containing scan options
  *                      and intermediate storage
  */
void cudppScanDispatch(void                *d_out,
                       const void          *d_in,
                       size_t              numElements,
                       size_t              numRows,
                       const CUDPPScanPlan *plan)
{
    if (CUDPP_OPTION_EXCLUSIVE & plan->m_config.options)
    {
        if (CUDPP_OPTION_BACKWARD & plan->m_config.options)
        {
            cudppScanDispatchType<true, true>(d_out, d_in, numElements,
                                              numRows, plan);
        }
        else
        {
            cudppScanDispatchType<false, true>(d_out, d_in, numElements,
                                               numRows, plan);
        }
    }
    else
    {
        if (CUDPP_OPTION_BACKWARD & plan->m_config.options)
        {
            cudppScanDispatchType<true, false>(d_out, d_in, numElements,
                                               numRows, plan);
        }
        else
        {
            cudppScanDispatchType<false, false>(d_out, d_in, numElements,
                                                numRows, plan);
        }
    }
}

#ifdef __cplusplus
}
#endif

/** @} */ // end scan functions
/** @} */ // end cudpp_app
