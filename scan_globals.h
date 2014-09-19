// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision: 5636 $
// $Date: 2009-07-02 13:39:38 +1000 (Thu, 02 Jul 2009) $
// -------------------------------------------------------------
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// -------------------------------------------------------------

/**
 * @file
 * cudpp_globals.h
 *
 * @brief Global declarations defining machine characteristics of GPU target.
 * These are currently set for best performance on G8X GPUs.  The optimal
 * parameters may change on future GPUs. In the future, we hope to make
 * CUDPP a self-tuning library.
 */

#ifndef __CUDPP_GLOBALS_H__
#define __CUDPP_GLOBALS_H__

#include <limits.h>
#include <float.h>
#include <stdio.h>

#if CUDART_VERSION >= 4000
#define CUDA_DEVICE_SYNCHRONIZE( )   cudaDeviceSynchronize();
#else
#define CUDA_DEVICE_SYNCHRONIZE( )   cudaThreadSynchronize();
#endif

#  define CUDA_SAFE_CALL_NO_SYNC( call) {                                    \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } }

#  define CUDA_SAFE_CALL( call)     CUDA_SAFE_CALL_NO_SYNC(call);

//! Check for CUDA error
#ifdef _DEBUG
#  define CUDA_CHECK_ERROR(errorMessage) {                                    \
    cudaError_t err = cudaGetLastError();                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
    err = CUDA_DEVICE_SYNCHRONIZE();                                           \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
    }
#else
#  define CUDA_CHECK_ERROR(errorMessage) {                                    \
    cudaError_t err = cudaGetLastError();                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
    }
#endif

class CUDPPScanPlan;

extern "C"
void allocScanStorage(CUDPPScanPlan *plan);

extern "C"
void freeScanStorage(CUDPPScanPlan *plan);

extern "C"
void cudppScanDispatch(void                *d_out,
                       const void          *d_in,
                       size_t              numElements,
                       size_t              numRows,
                       const CUDPPScanPlan *plan);

const int SORT_CTA_SIZE = 256;                   /**< Number of threads per CTA for radix sort. Must equal 16 * number of radices */
const int SCAN_CTA_SIZE = 128;                   /**< Number of threads in a CTA */
const int REDUCE_CTA_SIZE = 256;                 /**< Number of threads in a CTA */

const int LOG_SCAN_CTA_SIZE = 7;                 /**< log_2(CTA_SIZE) */

const int WARP_SIZE = 32;                        /**< Number of threads in a warp */

const int LOG_WARP_SIZE = 5;                     /**< log_2(WARP_SIZE) */
const int LOG_SIZEOF_FLOAT = 2;                  /**< log_2(sizeof(float)) */

const int SCAN_ELTS_PER_THREAD = 8;              /**< Number of elements per scan thread */
const int SEGSCAN_ELTS_PER_THREAD = 8;           /**< Number of elements per segmented scan thread */

/**
 * @brief CUDPP Result codes returned by CUDPP API functions.
 */
enum CUDPPResult
{
    CUDPP_SUCCESS = 0,                 /**< No error. */
    CUDPP_ERROR_INVALID_HANDLE,        /**< Specified handle (for example,
                                            to a plan) is invalid. **/
    CUDPP_ERROR_ILLEGAL_CONFIGURATION, /**< Specified configuration is
                                            illegal. For example, an
                                            invalid or illogical
                                            combination of options. */
    CUDPP_ERROR_INVALID_PLAN,          /**< The plan is not configured properly.
                                            For example, passing a plan for scan
                                            to cudppSegmentedScan. */
    CUDPP_ERROR_INSUFFICIENT_RESOURCES,/**< The function could not complete due to
                                            insufficient resources (typically CUDA
                                            device resources such as shared memory)
                                            for the specified problem size. */
    CUDPP_ERROR_UNKNOWN = 9999         /**< Unknown or untraceable error. */
};

/**
 * @brief Options for configuring CUDPP algorithms.
 *
 * @see CUDPPConfiguration, cudppPlan, CUDPPAlgorithm
 */
enum CUDPPOption
{
    CUDPP_OPTION_FORWARD   = 0x1,  /**< Algorithms operate forward:
                                    * from start to end of input
                                    * array */
    CUDPP_OPTION_BACKWARD  = 0x2,  /**< Algorithms operate backward:
                                    * from end to start of array */
    CUDPP_OPTION_EXCLUSIVE = 0x4,  /**< Exclusive (for scans) - scan
                                    * includes all elements up to (but
                                    * not including) the current
                                    * element */
    CUDPP_OPTION_INCLUSIVE = 0x8,  /**< Inclusive (for scans) - scan
                                    * includes all elements up to and
                                    * including the current element */
    CUDPP_OPTION_CTA_LOCAL = 0x10, /**< Algorithm performed only on
                                    * the CTAs (blocks) with no
                                    * communication between blocks.
                                    * @todo Currently ignored. */
    CUDPP_OPTION_KEYS_ONLY = 0x20, /**< No associated value to a key
                                    * (for global radix sort) */
    CUDPP_OPTION_KEY_VALUE_PAIRS = 0x40, /**< Each key has an associated value */
};


/**
 * @brief Datatypes supported by CUDPP algorithms.
 *
 * @see CUDPPConfiguration, cudppPlan
 */
enum CUDPPDatatype
{
    CUDPP_CHAR,     //!< Character type (C char)
    CUDPP_UCHAR,    //!< Unsigned character (byte) type (C unsigned char)
    CUDPP_INT,      //!< Integer type (C int)
    CUDPP_UINT,     //!< Unsigned integer type (C unsigned int)
    CUDPP_FLOAT,    //!< Float type (C float)
    CUDPP_DOUBLE,   //!< Double type (C double)
    CUDPP_LONGLONG, //!< 64-bit integer type (C long long)
    CUDPP_ULONGLONG,//!< 64-bit unsigned integer type (C unsigned long long)
    CUDPP_DATATYPE_INVALID,  //!< invalid datatype (must be last in list)
};

/**
 * @brief Operators supported by CUDPP algorithms (currently scan and
 * segmented scan).
 *
 * These are all binary associative operators.
 *
 * @see CUDPPConfiguration, cudppPlan
 */
enum CUDPPOperator
{
    CUDPP_ADD,      //!< Addition of two operands
    CUDPP_MULTIPLY, //!< Multiplication of two operands
    CUDPP_MIN,      //!< Minimum of two operands
    CUDPP_MAX,      //!< Maximum of two operands
    CUDPP_OPERATOR_INVALID, //!< invalid operator (must be last in list)
};

/**
* @brief Configuration struct used to specify algorithm, datatype,
* operator, and options when creating a plan for CUDPP algorithms.
*
* @see cudppPlan
*/
struct CUDPPConfiguration
{
    CUDPPOperator  op;        //!< The numerical operator to be applied
    CUDPPDatatype  datatype;  //!< The datatype of the input arrays
    unsigned int   options;   //!< Options to configure the algorithm
};

/** @brief Base class for CUDPP Plan data structures
  *
  * CUDPPPlan and its subclasses provide the internal (i.e. not visible to the
  * library user) infrastructure for planning algorithm execution.  They
  * own intermediate storage for CUDPP algorithms as well as, in some cases,
  * information about optimal execution configuration for the present hardware.
  *
  */
class CUDPPPlan
{
public:
    CUDPPPlan(CUDPPConfiguration config,
              size_t numElements, size_t numRows, size_t rowPitch);
    virtual ~CUDPPPlan() {}

    // Note anything passed to functions compiled by NVCC must be public
    CUDPPConfiguration m_config;        //!< @internal Options structure
    size_t             m_numElements;   //!< @internal Maximum number of input elements
    size_t             m_numRows;       //!< @internal Maximum number of input rows
    size_t             m_rowPitch;      //!< @internal Pitch of input rows in elements
    //CUDPPManager      *m_planManager;  //!< @internal pointer to the manager of this plan
    cudaStream_t       *m_stream;

    //! @internal Convert this pointer to an opaque handle
    //! @returns Handle to a CUDPP plan
//    CUDPPHandle getHandle()
//    {
//        return reinterpret_cast<CUDPPHandle>(this);
//    }
};

/** @brief Plan class for scan algorithm
  *
  */
class CUDPPScanPlan : public CUDPPPlan
{
public:
    CUDPPScanPlan(CUDPPConfiguration config, size_t numElements, size_t numRows, size_t rowPitch);
    virtual ~CUDPPScanPlan();

    void  **m_blockSums;          //!< @internal Intermediate block sums array
    size_t *m_rowPitches;         //!< @internal Pitch of each row in elements (for cudppMultiScan())
    size_t  m_numEltsAllocated;   //!< @internal Number of elements allocated (maximum scan size)
    size_t  m_numRowsAllocated;   //!< @internal Number of rows allocated (for cudppMultiScan())
    size_t  m_numLevelsAllocated; //!< @internal Number of levels allocaed (in _scanBlockSums)
};

/** @brief Utility template struct for generating small vector types from scalar types
  *
  * Given a base scalar type (\c int, \c float, etc.) and a vector length (1 through 4) as
  * template parameters, this struct defines a vector type (\c float3, \c int4, etc.) of the
  * specified length and base type.  For example:
  * \code
  * template <class T>
  * __device__ void myKernel(T *data)
  * {
  *     typeToVector<T,4>::Result myVec4;             // create a vec4 of type T
  *     myVec4 = (typeToVector<T,4>::Result*)data[0]; // load first element of data as a vec4
  * }
  * \endcode
  *
  * This functionality is implemented using template specialization.  Currently specializations
  * for int, float, and unsigned int vectors of lengths 2-4 are defined.  Note that this results
  * in types being generated at compile time -- there is no runtime cost.  typeToVector is used by
  * the optimized scan \c __device__ functions in scan_cta.cu.
  */
template <typename T, int N>
struct typeToVector
{
    typedef T Result;
};

template<>
struct typeToVector<int, 4>
{
    typedef int4 Result;
};
template<>
struct typeToVector<unsigned int, 4>
{
    typedef uint4 Result;
};
template<>
struct typeToVector<float, 4>
{
    typedef float4 Result;
};
template<>
struct typeToVector<double, 4>
{
    typedef double4 Result;
};
template<>
struct typeToVector<long long, 4>
{
    typedef longlong4 Result;
};
template<>
struct typeToVector<unsigned long long, 4>
{
    typedef ulonglong4 Result;
};
template<>
struct typeToVector<int, 3>
{
    typedef int3 Result;
};
template<>
struct typeToVector<unsigned int, 3>
{
    typedef uint3 Result;
};
template<>
struct typeToVector<float, 3>
{
    typedef float3 Result;
};
template<>
struct typeToVector<long long, 3>
{
    typedef longlong3 Result;
};
template<>
struct typeToVector<unsigned long long, 3>
{
    typedef ulonglong3 Result;
};
template<>
struct typeToVector<int, 2>
{
    typedef int2 Result;
};
template<>
struct typeToVector<unsigned int, 2>
{
    typedef uint2 Result;
};
template<>
struct typeToVector<float, 2>
{
    typedef float2 Result;
};
template<>
struct typeToVector<long long, 2>
{
    typedef longlong2 Result;
};
template<>
struct typeToVector<unsigned long long, 2>
{
    typedef ulonglong2 Result;
};

template <typename T>
class OperatorAdd
{
public:
    __device__ T operator()(const T a, const T b) { return a + b; }
    __device__ T identity() { return (T)0; }
};

template <typename T>
class OperatorMultiply
{
public:
    __device__ T operator()(const T a, const T b) { return a * b; }
    __device__ T identity() { return (T)1; }
};

template <typename T>
class OperatorMax
{
public:
    __device__ T operator() (const T a, const T b) const { return max(a, b); }
    __device__ T identity() const { return (T)0; }
};

template <>
__device__ inline int OperatorMax<int>::identity() const { return INT_MIN; }
template <>
__device__ inline unsigned int OperatorMax<unsigned int>::identity() const { return 0; }
template <>
__device__ inline float OperatorMax<float>::identity() const { return -FLT_MAX; }
template <>
__device__ inline double OperatorMax<double>::identity() const { return -DBL_MAX; }
template <>
__device__ inline long long OperatorMax<long long>::identity() const { return LLONG_MIN; }
template <>
__device__ inline unsigned long long OperatorMax<unsigned long long>::identity() const { return 0; }

template <typename T>
class OperatorMin
{
public:
    __device__ T operator() (const T a, const T b) const { return min(a, b); }
    __device__ T identity() const { return (T)0; }
};

template <>
__device__ inline int OperatorMin<int>::identity() const { return INT_MAX; }
template <>
__device__ inline unsigned int OperatorMin<unsigned int>::identity() const { return UINT_MAX; }
template <>
__device__ inline float OperatorMin<float>::identity() const { return FLT_MAX; }
template <>
__device__ inline double OperatorMin<double>::identity() const { return DBL_MAX; }
template <>
__device__ inline long long OperatorMin<long long>::identity() const { return LLONG_MAX; }
template <>
__device__ inline unsigned long long OperatorMin<unsigned long long>::identity() const { return ULLONG_MAX; }

#endif // __CUDPP_GLOBALS_H__

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
