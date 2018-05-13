/*
 ============================================================================
 Name        : radixSortMSD_DynPar.cu
 Author      : Nishith Tirpankar
 Version     :
 Copyright   : BeerWare!
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>

#include <random>
#include <vector>
#include <algorithm>
#include "CudaErrorCheck.cuh"

#include <cuda.h>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/util_allocator.cuh>

#include "drecho.h"

#define USETIMERS
enum {
	tid_this = 0,
	tid_that,
	tid_move,
	tid_local_sort,
	tid_thr_bound_calc,
	tid_split_buc_bound_calc,
	tid_count
};
//__device__ float cuda_timers[ tid_count ];
#ifdef USETIMERS
#define TIMER_TIC if ( threadIdx.x == 0 ) tic = clock();
#define TIMER_TOC(tid) toc = clock(); if ( threadIdx.x == 0 ) atomicAdd( &cuda_timers[tid] , ( toc > tic ) ? (toc - tic) : ( toc + (0xffffffff - tic) ) );
#else
#define TIMER_TIC
#define TIMER_TOC(tid)
#endif


const bool dr::log_timestamp = true;
const bool dr::log_branch = true;
const bool dr::log_branch_scope = true;
const bool dr::log_text = true;
const bool dr::log_errno = true;
const bool dr::log_location = true;


using namespace cub;

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

template<typename Key, int NUM_BUCKETS>
__device__ __forceinline__ int bucketnum(Key num, unsigned int bitnum){
	const unsigned int bits_in_buckets = Log2<NUM_BUCKETS>::VALUE;
	return (num >> (bitnum-bits_in_buckets+1)) & (NUM_BUCKETS-1);
}

template<
	typename Key,
	int BLOCK_THREADS,
	int ITEMS_PER_THREAD,
	int NUM_BUCKETS>
__launch_bounds__ (BLOCK_THREADS)
__global__ void countKernel(
		Key 			*d_in,			// Tile of input
		unsigned int	*d_scan_in,		// Device scan location of size NUM_BUCKETS*NUM_THREADS where local count will be written
		unsigned int 	num_elems,		// Total number of elements to be counted
		unsigned int 	bitnum
		)
{
	enum { TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD };
	// Our current block's offset
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int block_offset = blockIdx.x * TILE_SIZE;
	int thread_offset = block_offset + threadIdx.x * ITEMS_PER_THREAD;
	int num_threads = blockDim.x * gridDim.x;
	Key l_cnt[NUM_BUCKETS] = {0};

	for(unsigned int i = thread_offset; i < thread_offset+ITEMS_PER_THREAD && i < num_elems; i++) {
//		if(bucketnum<Key, NUM_BUCKETS>(d_in[i], bitnum) >= NUM_BUCKETS){
//			printf("ERROR LOOK HERE: input: %d, bucket: %d\n", d_in[i], bucketnum<Key, NUM_BUCKETS>(d_in[i], bitnum));
//			continue;
//		}
		l_cnt[bucketnum<Key, NUM_BUCKETS>(d_in[i], bitnum)]++;
	}

	// copy local count into global memory
	for(unsigned int i = 0; i < NUM_BUCKETS; i++){
		d_scan_in[i*num_threads+idx] = l_cnt[i];
	}
#if 0
	printf("Count tid %d: l_cnt{%d, %d}, d_scan_in{%d, %d} bitnum %d\n", idx, l_cnt[0], l_cnt[1], d_scan_in[0*num_threads+idx], d_scan_in[1*num_threads+idx], bitnum);
#endif
}

// copyKernel<<<ceil(l_num_elems/TILE_SIZE), BLOCK_THREADS>>>(l_out, l_in, l_num_elems);
template<
	typename Key,
	int BLOCK_THREADS,
	int ITEMS_PER_THREAD,
	int NUM_BUCKETS>
__launch_bounds__ (BLOCK_THREADS)
__global__ void copyKernel(
		Key				*d_in,		// Tile of input
		Key				*d_out,		// Tile of output
		unsigned int 	num_elems	// Total number of elements to be moved
		)
{
	enum { TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD };
	unsigned int block_offset = blockIdx.x * TILE_SIZE;
	unsigned int thread_offset = block_offset + threadIdx.x * ITEMS_PER_THREAD;
	//int idx = threadIdx.x + blockIdx.x*blockDim.x;

	for(unsigned int i = thread_offset; i < thread_offset+ITEMS_PER_THREAD && i < num_elems; i++) {
		d_out[i] = d_in[i];
	}
}

template<
	typename Key,
	int BLOCK_THREADS,
	int ITEMS_PER_THREAD,
	int NUM_BUCKETS>
__launch_bounds__ (BLOCK_THREADS)
__global__ void moveKernel(
		Key 			*d_in,		// Tile of input
		Key 			*d_out,	 	// Tile of output
		unsigned int	*d_scan_out,// Device scan location of size NUM_BUCKETS*NUM_THREADS where local count will be written
		unsigned int 	num_elems,		// Total number of elements to be counted
		unsigned int 	bitnum
		)
{
	enum { TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD };
	// Our current block's offset
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int block_offset = blockIdx.x * TILE_SIZE;
	int thread_offset = block_offset + threadIdx.x * ITEMS_PER_THREAD;
	int num_threads = blockDim.x * gridDim.x;
	Key l_scan_inc[NUM_BUCKETS] = {0};

	for(unsigned int i = 0; i < NUM_BUCKETS; i++){
		l_scan_inc[i] = d_scan_out[i*num_threads+idx];
	}

	for(unsigned int i = thread_offset; i < thread_offset+ITEMS_PER_THREAD && i < num_elems; i++) {
//		if(bucketnum<Key, NUM_BUCKETS>(d_in[i], bitnum) >= NUM_BUCKETS){
//			printf("ERROR LOOK HERE: input: %d, bucket: %d\n", d_in[i], bucketnum<Key, NUM_BUCKETS>(d_in[i], bitnum));
//			continue;
//		}
		l_scan_inc[bucketnum<Key, NUM_BUCKETS>(d_in[i], bitnum)]--;
		d_out[l_scan_inc[bucketnum<Key, NUM_BUCKETS>(d_in[i], bitnum)]] = d_in[i];
	}
}

#define LOCAL_SORT_THRESHOLD 1024*8

template <
    typename    Key,
    int         BLOCK_THREADS,
    int         ITEMS_PER_THREAD,
    int			NUM_BUCKETS>
__global__ void radixSort(
    Key           *d_in,		// Tile of input
    Key           *d_out,		// Tile of buffer - this is where the output should be at a particular level/depth finally after the operation is complete
    unsigned int  *num_elems,	// The total number of elements in the list at this level - array of size NUM_BUCKETS
    unsigned int  *offsets,		// The offset from d_in where the current bucket starts - array of size NUM_BUCKETS
    unsigned int  *bitnum_beg	// the bit from which the buckets will be counted
    )
{
	enum { TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD };
	unsigned int l_offset = offsets[threadIdx.x];
	unsigned int l_num_elems = num_elems[threadIdx.x];
	Key *l_in = &d_in[l_offset];
	Key *l_out = &d_out[l_offset];
	unsigned int GRID_SIZE = (unsigned int)ceilf(((float)l_num_elems)/((float)TILE_SIZE));
	unsigned int NUM_THREADS = GRID_SIZE*BLOCK_THREADS;
	unsigned int *d_scan_in, *d_scan_out;
	unsigned int *d_next_num_elems, *d_next_offsets, *d_next_bitnum_beg;
	cudaError_t err;

	// Storage pointers for cubs
	void *d_temp_storage = NULL;
	size_t temp_storage_bytes = 0;
#if 0
	printf("STARTING radixsort for bitnum_beg %d threadIdx.x %d l_offset %d l_num_elems %d \n", bitnum_beg[0], threadIdx.x, l_offset, l_num_elems);
#endif

	if(l_num_elems == 0){
#if 0
		printf("EXITING bitnum_beg %d threadIdx.x %d l_offset %d l_num_elems %d \n", bitnum_beg[0], threadIdx.x, l_offset, l_num_elems);
#endif
		return;
	}
	// --------------------------------------------------------------------------------------------------------
	// if the number of elements in this bucket is too small or the next recursion will go below the zeroth bit
	if(l_num_elems <= LOCAL_SORT_THRESHOLD || ((((int)bitnum_beg[0]-(int)Log2<NUM_BUCKETS>::VALUE))< 0) ) {
//		printf("LOCAL SORTING for bitnum_beg %d threadIdx.x %d l_offset %d l_num_elems %d \n", bitnum_beg[0], threadIdx.x, l_offset, l_num_elems);
		DoubleBuffer<Key> d_keys(l_in, l_out);
		d_temp_storage = NULL;
		temp_storage_bytes = 0;
		DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, l_num_elems);
		cudaMalloc(&d_temp_storage, temp_storage_bytes);
		if(d_temp_storage == NULL){ printf("ERROR: d_temp_storage cudaMalloc failed.\n"); return;}
		DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, l_num_elems);
		cudaDeviceSynchronize();
		//d_out = d_keys.Current();
		if(d_keys.Current() != l_out){
			copyKernel<Key, BLOCK_THREADS, ITEMS_PER_THREAD, NUM_BUCKETS><<<GRID_SIZE, BLOCK_THREADS>>>(l_in, l_out, l_num_elems);
			cudaDeviceSynchronize();
		}
		cudaFree((void *) d_temp_storage);
		return;
	}

	// --------------------------------------------------------------------------------------------------------
	// Count
	//printf("Running count kernel now....\n");
	cudaMalloc((void ** )&d_scan_in, sizeof(unsigned int)*NUM_THREADS*NUM_BUCKETS);
	if(d_scan_in == NULL){ printf("ERROR: d_scan_in cudaMalloc failed.\n"); return;}
	countKernel<Key, BLOCK_THREADS, ITEMS_PER_THREAD, NUM_BUCKETS><<<GRID_SIZE, BLOCK_THREADS>>>(l_in, d_scan_in, l_num_elems, bitnum_beg[0]);

	err = cudaGetLastError();
	if (err != cudaSuccess){ printf("ERROR: countKernel failed due to err code %d.\n", err); return;}

	cudaDeviceSynchronize();
	//TESTS
#if 0
	//cudaMalloc((void ** )&d_scan_in, sizeof(unsigned int)*NUM_THREADS*16);
	//countKernel<Key, BLOCK_THREADS, ITEMS_PER_THREAD, 16><<<GRID_SIZE, BLOCK_THREADS>>>(l_in, d_scan_in, l_num_elems, 11);
	//cudaDeviceSynchronize();
//	printf("Result of count with NUM_THREADS %d for bit_beg %d:\n", NUM_THREADS, bitnum_beg[0]);
//	for(int i = 0; i < NUM_THREADS; i++){
//		printf("TID %d: ", i);
//		for(int j = 0; j < NUM_BUCKETS; j++){
//			if((j*NUM_THREADS+i) > NUM_THREADS*NUM_BUCKETS)
//				printf("BAD ");
//			else
//				printf("%03d ", d_scan_in[j*NUM_THREADS+i]);
//		}
//		printf("\n");
//	}
//	printf("\n");
	printf("CHECKING MEMORY d_scan_in data NUM_THREADS %d for bit_beg %d:\n", NUM_THREADS, bitnum_beg[0]);
	for(int i = 0; i < NUM_THREADS*NUM_BUCKETS; i++){
		printf("%03d ", d_scan_in[i]);
	}
	printf("\n");
#endif

#if 0
	// ============================================NEVER DO THIS============================================================================
	float *cuda_timers;
	cudaMalloc((void ** )&cuda_timers, sizeof(float) * tid_count);
	clock_t tic, toc;
	TIMER_TIC

	TIMER_TOC(tid_this);
	// ============================================NEVER DO END============================================================================
#endif

	// prefix inclusive sum scan
	cudaMalloc((void ** )&d_scan_out, sizeof(unsigned int)*NUM_THREADS*NUM_BUCKETS);
	if(d_scan_out == NULL){ printf("ERROR: d_scan_out cudaMalloc failed.\n"); return;}
	d_temp_storage = NULL;
	temp_storage_bytes = 0;
	DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_scan_in, d_scan_out, NUM_THREADS*NUM_BUCKETS);
//	printf("Scan temp_storage %d\n", temp_storage_bytes);
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	if(d_temp_storage == NULL){ printf("ERROR: d_temp_storage cudaMalloc failed.\n"); return;}
	DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_scan_in, d_scan_out, NUM_THREADS*NUM_BUCKETS);
	err = cudaGetLastError();
	if (err != cudaSuccess){ printf("ERROR: DeviceScan::InclusiveSum failed due to err code %d.\n", err); return;}

	cudaDeviceSynchronize();
	cudaFree((void *) d_temp_storage);
#if 0
//	printf("Result of scan with NUM_THREADS %d for bit_beg %d:\n", NUM_THREADS, bitnum_beg[0]);
//	for(int i = 0; i < NUM_THREADS; i++){
//		printf("TID %d: ", i);
//		for(int j = 0; j < NUM_BUCKETS; j++){
//			if((j*NUM_THREADS+i) > NUM_THREADS*NUM_BUCKETS)
//				printf("BAD ");
//			else
//				printf("%03d ", d_scan_out[j*NUM_THREADS+i]);
//		}
//		printf("\n");
//	}
//	printf("\n");
	printf("CHECKING MEMORY d_scan_out data NUM_THREADS %d for bit_beg %d:\n", NUM_THREADS, bitnum_beg[0]);
	for(int i = 0; i < NUM_THREADS*NUM_BUCKETS; i++){
		printf("%03d ", d_scan_out[i]);
	}
	printf("\n");

#endif


	// --------------------------------------------------------------------------------------------------------
	// Bucket/move
	moveKernel<Key, BLOCK_THREADS, ITEMS_PER_THREAD, NUM_BUCKETS><<<GRID_SIZE, BLOCK_THREADS>>>(l_in, l_out, d_scan_out, l_num_elems, bitnum_beg[0]);
	err = cudaGetLastError();
	if (err != cudaSuccess){ printf("ERROR: moveKernel failed due to err code %d.\n", err); return;}
	cudaDeviceSynchronize();
#if 0
	printf("Moved with NUM_THREADS %d for bit_beg %d:\n", NUM_THREADS, bitnum_beg[0]);
	for (int i = 0; i < l_num_elems; i++) {
		printf("%03d ", l_out[i]);
	}
	printf("\n");

	printf("Shifted op for bit_beg %d:\n", bitnum_beg[0]);
	for (int i = 0; i < l_num_elems; i++) {
		printf("%03d ", bucketnum<Key, NUM_BUCKETS>(l_out[i], bitnum_beg[0]));
	}
	printf("\n");
#endif

	// --------------------------------------------------------------------------------------------------------
	// Recurse
	unsigned int l_bucket_scan[NUM_BUCKETS] = {0};									// for the buckets get the last element of d_scan_out
	cudaMalloc((void ** )&d_next_num_elems, sizeof(unsigned int)*NUM_BUCKETS);		// to get this for each value in l_bucket_scan subtract the previous value from it except for the first
	cudaMalloc((void ** )&d_next_offsets, sizeof(unsigned int)*NUM_BUCKETS); 		// for this convert the l_bucket_scan from inclusive to exclusive
	cudaMalloc((void ** )&d_next_bitnum_beg, sizeof(unsigned int));					// reduce the bitnum_beg by log2<num_buckets>:value-1

	if(d_next_num_elems == NULL){ printf("ERROR: d_next_num_elems cudaMalloc failed.\n"); return;}
	if(d_next_offsets == NULL){ printf("ERROR: d_next_offsets cudaMalloc failed.\n"); return;}
	if(d_next_bitnum_beg == NULL){ printf("ERROR: d_next_bitnum_beg cudaMalloc failed.\n"); return;}

	err = cudaGetLastError();
	if (err != cudaSuccess){ printf("ERROR: mallocs for d_next_* failed due to err code %d.\n", err); return;}

	for(int j = 0; j < NUM_BUCKETS; j++){
		l_bucket_scan[j] = d_scan_out[j*NUM_THREADS+(NUM_THREADS-1)];
		if(j == 0){
			d_next_num_elems[j] = l_bucket_scan[j];
			d_next_offsets[j] = 0;
		}else{
			d_next_num_elems[j] = l_bucket_scan[j] - l_bucket_scan[j-1];
			d_next_offsets[j] = l_bucket_scan[j-1];
		}
	}
	d_next_bitnum_beg[0] = bitnum_beg[0] - Log2<NUM_BUCKETS>::VALUE;

	// Free all but the next element arrays
	cudaFree((void *) d_scan_in);
	cudaFree((void *) d_scan_out);

#if 0
	printf("d_next_num_elems[%d] for bitnum_beg %d for threadIdx.x %d:\n", NUM_BUCKETS, bitnum_beg[0], threadIdx.x);
	for (int i = 0; i < NUM_BUCKETS; i++) {
		printf("%d ", d_next_num_elems[i]);
	}
	printf("\n");
	printf("d_next_offsets[%d] for bitnum_beg %d:\n", NUM_BUCKETS, bitnum_beg[0]);
	for (int i = 0; i < NUM_BUCKETS; i++) {
		printf("%d ", d_next_offsets[i]);
	}
	printf("\n");
	printf("bitnum_beg_curr %d, d_next_bitnum_beg %d\n\n", bitnum_beg[0], d_next_bitnum_beg[0]);
#endif

	//return;
	// actually recurse by calling multiple kernels - the l_in and l_out buffers ought to be reversed for the next iteration
	radixSort<Key, BLOCK_THREADS, ITEMS_PER_THREAD, NUM_BUCKETS><<<1, NUM_BUCKETS>>>(l_out, l_in, d_next_num_elems, d_next_offsets, d_next_bitnum_beg);
	err = cudaGetLastError();
	if (err != cudaSuccess){ printf("ERROR: radixSort failed due to err code %d.\n", err); return;}
	cudaDeviceSynchronize();
	// The result will be l_in now. It needs to be moved back into l_out
	copyKernel<Key, BLOCK_THREADS, ITEMS_PER_THREAD, NUM_BUCKETS><<<GRID_SIZE, BLOCK_THREADS>>>(l_in, l_out, l_num_elems);
	err = cudaGetLastError();
	if (err != cudaSuccess){ printf("ERROR: copyKernel failed due to err code %d.\n", err); return;}
	cudaDeviceSynchronize();
	return;
}


__host__ void reverseFill(int *h_in, int array_size) {
	for (int i = 0; i < array_size; i++) {
		h_in[i] = array_size - i;
	}
}

__host__ void randomFill(int *h_in, int array_size) {
	std::default_random_engine generator;
	std::uniform_int_distribution<int> distribution(1, INT_MAX);
	for (int i = 0; i < array_size; i++) {
		h_in[i] = distribution(generator);
	}
}


int main(void)
{
	/*
	const int ARRAY_SIZE = 1 << 9;
	const int BLOCK_NUM_THREAD = 8;
	const int ITEMS_PER_THREAD = 8;
	*/
	 // 128, 128, 1024 takes long - it works.

	const int BLOCK_NUM_THREAD = 128;
	const int ITEMS_PER_THREAD = 128;
	const int ARRAY_SIZE = BLOCK_NUM_THREAD*ITEMS_PER_THREAD*128;
//	const int BLOCK_NUM_THREAD = 32;
//	const int ITEMS_PER_THREAD = 8388608;
//	const int ARRAY_SIZE = BLOCK_NUM_THREAD*ITEMS_PER_THREAD*1;

	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);
//	const int GRID_SIZE = ARRAY_SIZE / (BLOCK_NUM_THREAD * ITEMS_PER_THREAD)+ 1;
//	const int ARRAY_SIZE_PADDED = GRID_SIZE * BLOCK_NUM_THREAD * ITEMS_PER_THREAD;
//	const int ARRAY_BYTES_PADDED = ARRAY_SIZE_PADDED * sizeof(int);
	cudaError_t err;

	int *d_in;
	int *d_out;
	unsigned int *d_num_elems, *d_offsets, *d_bitnum_beg;

	// SETUP DEVICE CONFIG PARAMS
	size_t size;
//	CUDA_CHECK_RETURN(cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize));
//	printf("cudaLimitMallocHeapSize before: %d\n", size);
//	CUDA_CHECK_RETURN(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 512 * 1024 * 1024));
//	CUDA_CHECK_RETURN(cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize));
//	printf("cudaLimitMallocHeapSize after: %d\n", size);

	CUDA_CHECK_RETURN(cudaDeviceGetLimit(&size, cudaLimitDevRuntimeSyncDepth));
	printf("cudaLimitDevRuntimeSyncDepth before: %d\n", size);
	CUDA_CHECK_RETURN(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 8));
	CUDA_CHECK_RETURN(cudaDeviceGetLimit(&size, cudaLimitDevRuntimeSyncDepth));
	printf("cudaLimitDevRuntimeSyncDepth after: %d\n", size);

	CUDA_CHECK_RETURN(cudaDeviceGetLimit(&size, cudaLimitDevRuntimePendingLaunchCount));
	printf("cudaLimitDevRuntimePendingLaunchCount before: %d\n", size);
//	CUDA_CHECK_RETURN(cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 16384));
//	CUDA_CHECK_RETURN(cudaDeviceGetLimit(&size, cudaLimitDevRuntimePendingLaunchCount));
//	printf("cudaLimitDevRuntimePendingLaunchCount after: %d\n", size);

	// generate inp array on host
	int h_in[ARRAY_SIZE];
	int h_out[ARRAY_SIZE];
	int h_bitnum_beg[1] = {(sizeof(int) * 8) - 1}; //TEST {9};;
	randomFill(h_in, ARRAY_SIZE);
	//reverseFill(h_in, ARRAY_SIZE);

	//std::cout<<"Starting mallocs and memsets..."<<std::endl;
	dr::echo << "Starting mallocs and memsets..." << std::endl;
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&d_in, ARRAY_BYTES));
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&d_out, ARRAY_BYTES));
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&d_num_elems, sizeof(unsigned int)));
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&d_offsets, sizeof(unsigned int)));
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&d_bitnum_beg, sizeof(unsigned int)));

	CUDA_CHECK_RETURN(cudaMemset((void * )d_in, 0, ARRAY_BYTES));
	CUDA_CHECK_RETURN(cudaMemcpy((void * )d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy((void * )d_num_elems, &ARRAY_SIZE, sizeof(unsigned int), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemset((void * )d_offsets, 0, sizeof(unsigned int)));
	CUDA_CHECK_RETURN(cudaMemcpy((void * )d_bitnum_beg, h_bitnum_beg, sizeof(unsigned int), cudaMemcpyHostToDevice));
	//std::cout<<"Done mallocs and memsets..."<<std::endl;
	dr::echo << "Done mallocs and memsets..." << std::endl;

#if 0
	printf("Input:\n");
	for (int i = 0; i < ARRAY_SIZE; i++) {
		printf("%04d ", h_in[i]);
	}
	printf("\n");
#endif
	err = cudaGetLastError();
	if (err != cudaSuccess){ printf("ERROR: host failed before calling radixsort due to err code %d.\n", err); return -1;}

	dr::echo << "RadixSort starting for " << ARRAY_SIZE<<" elements...."<< std::endl;
	radixSort<int, 128, 1024, 16><<<1, 1>>>(d_in, d_out, d_num_elems, d_offsets, d_bitnum_beg);
	err = cudaGetLastError();
	if (err != cudaSuccess){ printf("ERROR: radixSort failed due to err code %d.\n", err); return -1;}
	//CudaCheckError();
	cudaDeviceSynchronize();
	dr::echo << "RadixSort ended...." << std::endl;


	CUDA_CHECK_RETURN(cudaMemcpy(h_out, (void * )d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost));
#if 0
	printf("Output:\n");
	for(int i = 0; i < ARRAY_SIZE; i++) {
		printf("%04d ", h_out[i]);
	}
	printf("\n");
#endif

	dr::echo << "Simple Sort starting...." << std::endl;
	std::vector<int> myvector(h_in, h_in + ARRAY_SIZE);
	std::sort(myvector.begin(), myvector.end());
	dr::echo << "Simple Sort ended...." << std::endl;
	int differences = 0;
	for (int i = 0; i < ARRAY_SIZE; i++) {
		if (myvector[i] != h_out[i])
			differences++;
	}
	if (differences == 0)
		printf("PASS - The arrays of size %d sorted by std::sort and our sort are equal.\n", ARRAY_SIZE);
	else
		printf("FAIL - There are %d differences of %d elements in our sort.\n", differences, ARRAY_SIZE);


#if 1
	CUDA_CHECK_RETURN(cudaMemcpy((void * )d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemset((void * )d_out, 0, ARRAY_BYTES));

	DoubleBuffer<int> d_keys((int *)d_in, (int *)d_out);
	void *d_temp_storage = NULL;
	size_t temp_storage_bytes = 0;
	//DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, total_num_items);
	// TODO: Do we include the padding into the sort to replace the above line???
	dr::echo<<"DeviceRadixSort starting..."<<std::endl;
	CubDebugExit(DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, ARRAY_SIZE));
	CubDebugExit(cudaMalloc(&d_temp_storage, temp_storage_bytes));
	//DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, total_num_items);
	CubDebugExit(DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, ARRAY_SIZE));
	d_out = d_keys.Current();
	cudaDeviceSynchronize();
	dr::echo<<"DeviceRadixSort ending..."<<std::endl;
	CUDA_CHECK_RETURN(cudaMemcpy(h_out, (void *)d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost));

	differences = 0;
	for(int i = 0; i < ARRAY_SIZE; i++) {
		if(myvector[i] != h_out[i])
		differences++;
	}
	if(differences == 0)
		printf("PASS - The arrays sorted by std::sort and cub::DeviceRadixSort are equal.\n");
	else
		printf("FAIL - There are %d differences of %d elements in cub::DeviceRadixSort\n", differences, ARRAY_SIZE);
#endif
	return 0;
}

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}

