#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <cassert>

template<typename RandomAccessIterator, typename BinaryFunction>
inline __device__
void blockwise_inplace_inclusive_scan(RandomAccessIterator first, BinaryFunction op)
{
  typename thrust::iterator_value<RandomAccessIterator>::type x = first[threadIdx.x];

  for(unsigned int offset = 1; offset < blockDim.x; offset *= 2)
  {
    if(threadIdx.x >= offset)
    {
      x = op(first[threadIdx.x - offset], x);
    }

    __syncthreads();

    first[threadIdx.x] = x;

    __syncthreads();
  }
}


template<typename RandomAccessIterator, typename Size, typename BinaryFunction>
inline __device__ 
void blockwise_inplace_small_inclusive_scan(RandomAccessIterator first, Size n, BinaryFunction op)
{
  typename thrust::iterator_value<RandomAccessIterator>::type x;

  if(threadIdx.x < n)
  {
    x = first[threadIdx.x];
  }

  for(Size offset = 1; offset < n; offset *= 2)
  {
    if(threadIdx.x >= offset)
    {
      x = op(first[threadIdx.x - offset], x);
    }

    __syncthreads();

    first[threadIdx.x] = x;

    __syncthreads();
  }
}


template<typename RandomAccessIterator, typename Size, typename BinaryFunction>
inline __device__ 
void blockwise_inplace_inclusive_scan(RandomAccessIterator first, Size n, BinaryFunction op)
{
  blockwise_inplace_small_inclusive_scan(first, min(blockDim.x, n), op);

  RandomAccessIterator last = first + n;
  for(first += blockDim.x; first < last; first += blockDim.x, n -= blockDim.x)
  {
    // sum the previous iteration's carry
    if(threadIdx.x == 0)
    {
      *first = op(*(first-1), *first);
    }

    __syncthreads();

    blockwise_inplace_small_inclusive_scan(first, min(blockDim.x, n), op);
  }
}


__global__ void inplace_scan(int *x, int n)
{
  blockwise_inplace_inclusive_scan(x, n, thrust::plus<int>());
}


int main()
{
  thrust::host_vector<size_t> sizes;
  sizes.push_back(0);
  sizes.push_back(1);
  sizes.push_back(9);
  sizes.push_back(31);
  sizes.push_back(32);
  sizes.push_back(33);
  sizes.push_back(512);
  sizes.push_back(1024 + 1);
  sizes.push_back(1 << 20);
  sizes.push_back(16 << 20);

  for(int block_size = 32; block_size <= 512; block_size += 32)
  {
    std::cout << "testing block_size " << block_size << std::endl;

    for(int i = 0; i < sizes.size(); ++i)
    {
      size_t n = sizes[i];

      std::cout << "testing size " << n << std::endl;

      thrust::device_vector<int> vec1(n, 1), vec2(n, 1);

      thrust::inclusive_scan(vec1.begin(), vec1.end(), vec1.begin());

      inplace_scan<<<1,block_size>>>(vec2.data().get(), vec2.size());

      if(n < 50)
      {
        std::cout << "result: ";
        thrust::copy(vec2.begin(), vec2.end(), std::ostream_iterator<int>(std::cout, " "));
        std::cout << std::endl;
      }

      assert(vec1 == vec2);
    }
  }

  return 0;
}

