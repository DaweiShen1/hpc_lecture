#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void setbucket(int *bucket)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < 5)
  {
    bucket[i] = 0;
  }
}
__global__ void fillbucket(int *bucket, int *key)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
  if( i < 50)
  {    
    atomicAdd(&bucket[key[i]], 1);
  }
}

__global__ void prefixbucket(int *bucket, int *b)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < 5)
  {
    for(int j = 1; j < 5; j <<= 1)
    {
      b[i] = bucket[i];
      __syncthreads();
      bucket[i] += b[i-j];
      __syncthreads();
    }
  }
}
__global__ void rankbucket(int *bucket, int *key)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < bucket[0])
  {
    key[i] = 0;
  }
  else if( i >= bucket[0]&&i < bucket[1])
  {
    key[i] = 1;
  }
  else if( i >= bucket[1]&&i < bucket[2])
  {
    key[i] = 2;
  }
  else if( i >= bucket[2]&&i < bucket[3])
  {
    key[i] = 3;
  }
  else
  {
    key[i] = 4;
  }
}

int main() {
  int n = 50;
  int range = 5;
  //set variables
  int *key, *bucket, *b;
  //storage allocation
  cudaMallocManaged(&key   , n*sizeof(int));
  cudaMallocManaged(&b     , 5*sizeof(int));
  cudaMallocManaged(&bucket, 5*sizeof(int));

  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  dim3 threadsPerBlock(64);
  int blockNumber = 1;

  setbucket <<<blockNumber,threadsPerBlock>>>(bucket);
  fillbucket<<<blockNumber,threadsPerBlock>>>(bucket, key);
  prefixbucket<<<blockNumber,threadsPerBlock>>>(bucket, b);
  rankbucket<<<blockNumber,threadsPerBlock>>>(bucket, key);
  cudaDeviceSynchronize();

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
