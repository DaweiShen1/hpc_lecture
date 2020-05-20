#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N], a[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  __m256 xvec = _mm256_load_ps(x);
  __m256 yvec = _mm256_load_ps(y);
  __m256 mvec = _mm256_load_ps(m);
  for(int i=0; i<N; i++) {
    __m256 ivec = _mm256_set1_ps(i);
    __m256 xivec= _mm256_set1_ps(x[i]);
    __m256 yivec= _mm256_set1_ps(y[i]);

    for(int j=0; j<N; j++)
		a[j] = j;  	
    
		__m256 jvec  = _mm256_load_ps(a);
		__m256 mask  = _mm256_cmp_ps(jvec, ivec, _CMP_NEQ_OQ);

		__m256 xjvec = _mm256_setzero_ps();
		__m256 yjvec = _mm256_setzero_ps();
		__m256 mjvec = _mm256_setzero_ps();

		xjvec = _mm256_blendv_ps(xjvec, xvec, mask);
		yjvec = _mm256_blendv_ps(yjvec, yvec, mask);
		mjvec = _mm256_blendv_ps(mjvec, mvec, mask);

		__m256 rxvec = _mm256_sub_ps(xivec, xjvec);
		__m256 ryvec = _mm256_sub_ps(yivec, yjvec);
		__m256 rxsvec = _mm256_mul_ps(rxvec, rxvec);
		__m256 rysvec = _mm256_mul_ps(ryvec, ryvec);
		__m256 r1vec = _mm256_add_ps(rxsvec, rysvec);
		__m256 rvec  = _mm256_sqrt_ps(r1vec);

		__m256 rxmvec = _mm256_mul_ps(rxvec, mjvec);
		__m256 rymvec = _mm256_mul_ps(ryvec, mjvec);
		__m256 r3vec = _mm256_mul_ps(r1vec, rvec);
		__m256 fx1vec = -_mm256_div_ps(rxmvec, r3vec);
		__m256 fy1vec = -_mm256_div_ps(rymvec, r3vec);

		__m256 fxvec = _mm256_permute2f128_ps(fx1vec,fx1vec,1);
		fxvec = _mm256_add_ps(fxvec,fx1vec);
		fxvec = _mm256_hadd_ps(fxvec,fxvec);
		fxvec = _mm256_hadd_ps(fxvec,fxvec);

		__m256 fyvec = _mm256_permute2f128_ps(fy1vec,fy1vec,1);
		fyvec = _mm256_add_ps(fyvec,fy1vec);
		fyvec = _mm256_hadd_ps(fyvec,fyvec);
		fyvec = _mm256_hadd_ps(fyvec,fyvec);

		_mm256_store_ps(fx, fxvec);
		_mm256_store_ps(fy, fyvec);

		printf("%d %g %g\n",i,fx[i],fy[i]);
    }
}
