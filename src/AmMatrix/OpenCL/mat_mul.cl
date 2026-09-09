__kernel void mat_mul(__global float *A, __global float *B, __global float *C)
{
  int n = 3, m = 3, l = 3;
  int idx = get_global_id(0);
  int idy = get_global_id(1);
  if (idx < 3 && idy < 3) {
    int i;
    float sum = 0;
    for (i = 0 ; i < n ; ++i) {
      sum += A[idy*n+i] * B[idx+i*n];
    }
    C[idx+idy*n] = sum;
  }
}
