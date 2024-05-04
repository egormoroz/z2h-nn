/// perform a very slow matmul a x b = c: (n, k) x (k, m) = (n, m)
pub fn gemm(
    a: &[f32], b: &[f32], c: &mut [f32],
    n: usize, k: usize, m: usize) 
{
    assert_eq!(a.len(), n * k);
    assert_eq!(b.len(), k * m);
    assert_eq!(c.len(), n * m);

    for i in 0..n {
        for p in 0..k {
            for j in 0..m {
                c[i * m + j] += a[i * k + p] * b[p * m + j];
            }
        }
    }
}

/// transpose a and do matmul a^T x b = c: (k, n)^T x (k, m) = (n, m),
pub fn gemm_at(
    a: &[f32], b: &[f32], c: &mut [f32],
    n: usize, k: usize, m: usize) 
{
    assert_eq!(a.len(), n * k);
    assert_eq!(b.len(), k * m);
    assert_eq!(c.len(), n * m);

    for p in 0..k {
        for i in 0..n {
            for j in 0..m {
                c[i * m + j] += a[p * n + i] * b[p * m + j];
            }
        }
    }
}

/// transpose b and do matmul a x b^T = c: (n, k) x (m, k)^T = (n, m)
pub fn gemm_bt(
    a: &[f32], b: &[f32], c: &mut [f32],
    n: usize, k: usize, m: usize) 
{
    assert_eq!(a.len(), n * k);
    assert_eq!(b.len(), k * m);
    assert_eq!(c.len(), n * m);

    for i in 0..n {
        for j in 0..m {
            for p in 0..k {
                c[i * m + j] += a[i * k + p] * b[j * k + p];
            }
        }
    }
}
