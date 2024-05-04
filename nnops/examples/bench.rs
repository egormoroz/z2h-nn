/// Even with all this stupid bullshit, we stil are only 75% of Intel MKL sgemm.
/// And for large matrices we still suck somehow, despite blocking...
/// I don't think it's sensible to worry about perfomance of ops at this point.


use std::time::Instant;

use core::arch::x86_64::*;

struct XorShift(u64);

#[repr(C, align(32))]
struct F32x8([f32; 8]);

impl XorShift {
    fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;

        let x = self.0.overflowing_mul(0x2545F4914F6CDD1Du64).0;
        (x >> 32) as u32
    }

    fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 / u32::MAX as f32
    }
}

const RA: usize = 3;
const RB: usize = 4;
const A_STRIDE: usize = RA;
const B_STRIDE: usize = RB * 8;

unsafe fn gemm_inner(
    mut a_ptr: *const f32, 
    mut b_ptr: *const f32, 
    c_ptr: *mut f32, 
    k: usize, m: usize,
    i0: usize, j0: usize)
{
    let mut acc = [_mm256_setzero_ps(); RA*RB];
    for _ in 0..k {
        for j in 0..RB {
            let b_vec = _mm256_load_ps(b_ptr);
            let mut a_slice = a_ptr;
            for i in 0..RA {
                let a_vec = _mm256_set1_ps(*a_slice);
                acc[i*RB + j] = _mm256_fmadd_ps(a_vec, b_vec, acc[i*RB + j]);
                a_slice = a_slice.add(1);
            }
            b_ptr = b_ptr.add(8);
        }
        a_ptr = a_ptr.add(RA);
    }

    for i in 0..RA {
        for j in 0..RB {
            let c_ptr = c_ptr.add((i0 + i)*m + j0 + j*8*RB);
            let c_vec = _mm256_load_ps(c_ptr);
            _mm256_store_ps(c_ptr, _mm256_add_ps(c_vec, acc[i*RB + j]));
        }
    }
}

pub unsafe fn gemm(
    a: &[f32], b: &[f32], c: &mut [f32],
    n: usize, k: usize, m: usize) 
{
    let a_chunk_size = RA*k;
    let b_chunk_size = 8*RB*k;

    let mut a_ = alloc_aligned(a.len());
    let mut b_ = alloc_aligned(b.len());
    let a_ = &mut a_[..];
    let b_ = &mut b_[..];

    permute(&a[..], &b[..], a_, b_, n, k, m);

    let mut a_ptr = a_.as_ptr();
    let b_ptr = b_.as_ptr();

    // perform the A@B product, computing one RAx8*RB block at a time
    for i_block in 0..n / RA {
        let mut b_ptr = b_ptr;
        for j_block in 0..m / (8*RB) {
            gemm_inner(a_ptr, b_ptr, c.as_mut_ptr(), k, m, 
                                 i_block*RA, j_block*RB*8);
            b_ptr = b_ptr.add(b_chunk_size);
        }
        a_ptr = a_ptr.add(a_chunk_size);
    }
}

fn alloc_aligned(n: usize) -> Vec<f32> {
    assert!(n % 8 == 0);

    let mut v: Vec<F32x8> = Vec::with_capacity(n / 8);
    assert!(v.capacity() == n / 8);
    let ptr = v.as_mut_ptr();

    std::mem::forget(v);

    unsafe {
        Vec::from_raw_parts(ptr as *mut f32, n, n)
    }
}

fn permute(
    a: &[f32], b: &[f32], a_: &mut [f32], b_: &mut [f32],
    n: usize, k: usize, m: usize)
{
    let mut idx = 0;
    for i0 in (0..n).step_by(A_STRIDE) {
        for p in 0..k {
            for i in i0..i0+A_STRIDE {
                a_[idx] = a[i * k + p];
                idx += 1;
            }
        }
    }

    idx = 0;
    for j0 in (0..m).step_by(B_STRIDE) {
        for p in 0..k {
            for j in j0..j0+B_STRIDE {
                b_[idx] = b[p * m + j];
                idx += 1;
            }
        }
    }
}

fn main() {
    const N: usize = 768;
    const FLOP: usize = N*N*2*N;

    let mut rng = XorShift(0xdeadbeef);

    let mut a = alloc_aligned(N*N);
    let mut b = alloc_aligned(N*N);

    for ai in a.iter_mut() { *ai = rng.next_f32(); }
    for bi in b.iter_mut() { *bi = rng.next_f32(); }

    let mut c = alloc_aligned(N*N);

    println!("{:?}", a.as_ptr() as usize % 32);
    println!("{:?}", b.as_ptr() as usize % 32);
    println!("{:?}", c.as_ptr() as usize % 32);

    for _ in 0..100 {
        c.fill(0.);
        let start = Instant::now();
        unsafe { gemm(&a[..], &b[..], &mut c[..], N, N, N); }
        let delta = start.elapsed().as_secs_f64();

        let trace = c.iter().step_by(N + 1).sum::<f32>();
        println!("{:.2} GFLOPS, trace {:.4e}", 
                 FLOP as f64 / (delta * 1e9), trace);
    }
}

