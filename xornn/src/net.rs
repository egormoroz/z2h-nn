use rand::Rng;
use core::fmt;


#[derive(Debug)]
pub struct Tensor<const N: usize> {
    pub data: [f32; N],
    pub grad: [f32; N],
}

impl<const N: usize> Tensor<N> {
    pub fn zero() -> Self {
        Self {
            data: [0.; N],
            grad: [0.; N],
        }
    }

    pub fn uniform(n_out: usize) -> Self {
        let stdv = (n_out as f32).sqrt().recip();
        let dist = rand::distributions::Uniform::new(-stdv, stdv);
        let data = core::array::from_fn(|_|
            rand::thread_rng().sample(dist));

        Self {
            data,
            grad: [0.; N],
        }
    }
}

impl<const N: usize> fmt::Display for Tensor<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor(\n  data={:?},\n  grad={:?}\n)", self.data, self.grad)
    }
}

/// 2 sigmoid-> 2 sigmoid-> 1 MLP arch
pub struct XorNet {
    pub w0: Tensor<4>,
    pub b0: Tensor<2>,

    pub w1: Tensor<2>,
    pub b1: Tensor<1>,

    pub out: Vec<f32>,
    pub bs: usize,

    inp0: Vec<f32>,
    inp1: Vec<f32>,

    // two handy buffers for backprop
    grad_buf1: Vec<f32>,
    grad_buf2: Vec<f32>,
}

impl XorNet {
    pub fn new() -> Self {
        Self {
            w0: Tensor::uniform(2),
            b0: Tensor::uniform(2),

            w1: Tensor::uniform(1),
            b1: Tensor::zero(),

            out: vec![],
            bs: 0,

            inp0: vec![],
            inp1: vec![],
            grad_buf1: vec![],
            grad_buf2: vec![],
        }
    }

    pub fn forward(&mut self, x: &[f32], bs: usize) {
        assert_eq!(x.len(), bs * 2);

        if self.bs != bs {
            self.resize_buffers(bs);
        }

        self.inp0.copy_from_slice(x);
        nnops::linear_forward(
            &self.inp0[..], 
            &mut self.inp1[..], 
            &self.w0.data, &self.b0.data, 
            bs, 2, 2);
        nnops::sigmoid_forward_(&mut self.inp1[..]);

        nnops::linear_forward(
            &self.inp1[..],
            &mut self.out[..],
            &self.w1.data, &self.b1.data,
            bs, 2, 1);
        nnops::sigmoid_forward_(&mut self.out[..]);
    }

    pub fn backward(&mut self, target: &[f32]) {
        assert_eq!(target.len(), self.bs);

        self.grad_buf1.resize(self.bs, 0.);
        nnops::mse_grad(&self.out[..], &target, &mut self.grad_buf1[..]);

        nnops::sigmoid_backward_(&self.out[..], &mut self.grad_buf1[..]);
        self.grad_buf2.resize(self.inp1.len(), 0.);
        nnops::linear_backward(
            &self.inp1[..], 
            &self.w1.data, &self.b1.data, 
            &self.grad_buf1[..], 
            &mut self.w1.grad, &mut self.b1.grad, 
            Some(&mut self.grad_buf2[..]), 
            self.bs, 2, 1);

        nnops::sigmoid_backward_(&self.inp1[..], &mut self.grad_buf2[..]);
        nnops::linear_backward(
            &self.inp0[..], 
            &self.w0.data, &self.b0.data, 
            &self.grad_buf2[..], 
            &mut self.w0.grad, &mut self.b0.grad, 
            None, 
            self.bs, 2, 2);
    }

    pub fn zero_grad(&mut self) {
        self.w0.grad.fill(0.);
        self.b0.grad.fill(0.);
        self.w1.grad.fill(0.);
        self.b1.grad.fill(0.);
    }

    fn resize_buffers(&mut self, bs: usize) {
        self.inp0.resize(bs * 2, 0.);
        self.inp1.resize(bs * 2, 0.);
        self.out.resize(bs * 1, 0.);
        self.bs = bs;
    }
}

