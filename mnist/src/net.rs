use rand::Rng;
use nnops;

#[derive(Debug)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub grad: Vec<f32>,
}

impl Tensor {
    pub fn uniform(len: usize, n_out: usize) -> Self {
        let stdv = (n_out as f32).sqrt().recip();
        let dist = rand::distributions::Uniform::new(-stdv, stdv);
        Self {
            data: rand::thread_rng().sample_iter(dist).take(len).collect(),
            grad: vec![0.; len]
        }
    }
    pub fn zeros(len: usize) -> Self {
        Self {
            data: vec![0.; len],
            grad: vec![0.; len],
        }
    }
}

struct Linear {
    weight: Tensor,
    bias: Tensor,

    inp: Vec<f32>,

    n_in: usize,
    n_out: usize,
}

impl Linear {
    fn new(n_in: usize, n_out: usize) -> Self {
        Self {
            weight: Tensor::uniform(n_in * n_out, n_out),
            bias: Tensor::zeros(n_out),
            inp: vec![],
            n_in, n_out
        }
    }

    fn forward(&mut self, x_in: &[f32], x_out: &mut [f32]) {
        assert_eq!(x_in.len() % self.n_in, 0);
        let bs = x_in.len() / self.n_in;
        assert_eq!(x_out.len() % self.n_out, 0);
        assert_eq!(x_out.len() / self.n_out, bs);

        if self.inp.len() != x_in.len() { self.inp.resize(x_in.len(), 0.); }
        self.inp.copy_from_slice(x_in);

        nnops::linear_forward(
            x_in, x_out, 
            &self.weight.data[..], &self.bias.data[..], 
            bs, self.n_in, self.n_out);
    }

    fn backward(&mut self, grad: &[f32], grad_in: Option<&mut [f32]>) {
        assert_eq!(grad.len() % self.n_out, 0);
        let bs = grad.len() / self.n_out;
        assert_eq!(self.inp.len(), bs * self.n_in);
        if let Some(grad_in) = &grad_in {
            assert_eq!(grad_in.len() % self.n_in, 0);
            assert_eq!(grad_in.len() / self.n_in, bs);
        }

        nnops::linear_backward(
            &self.inp[..], 
            &self.weight.data[..], &self.bias.data[..],
            grad, 
            &mut self.weight.grad[..], &mut self.bias.grad[..], 
            grad_in, 
            bs, self.n_in, self.n_out)
    }

    fn zero_grad(&mut self) {
        self.weight.grad.fill(0.);
        self.bias.grad.fill(0.);
    }
}

const N_IN: usize = 768;
const N_OUT: usize = 10;

pub struct Perceptron {
    l0: Linear,

    pub out: Vec<f32>,
    bs: usize,

    grad_buf: Vec<f32>,
}

impl Perceptron {
    pub fn new() -> Self {
        Self {
            l0: Linear::new(N_IN, N_OUT),
            out: vec![],
            bs: 0,
            grad_buf: vec![],
        }
    }

    pub fn forward(&mut self, x: &[f32], bs: usize) {
        assert_eq!(x.len(), bs * N_IN);
        if self.bs != bs {
            self.out.resize(bs, 0.);
            self.bs = bs;
        }
        self.l0.forward(x, &mut self.out[..]);
        // TODO: this should be pure logits (or softmax during inference)
        nnops::sigmoid_forward_(&mut self.out[..]);
    }

    // onehot vectors for mse 
    // TODO: softmax and crossentropy loss (fused)
    pub fn backward(&mut self, target: &[f32]) {
        assert_eq!(target.len(), self.bs * N_OUT);
        self.grad_buf.resize(self.bs, 0.);

        nnops::mse_grad(&self.out[..], &target, &mut self.grad_buf[..]);

        nnops::sigmoid_backward_(&self.out[..], &mut self.grad_buf[..]);
        self.l0.backward(&self.grad_buf[..], None);
    }

    pub fn zero_grad(&mut self) {
        self.l0.zero_grad();
    }
}

