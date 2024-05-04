use crate::net::XorNet;

pub struct SGD {
    pub lr: f32,
}

impl SGD {
    pub fn new(lr: f32) -> Self {
        Self { lr }
    }

    pub fn step(&self, net: &mut XorNet) {
        apply_grad(&mut net.w0.data, &mut net.w0.grad, self.lr);
        apply_grad(&mut net.b0.data, &mut net.b0.grad, self.lr);
        apply_grad(&mut net.w1.data, &mut net.w1.grad, self.lr);
        apply_grad(&mut net.b1.data, &mut net.b1.grad, self.lr);
    }
}

fn apply_grad(params: &mut [f32], delta: &[f32], lr: f32) {
    for (param, delta) in params.iter_mut().zip(delta.iter()) {
        *param -= lr * delta;
    }
}
