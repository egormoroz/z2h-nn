pub mod gemm;

use gemm::*;

pub fn linear_forward(
    x_in: &[f32], x_out: &mut [f32], 
    weight: &[f32], bias: &[f32], 
    bs: usize, n_in: usize, n_out: usize) 
{
    assert_eq!(x_in.len(), bs * n_in);
    assert_eq!(x_out.len(), bs * n_out);

    assert_eq!(weight.len(), n_in * n_out);
    assert_eq!(bias.len(), n_out);

    for i_sample in 0..bs {
        let x_out = &mut x_out[i_sample*n_out..(i_sample + 1)*n_out];
        x_out.copy_from_slice(bias);
    }

    gemm(x_in, weight, x_out, bs, n_in, n_out);
}

pub fn linear_backward(
    x_in: &[f32],
    weight: &[f32], bias: &[f32],
    grad: &[f32],
    weight_grad: &mut [f32], bias_grad: &mut [f32], grad_in: Option<&mut [f32]>,
    bs: usize, n_in: usize, n_out: usize)
{
    assert_eq!(x_in.len(), bs * n_in);
    assert_eq!(grad.len(), bs * n_out);

    assert_eq!(weight.len(), n_in * n_out);
    assert_eq!(weight_grad.len(), n_in * n_out);
    assert_eq!(bias.len(), n_out);
    assert_eq!(bias_grad.len(), n_out);

    // sum the gradients along the batch axis
    for i in 0..bs {
        for j in 0..n_out {
            bias_grad[j] += grad[i * n_out + j];
        }
    }

    gemm_at(x_in, grad, weight_grad, n_in, bs, n_out);
    if let Some(grad_in) = grad_in {
        assert_eq!(grad_in.len(), bs * n_in);
        gemm_bt(grad, weight, grad_in, bs, n_out, n_in);
    }
}

pub fn sigmoid_forward_(x: &mut [f32]) {
    for xi in x.iter_mut() {
        *xi = 1. / (1. + f32::exp(-*xi));
    }
}

pub fn sigmoid_backward_(x_out: &[f32], grad: &mut [f32]) {
    assert_eq!(x_out.len(), grad.len());
    for (grad, y) in grad.iter_mut().zip(x_out) {
        *grad *= y * (1. - y);
    }
}

pub fn mse_loss(x_pred: &[f32], x_target: &[f32]) -> f32 {
    let bs = x_pred.len();
    assert_eq!(bs, x_target.len());

    x_pred.iter().zip(x_target.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f32>() / bs as f32
}

pub fn mse_grad(x_pred: &[f32], x_target: &[f32], grad: &mut [f32]) {
    let bs = x_pred.len();
    assert_eq!(bs, x_target.len());
    assert_eq!(bs, grad.len());

    for i in 0..bs {
        grad[i] += (x_pred[i] - x_target[i]) * 2. / bs as f32;
    }
}
