extern crate rand;

use nnops;
mod net;
mod xnopt;


fn main() {
    // rand::thread_rng().

    let bs = 4;
    let xb = [
        0., 0.,
        0., 1.,
        1., 0.,
        1., 1.,
    ];
    let yb = [
        0.,
        1.,
        1.,
        0.,
    ];

    let mut model = net::XorNet::new();
    let opt = xnopt::SGD::new(1.);

    for step in 0..10000 {
        model.forward(&xb, bs);
        model.zero_grad();
        model.backward(&yb);

        opt.step(&mut model);

        if step % 100 == 0 {
            let loss = nnops::mse_loss(&model.out[..], &yb);
            let acc = model.out.iter().zip(yb.iter())
                .filter(|(p, t)| p.round() == t.round())
                .count() as f32 / bs as f32;
            println!("step {} loss {:.4} acc {:.4}", step, loss, acc);

            if acc == 1.0 { break; }
        }
    }
}
