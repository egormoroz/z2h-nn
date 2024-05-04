mod dataset;
pub mod net;

extern crate rand;

fn main() {
    let im_path = "data/train-images-idx3-ubyte";
    let lbl_path = "data/train-labels-idx1-ubyte";
    let ds = dataset::Dataset::new(im_path, lbl_path)
        .expect("failed to open dataset");

    println!("{:?}", &ds.labels[..10]);
}
