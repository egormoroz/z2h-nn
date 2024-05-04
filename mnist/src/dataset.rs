use std::fs::File;
use std::io::{self, BufReader, Read};

pub struct Dataset {
    pub labels: Vec<u8>,
    pub images: Vec<[u8; 28*28]>,
}

#[derive(Debug)]
pub enum LoadError {
    IO(io::Error),
    IMagicMismatch,
    LMagicMismatch,
}

impl Dataset {
    pub fn new(im_path: &str, lbl_path: &str) -> Result<Self, LoadError> {
        let images = read_images(im_path)?;
        let labels = read_labels(lbl_path)?;
        assert_eq!(images.len(), labels.len());
        Ok(Self { images, labels })
    }
}

fn read_labels(lbl_path: &str) -> Result<Vec<u8>, LoadError> {
    let file = File::open(lbl_path).map_err(LoadError::IO)?;
    let mut reader = BufReader::new(file);
    let mut buf = [0; 4];

    reader.read_exact(&mut buf).map_err(LoadError::IO)?;
    if u32::from_be_bytes(buf) != 2049 { return Err(LoadError::LMagicMismatch); }

    reader.read_exact(&mut buf).map_err(LoadError::IO)?;
    let n_items = u32::from_be_bytes(buf);
    let mut items = Vec::with_capacity(n_items as usize);

    for _ in 0..n_items {
        reader.read_exact(&mut buf[..1]).map_err(LoadError::IO)?;
        items.push(buf[0]);
    }

    Ok(items)
}

fn read_images(im_path: &str) -> Result<Vec<[u8; 28*28]>, LoadError> {
    let file = File::open(im_path).map_err(LoadError::IO)?;
    let mut reader = BufReader::new(file);
    let mut buf = [0; 4];

    reader.read_exact(&mut buf).map_err(LoadError::IO)?;
    if u32::from_be_bytes(buf) != 2051 { return Err(LoadError::IMagicMismatch); }

    reader.read_exact(&mut buf).map_err(LoadError::IO)?;
    let n_items = u32::from_be_bytes(buf);

    reader.read_exact(&mut buf).map_err(LoadError::IO)?;
    let n_rows = u32::from_be_bytes(buf);
    reader.read_exact(&mut buf).map_err(LoadError::IO)?;
    let n_cols = u32::from_be_bytes(buf);

    assert_eq!(n_rows, 28);
    assert_eq!(n_cols, 28);

    let mut items = Vec::with_capacity(n_items as usize);
    let mut im = [0; 28*28];
    for _ in 0..n_items {
        reader.read_exact(&mut im).map_err(LoadError::IO)?;
        items.push(im);
    }

    Ok(items)
}

