use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("HDF5 error: {0}")]
    Hdf5(#[from] hdf5::Error),
}
