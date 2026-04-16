use std::sync::Mutex;

use once_cell::sync::OnceCell;
use ort::{execution_providers::CUDAExecutionProvider, session::Session};

pub static NN_SESSION: OnceCell<Mutex<Session>> = OnceCell::new();

pub fn load_model() {
    let model_path = std::env::var("CHESS_MODEL_PATH").unwrap_or_else(|_| "model.onnx".into());
    if !std::path::Path::new(&model_path).exists() {
        eprintln!(
            "[engine] No model at '{}'. Expected a pre-trained ONNX model to be mounted for the Rust server.",
            model_path
        );
        return;
    }

    let session = Session::builder()
        .expect("ort SessionBuilder")
        .with_execution_providers([CUDAExecutionProvider::default().build()])
        .expect("execution providers")
        .commit_from_file(&model_path)
        .expect("load model");

    NN_SESSION.set(Mutex::new(session)).ok();
    eprintln!("[engine] Neural model loaded from '{}'", model_path);
}
