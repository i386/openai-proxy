//! OpenAI-compatible proxy server
//! This crate provides an OpenAI-compatible HTTP proxy that routes requests
//! to llama.cpp inference servers.

pub mod backend;
pub mod llama;
pub mod server;
pub mod types;

pub use backend::{BackendConfig, BackendError, ErrorType, InferenceBackend};
pub use llama::LlamaBackend;
pub use server::run_proxy;
pub use types::*;
