//! Simple example of running the OpenAI proxy
//!
//! Run with: cargo run --example simple_proxy
//!
//! This will start a proxy server on port 8080 that forwards
//! requests to a llama.cpp server running on port 8081

use openai_proxy::{
    run_proxy, LlamaBackend, BackendConfig, InferenceBackend,
};
use std::sync::Arc;

#[tokio::main]
async fn main() {
    // Parse command line arguments or use defaults
    let proxy_port = std::env::var("PROXY_PORT")
        .unwrap_or_else(|_| "8080".to_string())
        .parse()
        .expect("Invalid proxy port");

    let llama_url = std::env::var("LLAMA_URL")
        .unwrap_or_else(|_| "http://localhost:8081".to_string());

    println!("Starting OpenAI Proxy...");
    println!("  Proxy port: {}", proxy_port);
    println!("  Llama server: {}", llama_url);

    // Configure the llama.cpp backend
    let config = BackendConfig {
        url: llama_url,
        model: "llama-3.1-8b".to_string(),
        models: vec!["llama-3.1-8b".to_string()],
    };

    // Create the backend and wrap in Arc for sharing
    let backend = Arc::new(LlamaBackend::new(config));

    // Run the health check first
    println!("Checking backend health...");
    match backend.health().await {
        Ok(_) => println!("  Backend is healthy!"),
        Err(e) => {
            eprintln!("  Warning: Backend health check failed: {}", e.message);
            eprintln!("  Continuing anyway...");
        }
    }

    // Start the proxy server
    println!("Starting proxy server...");
    if let Err(e) = run_proxy(backend, proxy_port).await {
        eprintln!("Error running proxy: {}", e);
    }
}
