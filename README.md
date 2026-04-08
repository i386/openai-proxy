# openai-proxy

OpenAI-compatible proxy for different inference backends (llama.cpp, MLX, vLLM, etc.)

This crate provides a way to serve OpenAI-compatible API responses using various inference backends, mapping their responses to match the OpenAI API format exactly as specified in `openapi.with-code-samples.yml`.

## Features

- **llama.cpp backend**: Connect to llama.cpp server instances
- **MLX backend**: Support for Apple Silicon MLX inference
- **vLLM backend**: Integration with vLLM servers
- **Custom backends**: Easy trait for implementing your own backend
- **Error handling**: OpenAI-compatible error responses
- **Streaming support**: Server-Sent Events (SSE) for streaming responses

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
openai-proxy = { version = "0.1", features = ["llama-backend", "mlx-backend"] }
```

## Quick Start

```rust
use openai_proxy::{run_proxy, LlamaBackend, BackendConfig};
use std::sync::Arc;

#[tokio::main]
async fn main() {
    // Configure the backend
    let config = BackendConfig {
        url: "http://localhost:8080".to_string(),
        timeout_secs: 60,
        ..Default::default()
    };
    
    // Create the backend
    let backend = Arc::new(LlamaBackend::new(config));
    
    // Run the proxy server on port 8080
    run_proxy(backend, 8080).await.unwrap();
}
```

## API Compatibility

This proxy is designed to be compatible with the OpenAI API as defined in `openapi.with-code-samples.yml`. It supports:

### Endpoints

- `POST /v1/chat/completions` - Create chat completions
- `GET /v1/models` - List available models
- `GET /health` - Health check

### Response Types

All response types match the OpenAPI specification exactly:

- `CreateChatCompletionResponse` - Standard chat completion response
- `CreateChatCompletionStreamResponse` - Streaming response
- `ErrorResponse` - Error responses with proper codes

### Error Handling

Errors are mapped to OpenAI-compatible formats:

```json
{
  "error": {
    "message": "Invalid model specified",
    "type": "invalid_request_error",
    "param": "model",
    "code": "invalid_model"
  }
}
```

## Custom Backends

To implement a custom backend, implement the `InferenceBackend` trait:

```rust
use async_trait::async_trait;
use openai_proxy::{
    CreateChatCompletionRequest, CreateChatCompletionResponse,
    BackendConfig, BackendResult, InferenceBackend, StreamResponse,
};

struct MyCustomBackend {
    config: BackendConfig,
    // Your backend-specific fields
}

#[async_trait]
impl InferenceBackend for MyCustomBackend {
    fn name(&self) -> &str {
        "my-custom-backend"
    }

    fn supports_model(&self, model: &str) -> bool {
        true // Your logic
    }

    async fn create_chat_completion(
        &self,
        request: CreateChatCompletionRequest,
    ) -> BackendResult<CreateChatCompletionResponse> {
        // Your implementation
        todo!()
    }

    async fn create_chat_completion_stream(
        &self,
        request: CreateChatCompletionRequest,
    ) -> BackendResult<StreamResponse> {
        // Your implementation
        todo!()
    }

    async fn health_check(&self) -> BackendResult<()> {
        Ok(())
    }
}
```

## Development

```bash
# Build the crate
cargo build

# Run tests
cargo test

# Run with example
cargo run --example simple_proxy
```

## License

MIT
