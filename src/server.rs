//! OpenAI-compatible proxy server
//! This module provides the HTTP server that accepts OpenAI-format requests
//! and routes them to the appropriate inference backend

use std::sync::Arc;
use std::convert::Infallible;

use hyper::body::{Bytes, Frame};
use http_body_util::{BodyExt, Full, StreamBody};
use http_body_util::combinators::UnsyncBoxBody;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use tracing::{info, error};
use tokio::net::TcpListener;
use futures_util::{StreamExt, stream};

use crate::backend::InferenceBackend;
use crate::types::{CreateChatCompletionRequest, ErrorResponse};

/// The main proxy server
pub struct OpenAIProxy {
    /// The backend to use for inference
    backend: Arc<dyn InferenceBackend>,
}

impl OpenAIProxy {
    /// Create a new proxy with the given backend
    pub fn new(backend: Arc<dyn InferenceBackend>) -> Self {
        Self { backend }
    }

    /// Start the proxy server
    pub async fn serve(self, addr: &str) -> Result<(), Box<dyn std::error::Error>> {
        let listener = TcpListener::bind(addr).await?;
        info!("OpenAI Proxy listening on {}", addr);

        let proxy = Arc::new(self);

        loop {
            let (stream, remote_addr) = listener.accept().await?;
            let io = TokioIo::new(stream);
            let backend = proxy.backend.clone();
            let proxy_clone = proxy.clone();

            tokio::task::spawn(async move {
                let service = service_fn(move |req| {
                    let backend = backend.clone();
                    let proxy = proxy_clone.clone();
                    async move {
                        proxy.handle_request(req, backend.as_ref()).await
                    }
                });

                if let Err(err) = http1::Builder::new()
                    .serve_connection(io, service)
                    .await
                {
                    error!("Error serving connection from {}: {}", remote_addr, err);
                }
            });
        }
    }

    /// Handle an incoming HTTP request
    async fn handle_request(
        &self,
        req: Request<hyper::body::Incoming>,
        backend: &dyn InferenceBackend,
    ) -> Result<Response<UnsyncBoxBody<Bytes, Infallible>>, Infallible> {
        let path = req.uri().path();
        let method = req.method().clone();

        // Route based on path and method
        match (method, path) {
            // Chat completions endpoint
            (Method::POST, "/v1/chat/completions") | (Method::POST, "/v1/responses") => {
                self.handle_chat_completions(req, backend).await
            }
            
            // Health check endpoint
            (Method::GET, "/health") | (Method::GET, "/v1/health") => {
                let body = Full::new(Bytes::from(serde_json::json!({"status": "ok"}).to_string().into_bytes()));
                Ok(Response::builder()
                    .status(StatusCode::OK)
                    .header("Content-Type", "application/json")
                    .body(UnsyncBoxBody::new(body))
                    .unwrap())
            }

            // Models list endpoint
            (Method::GET, "/v1/models") => {
                self.handle_list_models().await
            }

            // Catch-all for not found
            _ => {
                let error = ErrorResponse::not_found(format!("Unknown endpoint: {}", path));
                let body = Full::new(Bytes::from(serde_json::to_vec(&error).unwrap()));
                Ok(Response::builder()
                    .status(StatusCode::NOT_FOUND)
                    .header("Content-Type", "application/json")
                    .body(UnsyncBoxBody::new(body))
                    .unwrap())
            }
        }
    }

    fn handle_error(&self, e: anyhow::Error) -> Result<Response<UnsyncBoxBody<Bytes, Infallible>>, Infallible> {
        let (status, error_response) = if let Some(be) = e.downcast_ref::<crate::backend::BackendError>() {
            let status = match be.error_type {
                crate::backend::ErrorType::InvalidRequest => StatusCode::BAD_REQUEST,
                crate::backend::ErrorType::Authentication => StatusCode::UNAUTHORIZED,
                crate::backend::ErrorType::Permission => StatusCode::FORBIDDEN,
                crate::backend::ErrorType::NotFound => StatusCode::NOT_FOUND,
                crate::backend::ErrorType::RateLimit => StatusCode::TOO_MANY_REQUESTS,
                crate::backend::ErrorType::Internal => StatusCode::INTERNAL_SERVER_ERROR,
                crate::backend::ErrorType::ServiceUnavailable => StatusCode::SERVICE_UNAVAILABLE,
                crate::backend::ErrorType::ContextLengthExceeded => StatusCode::BAD_REQUEST,
                crate::backend::ErrorType::UnsupportedFeature => StatusCode::NOT_IMPLEMENTED,
            };
            (status, be.to_error_response())
        } else {
            (StatusCode::INTERNAL_SERVER_ERROR, ErrorResponse::internal(e.to_string()))
        };

        let body = Full::new(Bytes::from(serde_json::to_vec(&error_response).unwrap()));
            
        Ok(Response::builder()
            .status(status)
            .header("Content-Type", "application/json")
            .body(UnsyncBoxBody::new(body))
            .unwrap())
    }

    /// Handle chat completions request
    async fn handle_chat_completions(
        &self,
        req: Request<hyper::body::Incoming>,
        backend: &dyn InferenceBackend,
    ) -> Result<Response<UnsyncBoxBody<Bytes, Infallible>>, Infallible> {
        let path = req.uri().path().to_string();
        // Parse the request body
        let body_bytes = match req.collect().await {
            Ok(collected) => collected.to_bytes().to_vec(),
            Err(e) => {
                let error = ErrorResponse::invalid_request(
                    format!("Failed to read request body: {}", e),
                    None,
                );
                let body = UnsyncBoxBody::new(Full::new(Bytes::from(serde_json::to_vec(&error).unwrap())));
                return Ok(Response::builder()
                    .status(StatusCode::BAD_REQUEST)
                    .body(body)
                    .unwrap());
            }
        };

        // Deserialize as Value first to allow for normalization
        let mut body_json: serde_json::Value = match serde_json::from_slice(&body_bytes) {
            Ok(val) => val,
            Err(e) => {
                let error = ErrorResponse::invalid_request(
                    format!("Invalid JSON in request body: {}", e),
                    Some("body".to_string()),
                );
                let body = UnsyncBoxBody::new(Full::new(Bytes::from(serde_json::to_vec(&error).unwrap())));
                return Ok(Response::builder()
                    .status(StatusCode::BAD_REQUEST)
                    .body(body)
                    .unwrap());
            }
        };

        // Normalize the request
        if let Some(obj) = body_json.as_object_mut() {
            // 1. Handle /v1/responses -> /v1/chat/completions translation
            if path == "/v1/responses" {
                if let Some(input) = obj.remove("input") {
                    // Very basic translation of 'input' to 'messages'
                    // In a real implementation we'd want to handle all block types
                    let messages = if let Some(input_arr) = input.as_array() {
                        let mut msgs = Vec::new();
                        for item in input_arr {
                            if let Some(item_obj) = item.as_object() {
                                let role = item_obj.get("role").cloned().unwrap_or(serde_json::json!("user"));
                                let content = if let Some(content_val) = item_obj.get("content") {
                                    if let Some(content_arr) = content_val.as_array() {
                                        let mut text_content = String::new();
                                        for block in content_arr {
                                            if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                                                text_content.push_str(text);
                                            }
                                        }
                                        serde_json::json!(text_content)
                                    } else {
                                        content_val.clone()
                                    }
                                } else {
                                    serde_json::json!("")
                                };
                                msgs.push(serde_json::json!({
                                    "role": role,
                                    "content": content
                                }));
                            }
                        }
                        serde_json::json!(msgs)
                    } else {
                        input
                    };
                    obj.insert("messages".to_string(), messages);
                }
            }

            // 2. Parameter aliasing
            for alias in ["max_completion_tokens", "max_output_tokens"] {
                if let Some(val) = obj.remove(alias) {
                    obj.entry("max_tokens".to_string()).or_insert(val);
                }
            }

            // 3. Role mapping (developer -> system)
            if let Some(messages) = obj.get_mut("messages").and_then(|m| m.as_array_mut()) {
                for msg in messages {
                    if let Some(msg_obj) = msg.as_object_mut() {
                        if let Some(role) = msg_obj.get_mut("role") {
                            if role == "developer" {
                                *role = serde_json::json!("system");
                            }
                        }
                    }
                }
            }
        }

        // Deserialize the normalized request
        let request: CreateChatCompletionRequest = match serde_json::from_value(body_json) {
            Ok(req) => req,
            Err(e) => {
                let error = ErrorResponse::invalid_request(
                    format!("Invalid request after normalization: {}", e),
                    None,
                );
                let body = UnsyncBoxBody::new(Full::new(Bytes::from(serde_json::to_vec(&error).unwrap())));
                return Ok(Response::builder()
                    .status(StatusCode::BAD_REQUEST)
                    .body(body)
                    .unwrap());
            }
        };

        // Validate the request
        if request.model.is_empty() {
            let error = ErrorResponse::invalid_request(
                "Missing required parameter: model",
                Some("model".to_string()),
            );
            let body = UnsyncBoxBody::new(Full::new(Bytes::from(serde_json::to_vec(&error).unwrap())));
            return Ok(Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .body(body)
                .unwrap());
        }

        if request.messages.is_empty() {
            let error = ErrorResponse::invalid_request(
                "Missing required parameter: messages",
                Some("messages".to_string()),
            );
            let body = UnsyncBoxBody::new(Full::new(Bytes::from(serde_json::to_vec(&error).unwrap())));
            return Ok(Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .body(body)
                .unwrap());
        }

        // Check if streaming is requested
        let stream = request.stream.unwrap_or(false);

        // Call the backend
        if stream {
            match backend.create_chat_completion_stream(request).await {
                Ok(stream) => {
                    let sse_stream = stream.map(|res| {
                        match res {
                            Ok(chunk) => {
                                let data = serde_json::to_string(&chunk).unwrap_or_default();
                                Ok::<Frame<Bytes>, Infallible>(Frame::data(Bytes::from(format!("data: {}\n\n", data))))
                            }
                            Err(e) => {
                                let error_resp = ErrorResponse::internal(e.to_string());
                                let data = serde_json::to_string(&error_resp).unwrap_or_default();
                                Ok::<Frame<Bytes>, Infallible>(Frame::data(Bytes::from(format!("data: {}\n\n", data))))
                            }
                        }
                    });

                    // Add the [DONE] message at the end
                    let final_stream = sse_stream.chain(stream::once(async {
                        Ok::<Frame<Bytes>, Infallible>(Frame::data(Bytes::from("data: [DONE]\n\n")))
                    }));

                    let body = UnsyncBoxBody::new(StreamBody::new(final_stream));
                    
                    Ok(Response::builder()
                        .status(StatusCode::OK)
                        .header("Content-Type", "text/event-stream")
                        .header("Cache-Control", "no-cache")
                        .header("Connection", "keep-alive")
                        .body(body)
                        .unwrap())
                }
                Err(e) => {
                    self.handle_error(e)
                }
            }
        } else {
            match backend.create_chat_completion(request).await {
                Ok(response) => {
                    let json = serde_json::to_vec(&response).unwrap();
                    Ok(Response::builder()
                        .status(StatusCode::OK)
                        .header("Content-Type", "application/json")
                        .body(UnsyncBoxBody::new(Full::new(Bytes::from(json))))
                        .unwrap())
                }
                Err(e) => {
                    self.handle_error(e)
                }
            }
        }
    }

    /// Handle list models request
    async fn handle_list_models(&self) -> Result<Response<UnsyncBoxBody<Bytes, Infallible>>, Infallible> {
        match self.backend.list_models().await {
            Ok(models) => {
                let body = UnsyncBoxBody::new(Full::new(Bytes::from(serde_json::to_vec(&models).unwrap())));
                Ok(Response::builder()
                    .status(StatusCode::OK)
                    .header("Content-Type", "application/json")
                    .body(body)
                    .unwrap())
            }
            Err(e) => {
                self.handle_error(e)
            }
        }
    }
}

/// Build and run the proxy server
pub async fn run_proxy(backend: Arc<dyn InferenceBackend>, port: u16) -> Result<(), Box<dyn std::error::Error>> {
    let proxy = OpenAIProxy::new(backend);
    let addr = format!("0.0.0.0:{}", port);
    proxy.serve(&addr).await
}
