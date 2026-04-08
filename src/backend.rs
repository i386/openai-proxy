//! Backend provider traits and implementations
//! This module provides the backend trait for llama.cpp inference


use async_trait::async_trait;
use futures_util::Stream;
use std::pin::Pin;

use crate::types::{
    CreateChatCompletionRequest, CreateChatCompletionResponse, 
    CreateChatCompletionStreamResponse, ErrorResponse,
};

pub type BackendResult<T> = anyhow::Result<T>;
pub type BoxStream<'a, T> = Pin<Box<dyn Stream<Item = T> + Send + 'a>>;

/// Errors that can occur in the backend
#[derive(Debug, Clone)]
pub struct BackendError {
    pub message: String,
    pub error_type: ErrorType,
    pub param: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorType {
    InvalidRequest,
    Authentication,
    Permission,
    NotFound,
    RateLimit,
    Internal,
    ServiceUnavailable,
    ContextLengthExceeded,
    UnsupportedFeature,
}

impl BackendError {
    pub fn invalid_request(message: impl Into<String>, param: Option<String>) -> Self {
        Self {
            message: message.into(),
            error_type: ErrorType::InvalidRequest,
            param,
        }
    }

    pub fn authentication(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            error_type: ErrorType::Authentication,
            param: None,
        }
    }

    pub fn permission(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            error_type: ErrorType::Permission,
            param: None,
        }
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            error_type: ErrorType::NotFound,
            param: None,
        }
    }

    pub fn rate_limit(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            error_type: ErrorType::RateLimit,
            param: None,
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            error_type: ErrorType::Internal,
            param: None,
        }
    }

    pub fn service_unavailable(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            error_type: ErrorType::ServiceUnavailable,
            param: None,
        }
    }

    pub fn context_length_exceeded(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            error_type: ErrorType::ContextLengthExceeded,
            param: None,
        }
    }

    pub fn unsupported_feature(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            error_type: ErrorType::UnsupportedFeature,
            param: None,
        }
    }

    pub fn to_error_response(&self) -> ErrorResponse {
        let (r#type, code) = match self.error_type {
            ErrorType::InvalidRequest => ("invalid_request_error", "invalid_value"),
            ErrorType::Authentication => ("authentication_error", "invalid_api_key"),
            ErrorType::Permission => ("permission_error", "insufficient_quota"),
            ErrorType::NotFound => ("invalid_request_error", "model_not_found"),
            ErrorType::RateLimit => ("rate_limit_error", "rate_limit_exceeded"),
            ErrorType::Internal => ("server_error", "internal_server_error"),
            ErrorType::ServiceUnavailable => ("server_error", "service_unavailable"),
            ErrorType::ContextLengthExceeded => ("invalid_request_error", "context_length_exceeded"),
            ErrorType::UnsupportedFeature => ("invalid_request_error", "unsupported_model_feature"),
        };

        ErrorResponse {
            error: crate::types::Error {
                message: self.message.clone(),
                r#type: r#type.to_string(),
                param: self.param.clone(),
                code: Some(code.to_string()),
            },
        }
    }
}

impl std::fmt::Display for ErrorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErrorType::InvalidRequest => write!(f, "invalid_request_error"),
            ErrorType::Authentication => write!(f, "authentication_error"),
            ErrorType::Permission => write!(f, "permission_error"),
            ErrorType::NotFound => write!(f, "not_found_error"),
            ErrorType::RateLimit => write!(f, "rate_limit_error"),
            ErrorType::Internal => write!(f, "internal_error"),
            ErrorType::ServiceUnavailable => write!(f, "service_unavailable_error"),
            ErrorType::ContextLengthExceeded => write!(f, "context_length_exceeded_error"),
            ErrorType::UnsupportedFeature => write!(f, "unsupported_model_feature"),
        }
    }
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.error_type, self.message)
    }
}

impl std::error::Error for BackendError {}

/// Configuration for the backend
#[derive(Debug, Clone)]
pub struct BackendConfig {
    /// URL of the llama.cpp server
    pub url: String,
    /// Default model to use
    pub model: String,
    /// List of models supported by this backend
    pub models: Vec<String>,
}

impl BackendConfig {
    /// Get the URL of the backend
    pub fn url(&self) -> &str {
        &self.url
    }

    /// Get the list of models
    pub fn models(&self) -> Vec<String> {
        if self.models.is_empty() {
            vec![self.model.clone()]
        } else {
            self.models.clone()
        }
    }

    /// Get the default model
    pub fn model(&self) -> &str {
        &self.model
    }
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            url: "http://127.0.0.1:8080".to_string(),
            model: "llama-3.1-8b".to_string(),
            models: vec!["llama-3.1-8b".to_string()],
        }
    }
}

/// Stream response from the backend
#[derive(Debug, Clone, serde::Serialize)]
pub struct StreamResponse {
    pub chunk: String,
    pub is_final: bool,
}

/// Trait for inference backends
#[async_trait]
pub trait InferenceBackend: Send + Sync {
    /// Create a chat completion
    async fn create_chat_completion(
        &self,
        request: CreateChatCompletionRequest,
    ) -> BackendResult<CreateChatCompletionResponse>;

    /// Create a streaming chat completion
    async fn create_chat_completion_stream(
        &self,
        request: CreateChatCompletionRequest,
    ) -> BackendResult<BoxStream<'static, BackendResult<CreateChatCompletionStreamResponse>>>;

    /// List available models
    async fn list_models(&self) -> BackendResult<serde_json::Value>;

    /// Check if the backend is healthy
    async fn health(&self) -> BackendResult<bool>;
}

/// Map error from backend to OpenAI format
pub fn map_error(error: reqwest::Error) -> BackendError {
    if error.is_connect() {
        BackendError::service_unavailable("Cannot connect to backend")
    } else if error.is_timeout() {
        BackendError::internal("Backend timeout")
    } else {
        BackendError::internal(error.to_string())
    }
}
