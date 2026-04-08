//! OpenAI-compatible types for the proxy
//! These types match the OpenAPI spec in openapi.with-code-samples.yml

use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ============================================================================
// Chat Completion Request Types
// ============================================================================

/// Request for creating a chat completion
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub struct CreateChatCompletionRequest {
    /// ID of the model to use
    pub model: String,
    /// Messages to generate completions for
    pub messages: Vec<ChatMessage>,
    /// Maximum tokens to generate
    #[serde(default)]
    pub max_tokens: Option<i32>,
    /// Sampling temperature
    #[serde(default)]
    pub temperature: Option<f64>,
    /// Nucleus sampling parameter
    #[serde(default)]
    pub top_p: Option<f64>,
    /// Number of completions to generate
    #[serde(default = "default_n")]
    pub n: Option<i32>,
    /// Whether to stream the response
    #[serde(default)]
    pub stream: Option<bool>,
    /// Stop sequences
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    /// Presence penalty
    #[serde(default)]
    pub presence_penalty: Option<f64>,
    /// Frequency penalty
    #[serde(default)]
    pub frequency_penalty: Option<f64>,
    /// User identifier
    #[serde(default)]
    pub user: Option<String>,
    /// Logit bias
    #[serde(default)]
    pub logit_bias: Option<std::collections::HashMap<String, f64>>,
    /// Response format
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
}

fn default_n() -> Option<i32> { Some(1) }

/// A message in the chat
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub struct ChatMessage {
    /// Role of the message sender
    #[serde(default)]
    pub role: MessageRole,
    /// Content of the message
    #[serde(default)]
    pub content: ChatMessageContent,
    /// Name of the sender
    #[serde(default)]
    pub name: Option<String>,
    /// Tool calls (for assistant messages)
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Tool call ID (for tool messages)
    #[serde(default)]
    pub tool_call_id: Option<String>,
}

/// Content of the message
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChatMessageContent {
    /// Text content
    Text(String),
    /// Multi-part content
    Parts(Vec<ChatMessagePart>),
}

impl Default for ChatMessageContent {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

impl ChatMessage {
    pub fn new(role: MessageRole, content: impl Into<ChatMessageContent>) -> Self {
        Self {
            role,
            content: content.into(),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        }
    }
}

impl From<String> for ChatMessageContent {
    fn from(s: String) -> Self {
        ChatMessageContent::Text(s)
    }
}

impl From<&str> for ChatMessageContent {
    fn from(s: &str) -> Self {
        ChatMessageContent::Text(s.to_string())
    }
}

impl ChatMessageContent {
    pub fn as_str(&self) -> String {
        match self {
            Self::Text(s) => s.clone(),
            Self::Parts(parts) => parts
                .iter()
                .filter_map(|part| match part {
                    ChatMessagePart::Text { text } => Some(text.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(" "),
        }
    }
}

/// Part of a multi-part message content
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatMessagePart {
    /// Text part
    Text {
        /// Text content
        text: String,
    },
    /// Image part
    ImageUrl {
        /// Image URL or base64 data
        image_url: ImageUrl,
    },
    /// Audio part
    InputAudio {
        /// Audio data
        input_audio: InputAudio,
    },
}

/// Image URL details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    /// URL of the image or base64-encoded image data
    pub url: String,
    /// Detail level (optional)
    #[serde(default)]
    pub detail: Option<String>,
}

/// Input audio details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputAudio {
    /// Base64-encoded audio data
    pub data: String,
    /// Audio format (optional)
    pub format: String,
}

/// Role of the message sender
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    #[default]
    User,
    Assistant,
    Tool,
    Developer,
}

impl std::fmt::Display for MessageRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MessageRole::System => write!(f, "system"),
            MessageRole::User => write!(f, "user"),
            MessageRole::Assistant => write!(f, "assistant"),
            MessageRole::Tool => write!(f, "tool"),
            MessageRole::Developer => write!(f, "developer"),
        }
    }
}


/// A tool call in a message
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ToolCall {
    /// ID of the tool call
    pub id: String,
    /// Type of the tool call
    #[serde(rename = "type")]
    pub tool_type: String,
    /// The function to call
    pub function: ToolCallFunction,
}

/// Function details in a tool call
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ToolCallFunction {
    /// Name of the function
    pub name: String,
    /// Arguments to the function (JSON string)
    pub arguments: String,
}

/// Response format
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ResponseFormat {
    /// Type of response format
    #[serde(rename = "type")]
    pub format_type: String,
    /// JSON schema (for json_object type)
    #[serde(default)]
    pub schema: Option<serde_json::Value>,
}

// ============================================================================
// Chat Completion Response Types
// ============================================================================

/// Response from creating a chat completion
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct CreateChatCompletionResponse {
    /// Unique identifier for this completion
    pub id: String,
    /// Object type
    pub object: String,
    /// Unix timestamp of creation
    pub created: i64,
    /// Model used
    pub model: String,
    /// List of completion choices
    pub choices: Vec<ChatCompletionChoice>,
    /// Usage statistics
    pub usage: Usage,
    /// Service tier
    #[serde(default)]
    pub service_tier: Option<String>,
    /// System fingerprint
    #[serde(default)]
    pub system_fingerprint: Option<String>,
    /// The finish reason for the last choice
    #[serde(default)]
    pub finish_reason: Option<String>,
}

/// A choice in the completion response
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ChatCompletionChoice {
    /// Index of this choice
    pub index: i32,
    /// The message generated
    pub message: ChatCompletionMessage,
    /// Log probabilities (if requested)
    #[serde(default)]
    pub logprobs: Option<LogProbs>,
    /// Reason for completion stopping
    pub finish_reason: Option<String>,
}

/// Message in a chat completion
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ChatCompletionMessage {
    /// Role of the message sender
    pub role: String,
    /// Content of the message
    pub content: Option<String>,
    /// Tool calls (if any)
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Refusal (if model refused to answer)
    #[serde(default)]
    pub refusal: Option<String>,
    /// Annotations
    #[serde(default)]
    pub annotations: Option<Vec<serde_json::Value>>,
}

/// Log probabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct LogProbs {
    /// Token log probabilities
    pub tokens: Vec<String>,
    /// Token log probability values
    pub token_logprobs: Vec<f64>,
    /// Top log probabilities per position
    #[serde(default)]
    pub top_logprobs: Option<Vec<std::collections::HashMap<String, f64>>>,
    /// Text offset
    #[serde(default)]
    pub text_offset: Option<Vec<i32>>,
}

/// Usage statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub struct Usage {
    /// Tokens in the prompt
    pub prompt_tokens: i32,
    /// Tokens in the completion
    pub completion_tokens: i32,
    /// Total tokens used
    pub total_tokens: i32,
    /// Prompt token details
    #[serde(default)]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    /// Completion token details
    #[serde(default)]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

/// Details about prompt tokens
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct PromptTokensDetails {
    /// Cached tokens
    #[serde(default)]
    pub cached_tokens: Option<i32>,
    /// Audio tokens
    #[serde(default)]
    pub audio_tokens: Option<i32>,
}

/// Details about completion tokens
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct CompletionTokensDetails {
    /// Reasoning tokens
    #[serde(default)]
    pub reasoning_tokens: Option<i32>,
    /// Audio tokens
    #[serde(default)]
    pub audio_tokens: Option<i32>,
    /// Accepted prediction tokens
    #[serde(default)]
    pub accepted_prediction_tokens: Option<i32>,
    /// Rejected prediction tokens
    #[serde(default)]
    pub rejected_prediction_tokens: Option<i32>,
}

// ============================================================================
// Streaming Response Types
// ============================================================================

/// Streaming response for chat completion
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct CreateChatCompletionStreamResponse {
    /// Unique identifier for this chunk
    pub id: String,
    /// Object type
    pub object: String,
    /// Unix timestamp of creation
    pub created: i64,
    /// Model used
    pub model: String,
    /// Choices in this chunk
    pub choices: Vec<StreamChoice>,
}

/// A streaming choice
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct StreamChoice {
    /// Index of this choice
    pub index: i32,
    /// Delta content (for streaming)
    #[serde(default)]
    pub delta: Option<StreamDelta>,
    /// Finish reason
    #[serde(default)]
    pub finish_reason: Option<String>,
    /// Logprobs
    #[serde(default)]
    pub logprobs: Option<LogProbs>,
}

/// Delta content for streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct StreamDelta {
    /// Role of the sender
    #[serde(default)]
    pub role: Option<String>,
    /// Content delta
    #[serde(default)]
    pub content: Option<String>,
    /// Tool calls
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Refusal
    #[serde(default)]
    pub refusal: Option<String>,
}

// ============================================================================
// Error Response Types (OpenAI-compatible)
// ============================================================================

/// Error response structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    /// The error object
    pub error: Error,
}

/// Error details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Error {
    /// Error code (e.g., "invalid_api_key", "rate_limit_exceeded")
    #[serde(default)]
    pub code: Option<String>,
    /// Error message
    pub message: String,
    /// Parameter that caused the error
    #[serde(default)]
    pub param: Option<String>,
    /// Type of error
    #[serde(rename = "type")]
    pub r#type: String,
}

// ============================================================================
// Helper Functions
// ============================================================================

impl CreateChatCompletionResponse {
    /// Generate a new chat completion response with the given parameters
    pub fn new(
        model: &str,
        choices: Vec<ChatCompletionChoice>,
        usage: Usage,
    ) -> Self {
        Self {
            id: format!("chatcmpl-{}", &Uuid::new_v4().to_string().replace("-", "")[..12]),
            object: "chat.completion".to_string(),
            created: Utc::now().timestamp(),
            model: model.to_string(),
            choices,
            usage,
            service_tier: Some("default".to_string()),
            system_fingerprint: None,
            finish_reason: None,
        }
    }
}

impl Error {
    /// Create an invalid request error
    pub fn invalid_request(message: impl Into<String>, param: Option<String>) -> Self {
        Self {
            code: Some("invalid_request_error".to_string()),
            message: message.into(),
            param,
            r#type: "invalid_request_error".to_string(),
        }
    }

    /// Create an authentication error
    pub fn authentication(message: impl Into<String>) -> Self {
        Self {
            code: Some("invalid_api_key".to_string()),
            message: message.into(),
            param: None,
            r#type: "authentication_error".to_string(),
        }
    }

    /// Create a rate limit error
    pub fn rate_limit(message: impl Into<String>) -> Self {
        Self {
            code: Some("rate_limit_exceeded".to_string()),
            message: message.into(),
            param: None,
            r#type: "rate_limit_error".to_string(),
        }
    }

    /// Create an internal server error
    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            code: Some("internal_error".to_string()),
            message: message.into(),
            param: None,
            r#type: "internal_server_error".to_string(),
        }
    }

    /// Create a not found error
    pub fn not_found(message: impl Into<String>) -> Self {
        Self {
            code: Some("not_found".to_string()),
            message: message.into(),
            param: None,
            r#type: "not_found_error".to_string(),
        }
    }

    /// Create a context length exceeded error
    pub fn context_length_exceeded(message: impl Into<String>) -> Self {
        Self {
            code: Some("context_length_exceeded".to_string()),
            message: message.into(),
            param: Some("messages".to_string()),
            r#type: "invalid_request_error".to_string(),
        }
    }

    /// Create a service unavailable error
    pub fn service_unavailable(message: impl Into<String>) -> Self {
        Self {
            code: Some("server_error".to_string()),
            message: message.into(),
            param: None,
            r#type: "service_unavailable_error".to_string(),
        }
    }
}

impl ErrorResponse {
    /// Create an error response with the given error
    pub fn new(error: Error) -> Self {
        Self { error }
    }

    /// Create an invalid request error response
    pub fn invalid_request(message: impl Into<String>, param: Option<String>) -> Self {
        Self::new(Error::invalid_request(message, param))
    }

    /// Create an authentication error response
    pub fn authentication(message: impl Into<String>) -> Self {
        Self::new(Error::authentication(message))
    }

    /// Create a rate limit error response
    pub fn rate_limit(message: impl Into<String>) -> Self {
        Self::new(Error::rate_limit(message))
    }

    /// Create an internal server error response
    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(Error::internal(message))
    }

    /// Create a not found error response
    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new(Error::not_found(message))
    }

    /// Create a context length exceeded error response
    pub fn context_length_exceeded(message: impl Into<String>) -> Self {
        Self::new(Error::context_length_exceeded(message))
    }

    /// Create a service unavailable error response
    pub fn service_unavailable(message: impl Into<String>) -> Self {
        Self::new(Error::service_unavailable(message))
    }
}

impl Usage {
    /// Create a simple usage struct
    pub fn new(prompt_tokens: i32, completion_tokens: i32) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            prompt_tokens_details: None,
            completion_tokens_details: None,
        }
    }
}
