//! llama.cpp backend implementation
//! This module provides integration with llama.cpp's server

use async_trait::async_trait;
use reqwest::Client;
use serde_json::json;

use tracing::{info, error};

use crate::backend::{
    BackendConfig, BackendResult,
    InferenceBackend,
};
use crate::types::{
    ChatCompletionChoice, ChatCompletionMessage, ChatMessage, CreateChatCompletionRequest, 
    CreateChatCompletionResponse, CreateChatCompletionStreamResponse, MessageRole, 
    StreamChoice, StreamDelta, Usage,
};

mod error;
pub use error::map_llama_error;

/// llama.cpp backend implementation
pub struct LlamaBackend {
    client: Client,
    config: BackendConfig,
}

impl LlamaBackend {
    /// Create a new llama.cpp backend
    pub fn new(config: BackendConfig) -> Self {
        Self {
            client: Client::new(),
            config,
        }
    }

    /// Map OpenAI request to llama.cpp format
    fn map_request(&self, request: &CreateChatCompletionRequest) -> serde_json::Value {
        let messages: Vec<serde_json::Value> = request
            .messages
            .iter()
            .map(|msg| {
                json!({
                    "role": match msg.role {
                        MessageRole::System => "system",
                        MessageRole::User => "user",
                        MessageRole::Assistant => "assistant",
                        MessageRole::Tool => "tool",
                        MessageRole::Developer => "developer",
                    },
                    "content": msg.content,
                })
            })
            .collect();

        let mut body = json!({
            "model": request.model,
            "messages": messages,
            "stream": false,
        });

        if let Some(temp) = request.temperature {
            body["temperature"] = json!(temp);
        }
        if let Some(top_p) = request.top_p {
            body["top_p"] = json!(top_p);
        }
        if let Some(max_tokens) = request.max_tokens {
            body["max_tokens"] = json!(max_tokens);
        }
        if let Some(n) = request.n {
            body["n"] = json!(n);
        }
        if let Some(stream) = request.stream {
            body["stream"] = json!(stream);
        }
        if let Some(stop) = &request.stop {
            body["stop"] = json!(stop);
        }
        if let Some(user) = &request.user {
            body["user"] = json!(user);
        }

        body
    }

    /// Handle streaming response from llama.cpp
    async fn handle_stream_response(
        &self,
        url: &str,
        body: serde_json::Value,
    ) -> BackendResult<CreateChatCompletionStreamResponse> {
        let response = self
            .client
            .post(url)
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_json: serde_json::Value = response.json().await.map_err(|e| {
                anyhow::anyhow!("Failed to parse backend error response: {}", e)
            })?;
            return Err(map_llama_error(status, error_json).into());
        }

        let text = response.text().await.map_err(|e| {
            anyhow::anyhow!("Failed to read stream: {}", e)
        })?;

        let choices = parse_stream_response(&text);

        Ok(CreateChatCompletionStreamResponse {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            object: "chat.completion.chunk".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: self.config.model.clone(),
            choices,
        })
    }
}

impl LlamaBackend {
    fn list_models_from_config(&self) -> BackendResult<serde_json::Value> {
        let models = self.config.models();
        let data: Vec<serde_json::Value> = models
            .into_iter()
            .map(|id| {
                json!({
                    "id": id,
                    "object": "model",
                    "created": chrono::Utc::now().timestamp(),
                    "owned_by": "llama.cpp"
                })
            })
            .collect();

        Ok(json!({
            "object": "list",
            "data": data
        }))
    }
}

#[async_trait]
impl InferenceBackend for LlamaBackend {
    /// Create a chat completion (non-streaming)
    async fn create_chat_completion(
        &self,
        request: CreateChatCompletionRequest,
    ) -> BackendResult<CreateChatCompletionResponse> {
        let url = format!("{}/v1/chat/completions", self.config.url());
        let body = self.map_request(&request);

        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_json: serde_json::Value = response.json().await.map_err(|e| {
                anyhow::anyhow!("Failed to parse backend error response: {}", e)
            })?;
            return Err(map_llama_error(status, error_json).into());
        }

        let json: serde_json::Value = response.json().await.map_err(|e| {
            anyhow::anyhow!("Failed to parse backend response: {}", e)
        })?;

        map_llama_response(json, &request.model)
    }

    /// Create a streaming chat completion
    async fn create_chat_completion_stream(
        &self,
        request: CreateChatCompletionRequest,
    ) -> BackendResult<CreateChatCompletionStreamResponse> {
        let url = format!("{}/v1/chat/completions", self.config.url());
        let mut req = request;
        req.stream = Some(true);
        let body = self.map_request(&req);

        self.handle_stream_response(&url, body).await
    }

    /// List available models
    async fn list_models(&self) -> BackendResult<serde_json::Value> {
        let url = format!("{}/v1/models", self.config.url());

        let response = match self.client.get(&url).send().await {
            Ok(resp) => resp,
            Err(e) => {
                info!("Failed to fetch models from backend, falling back to config: {}", e);
                return self.list_models_from_config();
            }
        };

        if !response.status().is_success() {
            info!("Backend returned error for /v1/models, falling back to config: {}", response.status());
            return self.list_models_from_config();
        }

        let json: serde_json::Value = match response.json().await {
            Ok(j) => j,
            Err(e) => {
                error!("Failed to parse backend models response: {}", e);
                return self.list_models_from_config();
            }
        };

        Ok(json)
    }

    /// Check if the backend is healthy
    async fn health(&self) -> BackendResult<bool> {
        let url = format!("{}/health", self.config.url());

        match self.client.get(&url).send().await {
            Ok(resp) => Ok(resp.status().is_success()),
            Err(_) => Ok(false),
        }
    }
}

/// Map llama.cpp response to OpenAI format
fn map_llama_response(
    response: serde_json::Value,
    model: &str,
) -> BackendResult<CreateChatCompletionResponse> {
    let choices = response
        .get("choices")
        .and_then(|c| c.as_array())
        .map(|arr| {
            arr.iter()
                .map(|choice| {
                    let message = choice.get("message").map(|m| ChatMessage {
                        role: MessageRole::Assistant,
                        content: m.get("content").and_then(|c| c.as_str()).unwrap_or("").to_string(),
                        name: m.get("name").and_then(|n| n.as_str()).map(String::from),
                        tool_calls: None,
                        tool_call_id: None,
                    }).unwrap_or(ChatMessage {
                        role: MessageRole::Assistant,
                        content: "".to_string(),
                        name: None,
                        tool_calls: None,
                        tool_call_id: None,
                    });

                    let finish_reason = choice.get("finish_reason").and_then(|f| f.as_str()).map(String::from);
                    
                    let role_str = match message.role {
                        MessageRole::System => "system",
                        MessageRole::User => "user",
                        MessageRole::Assistant => "assistant",
                        MessageRole::Tool => "tool",
                        MessageRole::Developer => "developer",
                    };

                    ChatCompletionChoice {
                        index: choice.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as i32,
                        message: ChatCompletionMessage {
                            role: role_str.to_string(),
                            content: Some(message.content),
                            tool_calls: None,
                            refusal: None,
                            annotations: None,
                        },
                        finish_reason,
                        logprobs: None,
                    }
                })
                .collect()
        })
        .unwrap_or_default();

    let usage = response.get("usage").map(|u| Usage {
        prompt_tokens: u.get("prompt_tokens").and_then(|p| p.as_i64()).unwrap_or(0) as i32,
        completion_tokens: u.get("completion_tokens").and_then(|c| c.as_i64()).unwrap_or(0) as i32,
        total_tokens: u.get("total_tokens").and_then(|t| t.as_i64()).unwrap_or(0) as i32,
        ..Default::default()
    });

    Ok(CreateChatCompletionResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        object: "chat.completion".to_string(),
        created: chrono::Utc::now().timestamp(),
        model: model.to_string(),
        choices,
        usage: usage.unwrap_or_default(),
        finish_reason: None,
        service_tier: None,
        system_fingerprint: None,
    })
}

/// Parse llama.cpp stream response
fn parse_stream_response(text: &str) -> Vec<StreamChoice> {
    let mut choices: Vec<StreamChoice> = Vec::new();

    for line in text.lines() {
        if let Some(data) = line.strip_prefix("data: ") {
            if data == "[DONE]" {
                break;
            }
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
                let delta = json
                    .get("choices")
                    .and_then(|c| c.as_array())
                    .and_then(|arr| arr.first())
                    .and_then(|c| c.get("delta"))
                    .map(|d| StreamDelta {
                        role: Some(MessageRole::Assistant.to_string()),
                        content: Some(d
                            .get("content")
                            .and_then(|c| c.as_str())
                            .unwrap_or("")
                            .to_string()),
                        tool_calls: None,
                        refusal: None,
                    })
                    .unwrap_or(StreamDelta {
                        role: None,
                        content: Some(String::new()),
                        tool_calls: None,
                        refusal: None,
                    });

                let finish_reason = json
                    .get("choices")
                    .and_then(|c| c.as_array())
                    .and_then(|arr| arr.first())
                    .and_then(|c| c.get("finish_reason"))
                    .and_then(|f| f.as_str())
                    .map(String::from);

                choices.push(StreamChoice {
                    index: 0,
                    delta: Some(delta),
                    finish_reason,
                    logprobs: None,
                });
            }
        }
    }
    choices
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ChatMessage, MessageRole, CreateChatCompletionRequest};
    use serde_json::json;

    #[test]
    fn test_map_request_basic() {
        let config = BackendConfig {
            url: "http://localhost:8080".to_string(),
            model: "test-model".to_string(),
            models: vec!["test-model".to_string()],
        };
        let backend = LlamaBackend::new(config);
        
        let request = CreateChatCompletionRequest {
            model: "test-model".to_string(),
            messages: vec![
                ChatMessage {
                    role: MessageRole::User,
                    content: "Hello".to_string(),
                    ..Default::default()
                }
            ],
            ..Default::default()
        };

        let mapped = backend.map_request(&request);
        
        assert_eq!(mapped["model"], "test-model");
        assert_eq!(mapped["messages"][0]["role"], "user");
        assert_eq!(mapped["messages"][0]["content"], "Hello");
        assert_eq!(mapped["stream"], false);
    }

    #[test]
    fn test_map_request_with_params() {
        let config = BackendConfig {
            url: "http://localhost:8080".to_string(),
            model: "test-model".to_string(),
            models: vec!["test-model".to_string()],
        };
        let backend = LlamaBackend::new(config);
        
        let request = CreateChatCompletionRequest {
            model: "test-model".to_string(),
            messages: vec![
                ChatMessage {
                    role: MessageRole::User,
                    content: "Hello".to_string(),
                    ..Default::default()
                }
            ],
            temperature: Some(0.7),
            top_p: Some(0.9),
            max_tokens: Some(100),
            stream: Some(true),
            ..Default::default()
        };

        let mapped = backend.map_request(&request);
        
        assert_eq!(mapped["temperature"], 0.7);
        assert_eq!(mapped["top_p"], 0.9);
        assert_eq!(mapped["max_tokens"], 100);
        assert_eq!(mapped["stream"], true);
    }

    #[test]
    fn test_map_llama_response() {
        let llama_response = json!({
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello there!"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        });

        let mapped = map_llama_response(llama_response, "test-model").unwrap();
        
        assert_eq!(mapped.model, "test-model");
        assert_eq!(mapped.choices.len(), 1);
        assert_eq!(mapped.choices[0].message.content, Some("Hello there!".to_string()));
        assert_eq!(mapped.choices[0].finish_reason, Some("stop".to_string()));
        assert_eq!(mapped.usage.total_tokens, 30);
    }

    #[test]
    fn test_list_models_from_config() {
        let config = BackendConfig {
            url: "http://localhost:8080".to_string(),
            model: "default-model".to_string(),
            models: vec!["model1".to_string(), "model2".to_string()],
        };
        let backend = LlamaBackend::new(config);
        
        let result = backend.list_models_from_config().unwrap();
        
        assert_eq!(result["object"], "list");
        let data = result["data"].as_array().unwrap();
        assert_eq!(data.len(), 2);
        assert_eq!(data[0]["id"], "model1");
        assert_eq!(data[1]["id"], "model2");
    }

    #[test]
    fn test_parse_stream_response() {
        let stream_data = "data: {\"choices\": [{\"index\": 0, \"delta\": {\"content\": \"Hello\"}, \"finish_reason\": null}]}\n\
                           data: {\"choices\": [{\"index\": 0, \"delta\": {\"content\": \" world\"}, \"finish_reason\": \"stop\"}]}\n\
                           data: [DONE]";
        
        let choices = parse_stream_response(stream_data);
        
        assert_eq!(choices.len(), 2);
        assert_eq!(choices[0].delta.as_ref().unwrap().content, Some("Hello".to_string()));
        assert_eq!(choices[1].delta.as_ref().unwrap().content, Some(" world".to_string()));
        assert_eq!(choices[1].finish_reason, Some("stop".to_string()));
    }
}
