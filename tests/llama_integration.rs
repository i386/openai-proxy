use std::sync::Arc;
use tokio::net::TcpListener;
use wiremock::{MockServer, Mock, ResponseTemplate};
use wiremock::matchers::{method, path};
use serde_json::json;
use openai_proxy::{
    run_proxy, LlamaBackend, BackendConfig, CreateChatCompletionRequest,
    ChatMessage, MessageRole,
};

#[tokio::test]
async fn test_llama_chat_completion_integration() {
    // 1. Start a mock llama.cpp server
    let mock_server = MockServer::start().await;
    
    // Mock the llama.cpp /v1/chat/completions endpoint
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200)
            .set_body_json(json!({
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Hello! I am a mock assistant."
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30
                }
            })))
        .mount(&mock_server)
        .await;

    // Mock the /v1/models endpoint
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200)
            .set_body_json(json!({
                "object": "list",
                "data": [
                    {
                        "id": "mock-llama",
                        "object": "model",
                        "created": 123456789,
                        "owned_by": "llama.cpp"
                    }
                ]
            })))
        .mount(&mock_server)
        .await;

    // 2. Start the OpenAI Proxy
    // Find a free port for the proxy
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let proxy_port = listener.local_addr().unwrap().port();
    drop(listener); // Close it so run_proxy can bind to it

    let config = BackendConfig {
        url: mock_server.uri(),
        model: "mock-llama".to_string(),
        models: vec!["mock-llama".to_string()],
    };
    let backend = Arc::new(LlamaBackend::new(config));
    
    // Run proxy in the background
    let backend_clone = backend.clone();
    tokio::spawn(async move {
        run_proxy(backend_clone, proxy_port).await.unwrap();
    });

    // Wait a bit for proxy to start
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // 3. Send a request to the proxy
    let client = reqwest::Client::new();
    let proxy_url = format!("http://127.0.0.1:{}/v1/chat/completions", proxy_port);
    
    let request = CreateChatCompletionRequest {
        model: "mock-llama".to_string(),
        messages: vec![
            ChatMessage {
                role: MessageRole::User,
                content: "Hi".to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            }
        ],
        ..Default::default()
    };

    let response = client.post(&proxy_url)
        .json(&request)
        .send()
        .await
        .expect("Failed to send request to proxy");

    // 4. Verify the response
    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().await.unwrap();
    
    assert_eq!(body["choices"][0]["message"]["content"], "Hello! I am a mock assistant.");
    assert_eq!(body["usage"]["total_tokens"], 30);
    assert_eq!(body["model"], "mock-llama");
}

#[tokio::test]
async fn test_llama_list_models_integration() {
    let mock_server = MockServer::start().await;
    
    // We don't even need to mock /v1/models on the backend anymore 
    // because it should be served from config

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let proxy_port = listener.local_addr().unwrap().port();
    drop(listener);

    let config = BackendConfig {
        url: mock_server.uri(),
        model: "config-model".to_string(),
        models: vec!["config-model".to_string(), "another-model".to_string()],
    };
    let backend = Arc::new(LlamaBackend::new(config));
    
    tokio::spawn(async move {
        run_proxy(backend, proxy_port).await.unwrap();
    });

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    let client = reqwest::Client::new();
    let proxy_url = format!("http://127.0.0.1:{}/v1/models", proxy_port);
    
    let response = client.get(&proxy_url)
        .send()
        .await
        .expect("Failed to send request to proxy");

    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().await.unwrap();
    
    let model_ids: Vec<&str> = body["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap())
        .collect();
    
    assert!(model_ids.contains(&"config-model"));
    assert!(model_ids.contains(&"another-model"));
    assert_eq!(model_ids.len(), 2);
}

#[tokio::test]
async fn test_health_check_integration() {
    let mock_server = MockServer::start().await;
    
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let proxy_port = listener.local_addr().unwrap().port();
    drop(listener);

    let config = BackendConfig {
        url: mock_server.uri(),
        model: "test-model".to_string(),
        models: vec!["test-model".to_string()],
    };
    let backend = Arc::new(LlamaBackend::new(config));
    
    tokio::spawn(async move {
        run_proxy(backend, proxy_port).await.unwrap();
    });

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    let client = reqwest::Client::new();
    
    // Test /health
    let proxy_url = format!("http://127.0.0.1:{}/health", proxy_port);
    let response = client.get(&proxy_url)
        .send()
        .await
        .expect("Failed to send request to proxy");
    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().await.unwrap();
    assert_eq!(body["status"], "ok");

    // Test /v1/health
    let proxy_url_v1 = format!("http://127.0.0.1:{}/v1/health", proxy_port);
    let response_v1 = client.get(&proxy_url_v1)
        .send()
        .await
        .expect("Failed to send request to proxy");
    assert_eq!(response_v1.status(), 200);
    let body_v1: serde_json::Value = response_v1.json().await.unwrap();
    assert_eq!(body_v1["status"], "ok");
}

#[tokio::test]
async fn test_chat_completion_validation_integration() {
    let mock_server = MockServer::start().await;
    
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let proxy_port = listener.local_addr().unwrap().port();
    drop(listener);

    let config = BackendConfig {
        url: mock_server.uri(),
        model: "test-model".to_string(),
        models: vec!["test-model".to_string()],
    };
    let backend = Arc::new(LlamaBackend::new(config));
    
    tokio::spawn(async move {
        run_proxy(backend, proxy_port).await.unwrap();
    });

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    let client = reqwest::Client::new();
    let proxy_url = format!("http://127.0.0.1:{}/v1/chat/completions", proxy_port);
    
    // 1. Missing model
    let request_no_model = json!({
        "messages": [
            {"role": "user", "content": "hi"}
        ]
    });
    let response = client.post(&proxy_url)
        .json(&request_no_model)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 400);
    let body: serde_json::Value = response.json().await.unwrap();
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert!(body["error"]["message"].as_str().unwrap().contains("model"));

    // 2. Missing messages
    let request_no_messages = json!({
        "model": "test-model"
    });
    let response = client.post(&proxy_url)
        .json(&request_no_messages)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 400);
    let body: serde_json::Value = response.json().await.unwrap();
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert!(body["error"]["message"].as_str().unwrap().contains("messages"));
}

#[tokio::test]
async fn test_chat_completion_backend_error_mapping() {
    let mock_server = MockServer::start().await;
    
    // Mock a backend error (e.g., context length exceeded)
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(400)
            .set_body_json(json!({
                "type": "exceed_context_size_error",
                "message": "The message is too long for the context window."
            })))
        .mount(&mock_server)
        .await;

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let proxy_port = listener.local_addr().unwrap().port();
    drop(listener);

    let config = BackendConfig {
        url: mock_server.uri(),
        model: "test-model".to_string(),
        models: vec!["test-model".to_string()],
    };
    let backend = Arc::new(LlamaBackend::new(config));
    
    tokio::spawn(async move {
        run_proxy(backend, proxy_port).await.unwrap();
    });

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    let client = reqwest::Client::new();
    let proxy_url = format!("http://127.0.0.1:{}/v1/chat/completions", proxy_port);
    
    let request = CreateChatCompletionRequest {
        model: "test-model".to_string(),
        messages: vec![
            ChatMessage {
                role: MessageRole::User,
                content: "Very long message...".to_string(),
                ..Default::default()
            }
        ],
        ..Default::default()
    };

    let response = client.post(&proxy_url)
        .json(&request)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 400);
    let body: serde_json::Value = response.json().await.unwrap();
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert_eq!(body["error"]["code"].as_str().unwrap(), "context_length_exceeded");
    assert!(body["error"]["message"].as_str().unwrap().contains("context window"));
}

#[tokio::test]
async fn test_not_found_endpoint() {
    let mock_server = MockServer::start().await;
    
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let proxy_port = listener.local_addr().unwrap().port();
    drop(listener);

    let config = BackendConfig {
        url: mock_server.uri(),
        model: "test-model".to_string(),
        models: vec!["test-model".to_string()],
    };
    let backend = Arc::new(LlamaBackend::new(config));
    
    tokio::spawn(async move {
        run_proxy(backend, proxy_port).await.unwrap();
    });

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    let client = reqwest::Client::new();
    let proxy_url = format!("http://127.0.0.1:{}/v1/unknown", proxy_port);
    
    let response = client.get(&proxy_url)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 404);
    let body: serde_json::Value = response.json().await.unwrap();
    assert_eq!(body["error"]["type"], "not_found_error");
    assert!(body["error"]["message"].as_str().unwrap().contains("Unknown endpoint"));
}
