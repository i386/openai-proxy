use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use tokio::net::TcpListener;
use openai_proxy::{
    run_proxy, LlamaBackend, BackendConfig, CreateChatCompletionRequest,
    ChatMessage, MessageRole,
};
use std::time::Duration;
use tokio::time::sleep;
use std::path::PathBuf;
use serde_json::json;

struct LlamaServer {
    child: Child,
    url: String,
}

impl Drop for LlamaServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}

async fn start_llama_server(model_path: &PathBuf, port: u16) -> LlamaServer {
    let child = Command::new("llama-server")
        .arg("-m")
        .arg(model_path)
        .arg("--port")
        .arg(port.to_string())
        .arg("--host")
        .arg("127.0.0.1")
        .arg("-ngl") // Offload to GPU if available
        .arg("999")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to start llama-server");

    let url = format!("http://127.0.0.1:{}", port);
    
    // Wait for server to be ready
    let client = reqwest::Client::new();
    let health_url = format!("{}/health", url);
    
    let mut ready = false;
    for _ in 0..60 { // Wait up to 60 seconds
        if let Ok(resp) = client.get(&health_url).send().await {
            if resp.status().is_success() {
                ready = true;
                break;
            }
        }
        sleep(Duration::from_secs(1)).await;
    }

    if !ready {
        panic!("llama-server failed to become ready within 60 seconds");
    }

    LlamaServer { child, url }
}

#[tokio::test]
async fn test_llama_live_integration_suite() {
    // 1. Download the model using `hf`
    let model_repo = "Qwen/Qwen2.5-0.5B-Instruct-GGUF";
    let model_file = "qwen2.5-0.5b-instruct-q4_k_m.gguf";
    let local_dir = "models/qwen2.5-0.5b-instruct";

    println!("Downloading model {} from {}...", model_file, model_repo);
    let status = Command::new("hf")
        .arg("download")
        .arg(model_repo)
        .arg(model_file)
        .arg("--local-dir")
        .arg(local_dir)
        .status()
        .expect("Failed to run hf download");

    assert!(status.success(), "hf download failed");

    let mut model_path = PathBuf::from(local_dir);
    model_path.push(model_file);

    // 2. Start llama-server
    let llama_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let llama_port = llama_listener.local_addr().unwrap().port();
    drop(llama_listener);

    println!("Starting llama-server on port {} with model {:?}...", llama_port, model_path);
    let _llama_server = start_llama_server(&model_path, llama_port).await;

    // 3. Start the OpenAI Proxy
    let proxy_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let proxy_port = proxy_listener.local_addr().unwrap().port();
    drop(proxy_listener);

    let config = BackendConfig {
        url: _llama_server.url.clone(),
        model: "qwen-live".to_string(),
        models: vec!["qwen-live".to_string(), "qwen-live-stream".to_string()],
    };
    let backend = Arc::new(LlamaBackend::new(config));
    
    let backend_clone = backend.clone();
    tokio::spawn(async move {
        run_proxy(backend_clone, proxy_port).await.unwrap();
    });

    sleep(Duration::from_millis(500)).await;

    let client = reqwest::Client::new();
    let proxy_base_url = format!("http://127.0.0.1:{}", proxy_port);

    // --- Test Chat Completion ---
    {
        let proxy_url = format!("{}/v1/chat/completions", proxy_base_url);
        let request = CreateChatCompletionRequest {
            model: "qwen-live".to_string(),
            messages: vec![
                ChatMessage::new(
                    MessageRole::User,
                    "Say 'Hello, I am a live model' and nothing else.",
                )
            ],
            temperature: Some(0.0),
            ..Default::default()
        };

        println!("Sending request to proxy at {}...", proxy_url);
        let response = client.post(&proxy_url)
            .json(&request)
            .send()
            .await
            .expect("Failed to send request to proxy");

        assert_eq!(response.status(), 200);
        let body: serde_json::Value = response.json().await.unwrap();
        let content = body["choices"][0]["message"]["content"].as_str().unwrap();
        println!("Received response: {}", content);
        assert!(content.to_lowercase().contains("hello"));
        assert!(content.to_lowercase().contains("live model"));
        assert_eq!(body["model"], "qwen-live");
        
        // From mock tests: verify usage fields
        assert!(body["usage"]["total_tokens"].as_u64().is_some());
    }

    // --- Test Chat Completion Stream ---
    {
        let proxy_url = format!("{}/v1/chat/completions", proxy_base_url);
        let request = CreateChatCompletionRequest {
            model: "qwen-live-stream".to_string(),
            messages: vec![
                ChatMessage::new(
                    MessageRole::User,
                    "Say 'Hello, I am a live model streaming' and nothing else.",
                )
            ],
            stream: Some(true),
            temperature: Some(0.0),
            ..Default::default()
        };

        println!("Sending streaming request to proxy at {}...", proxy_url);
        let mut response = client.post(&proxy_url)
            .json(&request)
            .send()
            .await
            .expect("Failed to send request to proxy");

        assert_eq!(response.status(), 200);
        assert_eq!(response.headers()["content-type"], "text/event-stream");

        let mut full_content = String::new();
        while let Some(chunk) = response.chunk().await.expect("Failed to read chunk") {
            let text = String::from_utf8_lossy(&chunk);
            for line in text.lines() {
                if line.starts_with("data: ") && !line.contains("[DONE]") {
                    let json_str = &line[6..];
                    let body: serde_json::Value = serde_json::from_str(json_str).expect("Failed to parse SSE JSON");
                    if let Some(content) = body["choices"][0]["delta"]["content"].as_str() {
                        full_content.push_str(content);
                    }
                }
            }
        }
        
        println!("Received streaming content: {}", full_content);
        assert!(full_content.to_lowercase().contains("model"));
        assert!(full_content.to_lowercase().contains("streaming"));
    }

    // --- Test List Models ---
    {
        let proxy_url = format!("{}/v1/models", proxy_base_url);
        println!("Sending list models request to proxy at {}...", proxy_url);
        let response = client.get(&proxy_url)
            .send()
            .await
            .expect("Failed to send request to proxy");

        assert_eq!(response.status(), 200);
        let body: serde_json::Value = response.json().await.unwrap();
        println!("Received models: {:?}", body);
        let model_ids: Vec<&str> = body["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["id"].as_str().unwrap())
            .collect();
        assert!(!model_ids.is_empty());
        
        // List models should reflect backend models (llama-server usually provides this)
        // or at least have some data.
        assert_eq!(body["object"], "list");
    }

    // --- Test Health Checks (consolidated from mock tests) ---
    {
        // Test /health
        let health_url = format!("{}/health", proxy_base_url);
        let response = client.get(&health_url)
            .send()
            .await
            .expect("Failed to send request to proxy");
        assert_eq!(response.status(), 200);
        let body: serde_json::Value = response.json().await.unwrap();
        assert_eq!(body["status"], "ok");

        // Test /v1/health
        let health_v1_url = format!("{}/v1/health", proxy_base_url);
        let response_v1 = client.get(&health_v1_url)
            .send()
            .await
            .expect("Failed to send request to proxy");
        assert_eq!(response_v1.status(), 200);
        let body_v1: serde_json::Value = response_v1.json().await.unwrap();
        assert_eq!(body_v1["status"], "ok");
    }

    // --- Test /v1/responses Translation (OpenAI compatibility) ---
    {
        let proxy_url = format!("{}/v1/responses", proxy_base_url);
        let request = json!({
            "model": "qwen-live",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Say 'Hi, I am OpenAI compatible' and nothing else."
                        }
                    ]
                }
            ],
            "temperature": 0.0
        });

        println!("Sending /v1/responses request to proxy at {}...", proxy_url);
        let response = client.post(&proxy_url)
            .json(&request)
            .send()
            .await
            .expect("Failed to send request to proxy");

        assert_eq!(response.status(), 200);
        let body: serde_json::Value = response.json().await.unwrap();
        let content = body["choices"][0]["message"]["content"].as_str().unwrap();
        println!("Received /v1/responses response: {}", content);
        assert!(content.to_lowercase().contains("hi") || content.to_lowercase().contains("hello"));
    }

    // --- Test Multi-turn Conversation ---
    {
        let proxy_url = format!("{}/v1/chat/completions", proxy_base_url);
        let request = CreateChatCompletionRequest {
            model: "qwen-live".to_string(),
            messages: vec![
                ChatMessage::new(MessageRole::User, "My name is Alice."),
                ChatMessage::new(MessageRole::Assistant, "Hello Alice! How can I help you?"),
                ChatMessage::new(MessageRole::User, "What is my name? Say only the name."),
            ],
            temperature: Some(0.0),
            ..Default::default()
        };

        println!("Sending multi-turn request to proxy at {}...", proxy_url);
        let response = client.post(&proxy_url)
            .json(&request)
            .send()
            .await
            .expect("Failed to send request to proxy");

        assert_eq!(response.status(), 200);
        let body: serde_json::Value = response.json().await.unwrap();
        let content = body["choices"][0]["message"]["content"].as_str().unwrap();
        println!("Received multi-turn response: {}", content);
        assert!(content.contains("Alice"));
    }

    // --- Test Stop Sequences ---
    {
        let proxy_url = format!("{}/v1/chat/completions", proxy_base_url);
        let request = CreateChatCompletionRequest {
            model: "qwen-live".to_string(),
            messages: vec![
                ChatMessage::new(MessageRole::User, "Count from 1 to 5. Stop before 3."),
            ],
            stop: Some(vec!["3".to_string()]),
            temperature: Some(0.0),
            ..Default::default()
        };

        println!("Sending request with stop sequences to proxy at {}...", proxy_url);
        let response = client.post(&proxy_url)
            .json(&request)
            .send()
            .await
            .expect("Failed to send request to proxy");

        assert_eq!(response.status(), 200);
        let body: serde_json::Value = response.json().await.unwrap();
        let content = body["choices"][0]["message"]["content"].as_str().unwrap();
        println!("Received stop sequence response: {}", content);
        assert!(!content.contains('3'));
        assert!(!content.contains('4'));
    }

    // --- Test Max Tokens ---
    {
        let proxy_url = format!("{}/v1/chat/completions", proxy_base_url);
        let request = CreateChatCompletionRequest {
            model: "qwen-live".to_string(),
            messages: vec![
                ChatMessage::new(MessageRole::User, "Write a long poem about the sea."),
            ],
            max_tokens: Some(5),
            temperature: Some(0.0),
            ..Default::default()
        };

        println!("Sending request with max_tokens=5 to proxy at {}...", proxy_url);
        let response = client.post(&proxy_url)
            .json(&request)
            .send()
            .await
            .expect("Failed to send request to proxy");

        assert_eq!(response.status(), 200);
        let body: serde_json::Value = response.json().await.unwrap();
        let finish_reason = body["choices"][0]["finish_reason"].as_str().unwrap();
        println!("Received max_tokens response with finish_reason: {}", finish_reason);
        assert_eq!(finish_reason, "length");
    }

    // --- Test Parameter Aliasing & Role Mapping (OpenAI compatibility) ---
    {
        let proxy_url = format!("{}/v1/chat/completions", proxy_base_url);
        let request = json!({
            "model": "qwen-live",
            "messages": [
                {
                    "role": "developer",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "Say 'I understand my instructions' and nothing else."
                }
            ],
            "max_completion_tokens": 50,
            "temperature": 0.0
        });

        println!("Sending request with developer role and aliased tokens to proxy at {}...", proxy_url);
        let response = client.post(&proxy_url)
            .json(&request)
            .send()
            .await
            .expect("Failed to send request to proxy");

        assert_eq!(response.status(), 200);
        let body: serde_json::Value = response.json().await.unwrap();
        let content = body["choices"][0]["message"]["content"].as_str().unwrap();
        println!("Received aliased/mapped response: {}", content);
        assert!(content.to_lowercase().contains("understand"));
        assert!(content.to_lowercase().contains("instructions"));
    }

    // --- Test Validation Errors (consolidated from mock tests) ---
    {
        let proxy_url = format!("{}/v1/chat/completions", proxy_base_url);
        
        // Test missing model
        let request_no_model = json!({
            "messages": [{"role": "user", "content": "hi"}]
        });
        println!("Sending request without model to proxy at {}...", proxy_url);
        let response = client.post(&proxy_url)
            .json(&request_no_model)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 400);
        let body: serde_json::Value = response.json().await.unwrap();
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert!(body["error"]["message"].as_str().unwrap().contains("model"));

        // Test missing messages
        let request_no_messages = json!({
            "model": "qwen-live"
        });
        println!("Sending request without messages to proxy at {}...", proxy_url);
        let response = client.post(&proxy_url)
            .json(&request_no_messages)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 400);
        let body: serde_json::Value = response.json().await.unwrap();
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert!(body["error"]["message"].as_str().unwrap().contains("messages"));

        // Test invalid role (unrecognized by proxy or backend)
        let request_invalid_role = json!({
            "model": "qwen-live",
            "messages": [{"role": "super-user", "content": "hi"}]
        });
        println!("Sending request with invalid role to proxy at {}...", proxy_url);
        let response = client.post(&proxy_url)
            .json(&request_invalid_role)
            .send()
            .await
            .unwrap();
        // Since super-user isn't a valid MessageRole enum variant, it should fail at deserialization (400)
        assert_eq!(response.status(), 400);
    }

    // --- Test Not Found Endpoint (from mock tests) ---
    {
        let proxy_url = format!("{}/v1/unknown", proxy_base_url);
        let response = client.get(&proxy_url)
            .send()
            .await
            .unwrap();

        assert_eq!(response.status(), 404);
        let body: serde_json::Value = response.json().await.unwrap();
        assert_eq!(body["error"]["type"], "not_found_error");
        assert!(body["error"]["message"].as_str().unwrap().contains("Unknown endpoint"));
    }
}
