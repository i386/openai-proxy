use reqwest::StatusCode;
use serde_json::Value;
use crate::backend::BackendError;

/// Map llama.cpp error type to BackendError
pub fn map_llama_error(status: StatusCode, body: Value) -> BackendError {
    let message = body.get("message")
        .and_then(|m| m.as_str())
        .unwrap_or("Unknown error")
        .to_string();
    
    let llama_type = body.get("type")
        .and_then(|t| t.as_str())
        .unwrap_or("");

    match (status.as_u16(), llama_type) {
        (400, "invalid_request_error") => BackendError::invalid_request(message, None),
        (401, "authentication_error") => BackendError::authentication(message),
        (404, "not_found_error") => BackendError::not_found(message),
        (500, "server_error") => BackendError::internal(message),
        (403, "permission_error") => BackendError::permission(message),
        (501, "not_supported_error") => BackendError::unsupported_feature(message),
        (503, "unavailable_error") => BackendError::service_unavailable(message),
        (400, "exceed_context_size_error") => BackendError::context_length_exceeded(message),
        // Fallback based on status code if type doesn't match
        (400, _) => BackendError::invalid_request(message, None),
        (401, _) => BackendError::authentication(message),
        (403, _) => BackendError::permission(message),
        (404, _) => BackendError::not_found(message),
        (429, _) => BackendError::rate_limit(message),
        (503, _) => BackendError::service_unavailable(message),
        _ => BackendError::internal(message),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::StatusCode;
    use serde_json::json;

    #[test]
    fn test_error_mapping() {
        let mappings = vec![
            (StatusCode::BAD_REQUEST, json!({"type": "invalid_request_error", "message": "error"}), "invalid_request_error", "invalid_value"),
            (StatusCode::UNAUTHORIZED, json!({"type": "authentication_error", "message": "error"}), "authentication_error", "invalid_api_key"),
            (StatusCode::NOT_FOUND, json!({"type": "not_found_error", "message": "error"}), "invalid_request_error", "model_not_found"),
            (StatusCode::INTERNAL_SERVER_ERROR, json!({"type": "server_error", "message": "error"}), "server_error", "internal_server_error"),
            (StatusCode::FORBIDDEN, json!({"type": "permission_error", "message": "error"}), "permission_error", "insufficient_quota"),
            (StatusCode::NOT_IMPLEMENTED, json!({"type": "not_supported_error", "message": "error"}), "invalid_request_error", "unsupported_model_feature"),
            (StatusCode::SERVICE_UNAVAILABLE, json!({"type": "unavailable_error", "message": "error"}), "server_error", "service_unavailable"),
            (StatusCode::BAD_REQUEST, json!({"type": "exceed_context_size_error", "message": "error"}), "invalid_request_error", "context_length_exceeded"),
        ];

        for (status, body, expected_type, expected_code) in mappings {
            let backend_error = map_llama_error(status, body);
            let response = backend_error.to_error_response();
            assert_eq!(response.error.r#type, expected_type);
            assert_eq!(response.error.code, Some(expected_code.to_string()));
        }
    }
}
