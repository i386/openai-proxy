# OpenAI to llama.cpp Type Mapping

This document explains how types and data are mapped between OpenAI API format and the internal llama.cpp format in this proxy.

## Request Mapping (OpenAI → llama.cpp)

When a request is sent to `/v1/chat/completions`, it is mapped from the OpenAI `CreateChatCompletionRequest` to the llama.cpp server's `/v1/chat/completions` input format.

| OpenAI Field | llama.cpp Field | Mapping Logic / Notes |
|--------------|-----------------|-----------------------|
| `model` | `model` | Passed directly. |
| `messages` | `messages` | Array of objects with `role` and `content`. |
| `messages[].role` | `messages[].role` | Mapped directly: `system`, `user`, `assistant`, `tool`, `developer`. |
| `temperature` | `temperature` | Passed if provided. |
| `top_p` | `top_p` | Passed if provided. |
| `max_tokens` | `max_tokens` | Passed if provided. |
| `stream` | `stream` | Boolean, defaults to `false` unless explicitly set or using the streaming endpoint. |
| `stop` | `stop` | Passed if provided (string or array of strings). |
| `n` | `n` | Passed if provided. |
| `user` | `user` | Passed if provided. |

## Response Mapping (llama.cpp → OpenAI)

Successful responses from llama.cpp are mapped back to the OpenAI `CreateChatCompletionResponse` format.

| llama.cpp Field | OpenAI Field | Mapping Logic / Notes |
|-----------------|--------------|-----------------------|
| `choices` | `choices` | Array of completion choices. |
| `choices[].message` | `choices[].message` | Contains `role` (fixed as `assistant`) and `content`. |
| `choices[].finish_reason`| `choices[].finish_reason` | Passed directly (e.g., `stop`, `length`). |
| `choices[].index` | `choices[].index` | Passed directly or defaults to 0. |
| `usage` | `usage` | Token counts: `prompt_tokens`, `completion_tokens`, `total_tokens`. |
| (Generated) | `id` | Generated as `chatcmpl-<uuid>`. |
| (Generated) | `object` | Fixed as `chat.completion`. |
| (Generated) | `created` | Current Unix timestamp. |

## Error Mapping (llama.cpp → OpenAI)

Errors from llama.cpp are mapped to OpenAI-compatible error objects. The mapping depends on both the HTTP status code and the `type` field returned by llama.cpp.

### Complete Error Mapping Table

| llama.cpp Type | llama.cpp HTTP | OpenAI Type | OpenAI Code | Status |
|----------------|----------------|-------------|-------------|--------|
| `invalid_request_error` | 400 | `invalid_request_error` | `invalid_value` | ✅ DIRECT |
| `authentication_error` | 401 | `authentication_error` | `invalid_api_key` | ✅ DIRECT |
| `not_found_error` | 404 | `invalid_request_error` | `model_not_found` | ✅ Mapped to 400 |
| `server_error` | 500 | `server_error` | `internal_server_error` | ✅ DIRECT |
| `permission_error` | 403 | `permission_error` | `insufficient_quota` | ✅ DIRECT |
| `not_supported_error` | 501 | `invalid_request_error` | `unsupported_model_feature` | ⚠️ Mapped to 400 |
| `unavailable_error` | 503 | `server_error` | `service_unavailable` | ✅ Mapped to 500 |
| `exceed_context_size_error` | 400 | `invalid_request_error` | `context_length_exceeded` | ✅ DIRECT |

### Mapping Example

**llama.cpp Error Response:**
```json
{
  "code": 400,
  "message": "\"prompt\" is required",
  "type": "invalid_request_error"
}
```

**OpenAI-Compatible Proxy Response (400 Bad Request):**
```json
{
  "error": {
    "message": "\"prompt\" is required",
    "type": "invalid_request_error",
    "code": "invalid_value",
    "param": "prompt"
  }
}
```

