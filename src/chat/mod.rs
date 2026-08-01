//! Chat completion request/response models.
//!
//! This module contains the data structures for the `/chat/completions` API
//! and re-exports streaming helpers from the request implementation.
use crate::DeepSeekClient;
use serde::{Deserialize, Serialize};

pub mod request;
pub mod response;
pub mod stream;

pub use stream::{ChatStreamBlocking, ChatStreamItem};

/// Helper to skip serialization of empty `Vec` fields wrapped in `Option`.
pub(crate) fn is_none_or_empty_vec<T>(opt: &Option<Vec<T>>) -> bool {
    opt.as_ref().map(|v| v.is_empty()).unwrap_or(true)
}

/// Non-streaming chat completion response type.
pub type Chat = response::ChatGeneric<response::ChatChoice>;

/// Streaming chat completion response type (SSE chunks).
pub type ChatStream = response::ChatGeneric<response::ChatChoiceStream>;

#[cfg(test)]
mod tests {
    use crate::{DEFAULT_BASE_URL, DeepSeekClient};

    use super::request::*;
    use super::response::*;
    use serde_json::{Value, json};

    #[test]
    fn response_format_serializes_to_json_object() {
        let format = ResponseFormat::json_object();
        let value = serde_json::to_value(format).unwrap();
        assert_eq!(value, json!({"type": "json_object"}));
    }

    #[test]
    fn stop_supports_string_and_array() {
        let single = Stop::from("END");
        let many = Stop::from(vec!["END", "STOP"]);

        let single_value = serde_json::to_value(single).unwrap();
        let many_value = serde_json::to_value(many).unwrap();

        assert_eq!(single_value, json!("END"));
        assert_eq!(many_value, json!(["END", "STOP"]));

        let single_back: Stop = serde_json::from_value(json!("END")).unwrap();
        let many_back: Stop = serde_json::from_value(json!(["A", "B"])).unwrap();
        assert!(matches!(single_back, Stop::One(_)));
        assert!(matches!(many_back, Stop::Many(_)));

        let none_back: Option<Stop> = serde_json::from_value(Value::Null).unwrap();
        assert!(none_back.is_none());
    }

    #[test]
    fn tool_choice_serializes_simple_and_named() {
        let simple = ToolChoice::Simple(ChatToolChoice::Auto);
        let simple_value = serde_json::to_value(simple).unwrap();
        assert_eq!(simple_value, json!("auto"));

        let named = ToolChoice::named(json!({"name": "get_weather"}));
        let named_value = serde_json::to_value(named).unwrap();
        assert_eq!(
            named_value,
            json!({"type": "function", "function": {"name": "get_weather"}})
        );
    }

    #[test]
    fn chat_message_serializes_role_and_omits_prefix_by_default() {
        let message = ChatMessage::Assistant {
            content: Some("Hello".to_string()),
            name: None,
            tool_calls: None,
        };
        let value = serde_json::to_value(message).unwrap();
        assert_eq!(value.get("role"), Some(&json!("assistant")));
        assert_eq!(value.get("content"), Some(&json!("Hello")));
        assert!(value.get("reasoning_content").is_none());
    }

    #[test]
    fn response_tool_call_type_serializes_as_function() {
        let call = ToolCall::new("call_i", "get_weather", "{}");
        let value = serde_json::to_value(call).unwrap();
        assert_eq!(value.get("type"), Some(&json!("function")));
    }

    #[test]
    fn builder_validation_rejects_out_of_range_values() {
        fn base_builder() -> ChatRequestBuilder {
            ChatRequestBuilder::default()
                .model("deepseek-v4-pro")
                .message(ChatMessage::User {
                    content: "Hi".to_string(),
                    name: None,
                })
        }

        let too_hot = base_builder().temperature(2.5).build();
        assert!(too_hot.is_err());

        let bad_top_p = base_builder().top_p(1.1).build();
        assert!(bad_top_p.is_err());

        let bad_top_logprobs = base_builder().top_logprobs(21_u32).logprobs(true).build();
        assert!(bad_top_logprobs.is_err());

        let missing_logprobs = base_builder().top_logprobs(2_u32).build();
        assert!(missing_logprobs.is_err());
    }

    #[test]
    fn thinking_struct_serializes_type() {
        let thinking = Thinking::disabled();
        let value = serde_json::to_value(&thinking).unwrap();
        assert_eq!(value.get("type"), Some(&json!("disabled")));

        let req = ChatRequestBuilder::default()
            .client(DeepSeekClient::new("sk-...", DEFAULT_BASE_URL.clone()))
            .model("deepseek-v4-flash")
            .message(ChatMessage::User {
                content: "Hi".to_string(),
                name: None,
            })
            .thinking(thinking)
            .reasoning_effort(ReasoningEffort::Max)
            .build();
        // API no longer rejects reasoning_effort with disabled thinking
        assert!(req.is_ok());
    }
}
