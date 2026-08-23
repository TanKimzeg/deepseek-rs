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
                    content: "Hi".into(),
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
                content: "Hi".into(),
                name: None,
            })
            .thinking(thinking)
            .reasoning_effort(ReasoningEffort::Max)
            .build();
        // API no longer rejects reasoning_effort with disabled thinking
        assert!(req.is_ok());
    }

    #[test]
    fn user_content_text_serializes_as_string() {
        let content = UserContent::Text("Hello".to_string());
        let value = serde_json::to_value(&content).unwrap();
        assert_eq!(value, json!("Hello"));
    }

    #[test]
    fn user_content_parts_serializes_as_array() {
        let content = UserContent::Parts(vec![
            UserContentPart::text("What is in this image?"),
            UserContentPart::file_id("file-api-abc123"),
        ]);
        let value = serde_json::to_value(&content).unwrap();
        assert_eq!(
            value,
            json!([
                {"type": "text", "text": "What is in this image?"},
                {"type": "file", "file_id": "file-api-abc123"}
            ])
        );
    }

    #[test]
    fn user_content_image_url_serializes() {
        let content = UserContent::image_url("https://example.com/image.jpg");
        let value = serde_json::to_value(&content).unwrap();
        assert_eq!(
            value,
            json!([
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image.jpg"}
                }
            ])
        );
    }

    #[test]
    fn user_content_image_url_with_detail_serializes() {
        let content =
            UserContent::image_url_with_detail("https://example.com/img.png", ImageDetail::Low);
        let value = serde_json::to_value(&content).unwrap();
        assert_eq!(
            value,
            json!([
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/img.png", "detail": "low"}
                }
            ])
        );
    }

    #[test]
    fn user_content_file_data_serializes() {
        let content = UserContent::file_data("data:image/jpeg;base64,abc123", "photo.jpg");
        let value = serde_json::to_value(&content).unwrap();
        assert_eq!(
            value,
            json!([
                {
                    "type": "file",
                    "file_data": "data:image/jpeg;base64,abc123",
                    "filename": "photo.jpg"
                }
            ])
        );
    }

    #[test]
    fn user_content_from_str_into() {
        let content: UserContent = "Hello".into();
        let value = serde_json::to_value(&content).unwrap();
        assert_eq!(value, json!("Hello"));
    }

    #[test]
    fn user_content_from_string_into() {
        let content: UserContent = "Hello".to_string().into();
        let value = serde_json::to_value(&content).unwrap();
        assert_eq!(value, json!("Hello"));
    }

    #[test]
    fn user_content_from_vec_parts_into() {
        let parts = vec![
            UserContentPart::text("Hi"),
            UserContentPart::file_id("file-xxx"),
        ];
        let content: UserContent = parts.into();
        let value = serde_json::to_value(&content).unwrap();
        assert!(value.is_array());
    }

    #[test]
    fn chat_message_user_with_text_content() {
        let msg = ChatMessage::User {
            content: "Hello".into(),
            name: None,
        };
        let value = serde_json::to_value(&msg).unwrap();
        assert_eq!(value.get("role"), Some(&json!("user")));
        assert_eq!(value.get("content"), Some(&json!("Hello")));
    }

    #[test]
    fn chat_message_user_with_multimodal_content() {
        let msg = ChatMessage::User {
            content: UserContent::Parts(vec![
                UserContentPart::text("Describe this image"),
                UserContentPart::image_url("https://example.com/img.jpg"),
            ]),
            name: None,
        };
        let value = serde_json::to_value(&msg).unwrap();
        assert_eq!(value.get("role"), Some(&json!("user")));
        assert!(value.get("content").unwrap().is_array());
        let content = value.get("content").unwrap().as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"], json!("text"));
        assert_eq!(content[1]["type"], json!("image_url"));
    }

    #[test]
    fn user_content_deserializes_from_string() {
        let content: UserContent = serde_json::from_value(json!("Hello")).unwrap();
        assert_eq!(content, UserContent::Text("Hello".to_string()));
    }

    #[test]
    fn user_content_part_file_with_id_deserializes() {
        let part: UserContentPart =
            serde_json::from_value(json!({"type": "file", "file_id": "file-api-abc123"})).unwrap();
        assert_eq!(part, UserContentPart::file_id("file-api-abc123"));
    }

    #[test]
    fn user_content_part_file_with_data_and_filename_deserializes() {
        let part: UserContentPart = serde_json::from_value(json!({
            "type": "file",
            "file_data": "data:image/jpeg;base64,abc123",
            "filename": "photo.jpg"
        }))
        .unwrap();
        assert_eq!(
            part,
            UserContentPart::file_data("data:image/jpeg;base64,abc123", "photo.jpg")
        );
    }

    #[test]
    fn user_content_part_image_url_without_detail_deserializes() {
        let part: UserContentPart = serde_json::from_value(json!({
            "type": "image_url",
            "image_url": {"url": "https://example.com/img.png"}
        }))
        .unwrap();
        assert_eq!(
            part,
            UserContentPart::image_url("https://example.com/img.png")
        );
    }

    #[test]
    fn chat_message_user_multimodal_round_trips() {
        let msg = ChatMessage::User {
            content: UserContent::Parts(vec![
                UserContentPart::text("What is in this image?"),
                UserContentPart::file_id("file-api-abc123"),
            ]),
            name: None,
        };
        let value = serde_json::to_value(&msg).unwrap();
        let back: ChatMessage = serde_json::from_value(value).unwrap();
        assert_eq!(back, msg);
    }

    #[test]
    fn chat_message_user_with_file_id_content() {
        let msg = ChatMessage::User {
            content: UserContent::file_id("file-api-abc123"),
            name: None,
        };
        let value = serde_json::to_value(&msg).unwrap();
        assert_eq!(value.get("role"), Some(&json!("user")));
        let content = value.get("content").unwrap().as_array().unwrap();
        assert_eq!(content[0]["type"], json!("file"));
        assert_eq!(content[0]["file_id"], json!("file-api-abc123"));
    }
}
