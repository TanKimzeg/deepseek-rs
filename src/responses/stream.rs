//! Responses API streaming and request implementation for `/responses`.
use crate::DeepSeekRequest;
use crate::error::DeepSeekError;
use crate::{api_post, api_request_stream, consume_sse, spawn_blocking_stream};

use super::request::*;
use super::response::*;
use reqwest::Method;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

/// Stream item produced by Responses API streaming.
pub type ResponsesStreamItem = Result<ResponsesStreamEvent, DeepSeekError>;

/// Blocking iterator over streamed Responses API events.
pub struct ResponsesStreamBlocking {
    pub rx: std::sync::mpsc::Receiver<ResponsesStreamItem>,
}

impl Iterator for ResponsesStreamBlocking {
    type Item = ResponsesStreamItem;

    fn next(&mut self) -> Option<Self::Item> {
        self.rx.recv().ok()
    }
}

/// A semantic server-sent event emitted by the Responses API stream.
///
/// Each event carries a `sequence_number`. The final event is `response.completed` /
/// `response.incomplete` / `response.failed`; there is no `data: [DONE]` message.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum ResponsesStreamEvent {
    /// The first event; the response has been created with status `in_progress`.
    #[serde(rename = "response.created")]
    ResponseCreated {
        sequence_number: u64,
        response: Response,
    },
    /// The response is being generated.
    #[serde(rename = "response.in_progress")]
    ResponseInProgress {
        sequence_number: u64,
        response: Response,
    },
    /// An output item starts.
    #[serde(rename = "response.output_item.added")]
    ResponseOutputItemAdded {
        sequence_number: u64,
        output_index: u64,
        item: OutputItem,
    },
    /// An output item completes.
    #[serde(rename = "response.output_item.done")]
    ResponseOutputItemDone {
        sequence_number: u64,
        output_index: u64,
        item: OutputItem,
    },
    /// A content part within an output item starts.
    #[serde(rename = "response.content_part.added")]
    ResponseContentPartAdded {
        sequence_number: u64,
        item_id: String,
        output_index: u64,
        content_index: u64,
        part: ContentPart,
    },
    /// A content part within an output item completes.
    #[serde(rename = "response.content_part.done")]
    ResponseContentPartDone {
        sequence_number: u64,
        item_id: String,
        output_index: u64,
        content_index: u64,
        part: ContentPart,
    },
    /// Incremental chain-of-thought text.
    #[serde(rename = "response.reasoning_text.delta")]
    ResponseReasoningTextDelta {
        sequence_number: u64,
        item_id: String,
        output_index: u64,
        content_index: u64,
        delta: String,
    },
    /// The full chain-of-thought text.
    #[serde(rename = "response.reasoning_text.done")]
    ResponseReasoningTextDone {
        sequence_number: u64,
        item_id: String,
        output_index: u64,
        content_index: u64,
        text: String,
    },
    /// Incremental output text.
    #[serde(rename = "response.output_text.delta")]
    ResponseOutputTextDelta {
        sequence_number: u64,
        item_id: String,
        output_index: u64,
        content_index: u64,
        delta: String,
    },
    /// The full output text.
    #[serde(rename = "response.output_text.done")]
    ResponseOutputTextDone {
        sequence_number: u64,
        item_id: String,
        output_index: u64,
        content_index: u64,
        text: String,
    },
    /// Incremental function call arguments.
    #[serde(rename = "response.function_call_arguments.delta")]
    ResponseFunctionCallArgumentsDelta {
        sequence_number: u64,
        item_id: String,
        output_index: u64,
        delta: String,
    },
    /// The full function call arguments.
    #[serde(rename = "response.function_call_arguments.done")]
    ResponseFunctionCallArgumentsDone {
        sequence_number: u64,
        item_id: String,
        output_index: u64,
        arguments: String,
    },
    /// Incremental custom tool call (`apply_patch`) input.
    #[serde(rename = "response.custom_tool_call_input.delta")]
    ResponseCustomToolCallInputDelta {
        sequence_number: u64,
        item_id: String,
        output_index: u64,
        delta: String,
    },
    /// The full custom tool call (`apply_patch`) input.
    #[serde(rename = "response.custom_tool_call_input.done")]
    ResponseCustomToolCallInputDone {
        sequence_number: u64,
        item_id: String,
        output_index: u64,
        input: String,
    },
    /// A server-side web search tool call is starting.
    #[serde(rename = "response.web_search_call.in_progress")]
    ResponseWebSearchCallInProgress {
        sequence_number: u64,
        item_id: String,
        output_index: u64,
    },
    /// A server-side web search tool call is searching.
    #[serde(rename = "response.web_search_call.searching")]
    ResponseWebSearchCallSearching {
        sequence_number: u64,
        item_id: String,
        output_index: u64,
    },
    /// A server-side web search tool call completed.
    #[serde(rename = "response.web_search_call.completed")]
    ResponseWebSearchCallCompleted {
        sequence_number: u64,
        item_id: String,
        output_index: u64,
    },
    /// The final event when the response completes normally, carrying the full response object.
    #[serde(rename = "response.completed")]
    ResponseCompleted {
        sequence_number: u64,
        response: Response,
    },
    /// The final event when the response is truncated, carrying the full response object.
    #[serde(rename = "response.incomplete")]
    ResponseIncomplete {
        sequence_number: u64,
        response: Response,
    },
    /// The final event when the response fails, carrying the full response object with error details.
    #[serde(rename = "response.failed")]
    ResponseFailed {
        sequence_number: u64,
        response: Response,
    },
    /// An unrecognized event type, tolerated during deserialization.
    #[serde(other)]
    Unknown,
}

impl ResponsesStreamEvent {
    /// Whether this event is one of the terminal events (`response.completed` /
    /// `response.incomplete` / `response.failed`).
    #[allow(clippy::match_like_matches_macro)] // non_exhaustive enum requires a catch-all arm
    pub fn is_terminal(&self) -> bool {
        match self {
            ResponsesStreamEvent::ResponseCompleted { .. }
            | ResponsesStreamEvent::ResponseIncomplete { .. }
            | ResponsesStreamEvent::ResponseFailed { .. } => true,
            _ => false,
        }
    }

    /// The full `response` object carried by created / in-progress / terminal events.
    pub fn response(&self) -> Option<&Response> {
        match self {
            ResponsesStreamEvent::ResponseCreated { response, .. }
            | ResponsesStreamEvent::ResponseInProgress { response, .. }
            | ResponsesStreamEvent::ResponseCompleted { response, .. }
            | ResponsesStreamEvent::ResponseIncomplete { response, .. }
            | ResponsesStreamEvent::ResponseFailed { response, .. } => Some(response),
            _ => None,
        }
    }

    /// The incremental delta text carried by `*_text.delta` / `*_arguments.delta` events.
    pub fn delta(&self) -> Option<&str> {
        match self {
            ResponsesStreamEvent::ResponseOutputTextDelta { delta, .. }
            | ResponsesStreamEvent::ResponseReasoningTextDelta { delta, .. }
            | ResponsesStreamEvent::ResponseFunctionCallArgumentsDelta { delta, .. }
            | ResponsesStreamEvent::ResponseCustomToolCallInputDelta { delta, .. } => Some(delta),
            _ => None,
        }
    }
}

impl DeepSeekRequest for ResponsesRequest {
    type Response = Response;
    type StreamItem = ResponsesStreamItem;
    type BlockingStream = ResponsesStreamBlocking;

    async fn send(self) -> Result<Response, DeepSeekError> {
        let client = self.client.clone();
        api_post("/responses", &self, client).await
    }

    async fn stream(self) -> Result<mpsc::Receiver<ResponsesStreamItem>, DeepSeekError> {
        let mut request = self;
        request.stream = Some(true);

        let client = request.client.clone();
        let event_source = api_request_stream(
            Method::POST,
            "/responses",
            |builder| builder.json(&request),
            client,
        )
        .await?;

        Ok(consume_sse(event_source, |data| {
            serde_json::from_str::<ResponsesStreamEvent>(&data)
                .map(Some)
                .map_err(|err| DeepSeekError::decode(err.to_string(), data))
        }))
    }

    fn stream_blocking(self) -> Result<ResponsesStreamBlocking, DeepSeekError> {
        let rx = spawn_blocking_stream(self.stream())?;
        Ok(ResponsesStreamBlocking { rx })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DEFAULT_BASE_URL, DeepSeekClient};

    #[test]
    fn deserialize_stream_event_dotted_names() {
        let created = serde_json::from_str::<ResponsesStreamEvent>(
            r#"{"type":"response.created","sequence_number":0,"response":{"id":"r_1","object":"response","created_at":1753000000,"status":"in_progress","model":"deepseek-v4-flash","output":[],"usage":null}}"#,
        )
        .unwrap();
        assert!(matches!(
            created,
            ResponsesStreamEvent::ResponseCreated { response, .. }
                if response.usage.is_none() && response.status == ResponseStatus::InProgress
        ));

        let delta = serde_json::from_str::<ResponsesStreamEvent>(
            r#"{"type":"response.output_text.delta","sequence_number":11,"item_id":"msg_1","output_index":1,"content_index":0,"delta":"Hello"}"#,
        )
        .unwrap();
        assert!(matches!(
            delta,
            ResponsesStreamEvent::ResponseOutputTextDelta { delta, .. } if delta == "Hello"
        ));

        let done = serde_json::from_str::<ResponsesStreamEvent>(
            r#"{"type":"response.output_item.done","sequence_number":20,"output_index":1,"item":{"type":"message","id":"msg_1","status":"completed","role":"assistant","content":[{"type":"output_text","text":"Hello"}]}}"#,
        )
        .unwrap();
        assert!(matches!(
            done,
            ResponsesStreamEvent::ResponseOutputItemDone { .. }
        ));

        let unknown = serde_json::from_str::<ResponsesStreamEvent>(
            r#"{"type":"something.new","sequence_number":99}"#,
        )
        .unwrap();
        assert!(matches!(unknown, ResponsesStreamEvent::Unknown));
    }

    #[test]
    fn stream_event_serializes_dotted_names() {
        let evt = ResponsesStreamEvent::ResponseCompleted {
            sequence_number: 20,
            response: Response {
                id: "r_1".to_string(),
                object: "response".to_string(),
                created_at: 1753000000,
                status: ResponseStatus::Completed,
                error: None,
                incomplete_details: None,
                model: "deepseek-v4-flash".to_string(),
                output: vec![],
                usage: None,
                store: None,
                parallel_tool_calls: None,
                previous_response_id: None,
            },
        };
        let value = serde_json::to_value(evt).unwrap();
        assert_eq!(
            value.get("type"),
            Some(&serde_json::json!("response.completed"))
        );
    }

    fn get_client() -> DeepSeekClient {
        DeepSeekClient::new(
            std::env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY is not set"),
            DEFAULT_BASE_URL.clone(),
        )
    }

    fn get_builder() -> ResponsesRequestBuilder {
        ResponsesRequestBuilder::default()
            .client(get_client())
            .model("deepseek-v4-flash")
            .instructions("You are a helpful assistant.")
            .reasoning(Reasoning::new(ReasoningEffort::None))
    }

    #[tokio::test]
    async fn responses_basic() {
        let req = get_builder()
            .input("Reply with exactly: OK")
            .max_output_tokens(64_u32)
            .build()
            .unwrap();
        let response = req.send().await.unwrap();
        println!("{:#?}", response);
        assert_eq!(response.object, "response");
        assert_eq!(response.status, ResponseStatus::Completed);
        assert!(!response.output_text().is_empty());
        assert!(response.usage.is_some());
    }

    #[tokio::test]
    async fn responses_thinking_mode() {
        let req = get_builder()
            .reasoning(Reasoning::new(ReasoningEffort::Low))
            .input("What is 2+2? Reply briefly.")
            .max_output_tokens(256_u32)
            .build()
            .unwrap();
        let response = req.send().await.unwrap();
        println!("{:#?}", response);
        let has_reasoning = response
            .output
            .iter()
            .any(|item| matches!(item, OutputItem::Reasoning { .. }));
        assert!(
            has_reasoning,
            "expected a reasoning output item in thinking mode"
        );
        assert!(!response.output_text().is_empty());
    }

    #[tokio::test]
    async fn responses_json_object() {
        let req = get_builder()
            .text(Text::new(TextFormat::json_object()))
            .input("Return a JSON object with a `city` field set to Hangzhou.")
            .max_output_tokens(128_u32)
            .build()
            .unwrap();
        let response = req.send().await.unwrap();
        println!("{:#?}", response);
        let parsed: serde_json::Value = serde_json::from_str(&response.output_text())
            .expect("output text should be valid JSON");
        assert_eq!(
            parsed.get("city").and_then(|c| c.as_str()),
            Some("Hangzhou")
        );
    }

    #[tokio::test]
    async fn responses_json_schema() {
        let req = get_builder()
            .text(Text::new(TextFormat::json_schema(
                "city_response",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "temperature": {"type": "number"}
                    },
                    "required": ["city", "temperature"]
                }),
            )))
            .input("What is the weather in Hangzhou? Say 24 degrees.")
            .max_output_tokens(128_u32)
            .build()
            .unwrap();
        let response = req.send().await.unwrap();
        println!("{:#?}", response);
        let parsed: serde_json::Value = serde_json::from_str(&response.output_text())
            .expect("output text should be valid JSON");
        assert_eq!(
            parsed.get("city").and_then(|c| c.as_str()),
            Some("Hangzhou")
        );
        assert!(parsed.get("temperature").and_then(|t| t.as_f64()).is_some());
    }

    #[tokio::test]
    async fn responses_input_item_list() {
        let req = get_builder()
            .input(vec![
                InputItem::user("Remember my name is Alice."),
                InputItem::assistant("Got it, Alice!"),
                InputItem::user("What is my name?"),
            ])
            .max_output_tokens(64_u32)
            .build()
            .unwrap();
        let response = req.send().await.unwrap();
        println!("{:#?}", response);
        assert!(response.output_text().contains("Alice"));
    }

    #[tokio::test]
    async fn responses_tool_call() {
        let tool = Tool::function(
            "get_weather",
            "Get the weather of a location.",
            Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            })),
        );
        let req = get_builder()
            .input("What is the weather in Hangzhou?")
            .tool(tool)
            .tool_choice(ToolChoice::named("get_weather"))
            .max_output_tokens(128_u32)
            .build()
            .unwrap();
        let response = req.send().await.unwrap();
        println!("{:#?}", response);
        let function_call = response
            .output
            .iter()
            .find(|item| matches!(item, OutputItem::FunctionCall { .. }))
            .expect("expected a function_call output item");
        let OutputItem::FunctionCall {
            name, arguments, ..
        } = function_call
        else {
            unreachable!()
        };
        assert_eq!(name, "get_weather");
        let parsed: serde_json::Value =
            serde_json::from_str(arguments).expect("arguments should be valid JSON");
        assert_eq!(
            parsed.get("location").and_then(|l| l.as_str()),
            Some("Hangzhou")
        );
    }

    #[tokio::test]
    async fn responses_multi_turn_tool_call() {
        let tool = Tool::function(
            "get_weather",
            "Get the weather of a location.",
            Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            })),
        );

        let first = get_builder()
            .input("What is the weather in Hangzhou?")
            .tool(tool.clone())
            .tool_choice(ToolChoice::named("get_weather"))
            .max_output_tokens(128_u32)
            .build()
            .unwrap();
        let first_resp = first.send().await.unwrap();
        let function_call = first_resp
            .output
            .iter()
            .find(|item| matches!(item, OutputItem::FunctionCall { .. }))
            .expect("expected a function_call output item");
        let OutputItem::FunctionCall {
            call_id,
            name,
            arguments,
            ..
        } = function_call
        else {
            unreachable!()
        };
        let call_id = call_id.clone().expect("function_call carries a call_id");
        let name = name.clone();
        let arguments = arguments.clone();

        let second = get_builder()
            .input(vec![
                InputItem::user("What is the weather in Hangzhou?"),
                InputItem::function_call(call_id.clone(), name, arguments),
                InputItem::function_call_output(call_id, "24°C, clear sky"),
            ])
            .tool(tool)
            .tool_choice(ToolChoice::auto())
            .max_output_tokens(128_u32)
            .build()
            .unwrap();
        let second_resp = second.send().await.unwrap();
        println!("{:#?}", second_resp);
        assert!(second_resp.output_text().contains("24°C"));
    }

    #[tokio::test]
    async fn responses_web_search() {
        let req = get_builder()
            .tool(Tool::web_search())
            .tool_choice(ToolChoice::web_search())
            .input("Search the web and report the weather in Hangzhou.")
            .max_output_tokens(256_u32)
            .build()
            .unwrap();
        let response = req.send().await.unwrap();
        println!("{:#?}", response);
        assert_eq!(response.status, ResponseStatus::Completed);
        // Forcing web_search executes the search on the server and returns the
        // `web_search_call` items; a final assistant message is not guaranteed.
        assert!(
            response
                .output
                .iter()
                .any(|item| matches!(item, OutputItem::WebSearchCall { .. })),
            "expected web_search_call output items"
        );
    }

    #[tokio::test]
    async fn responses_stream_async() {
        let req = get_builder()
            .input("Count from 1 to 5.")
            .max_output_tokens(128_u32)
            .build()
            .unwrap();

        let mut rx = req.stream().await.unwrap();
        let mut text = String::new();
        let mut saw_terminal = false;
        while let Some(item) = rx.recv().await {
            match item {
                Ok(evt) => {
                    if let Some(delta) = evt.delta() {
                        text.push_str(delta);
                    }
                    if evt.is_terminal() {
                        saw_terminal = true;
                        let response = evt.response().expect("terminal event carries the response");
                        assert!(response.usage.is_some());
                    }
                }
                Err(err) => eprintln!("Error>\t {err:?}"),
            }
        }
        println!("Model>\t {text}");
        assert!(!text.is_empty());
        assert!(saw_terminal, "expected a terminal stream event");
    }

    #[test]
    fn responses_stream_blocking() {
        let req = get_builder()
            .input("Count from 1 to 5.")
            .max_output_tokens(128_u32)
            .build()
            .unwrap();

        let stream = req.stream_blocking().unwrap();
        let mut text = String::new();
        for item in stream.take(500) {
            match item {
                Ok(evt) => {
                    if let Some(delta) = evt.delta() {
                        text.push_str(delta);
                    }
                }
                Err(err) => eprintln!("Error>\t {err:?}"),
            }
        }
        println!("Model>\t {text}");
        assert!(!text.is_empty());
    }
}
