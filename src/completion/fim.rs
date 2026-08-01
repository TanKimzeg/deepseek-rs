//! FIM (Fill-In-the-Middle) completion models and request types.
//!
//! This endpoint is beta and requires the beta base URL:
//! `https://api.deepseek.com/beta`.
use std::collections::HashMap;

use crate::DeepSeekRequest;
use crate::chat::request::{Stop, StreamOptions, is_none_or_empty_stop};
use crate::chat::response::ChatGeneric;
use crate::error::DeepSeekError;
use crate::{DeepSeekClient, api_post, api_request_stream, consume_sse, spawn_blocking_stream};
use derive_builder::Builder;
use reqwest::Method;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

/// Non-streaming FIM completion response.
pub type Completion = ChatGeneric<CompletionChoice>;

/// FIM completion request payload.
#[derive(Clone, Debug, PartialEq, Serialize, Builder)]
#[builder(
    pattern = "owned",
    setter(into, strip_option),
    build_fn(validate = "Self::validate"),
    name = "FIMCompletionRequestBuilder"
)]
pub struct FIMCompletionRequest {
    #[serde(skip_serializing)]
    pub client: DeepSeekClient,

    /// Possible values: \[`deepseek-v4-pro`\]
    ///
    /// ID of the model to use.
    pub model: String,

    /// The prompt to generate completions for.
    pub prompt: String,

    /// Echo back the prompt in addition to the completion
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub echo: Option<bool>,

    /// Possible values: `<= 20`
    ///
    /// Include the log probabilities on the `logprobs` most likely output tokens,
    /// as well the chosen tokens. For example, if `logprobs` is 20, the API will return a list of the 20 most likely tokens.
    /// The API will always return the logprob of the sampled token, so there may be up to `logprobs+1` elements in the response.
    /// The maximum value for `logprobs` is 20.
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<u32>,

    /// The maximum number of tokens that can be generated in the completion.
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// Up to 16 sequences where the API will stop generating further tokens.
    /// The returned text will not contain the stop sequence.
    #[builder(default)]
    #[serde(skip_serializing_if = "is_none_or_empty_stop")]
    pub stop: Option<Stop>,

    /// Whether to stream back partial progress. If set, tokens will be sent as data-only server-sent events as they become available,
    /// with the stream terminated by a · message. [Example Python code](https://cookbook.openai.com/examples/how_to_stream_completions).
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    /// Options for streaming response. Only set this when you set `stream: true`.
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,

    /// The suffix that comes after a completion of inserted text.
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suffix: Option<String>,

    /// Possible values: `<= 2`
    ///
    /// Default value: `1`
    ///
    /// What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random,
    /// while lower values like 0.2 will make it more focused and deterministic.
    /// We generally recommend altering this or `top_p` but not both.
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    /// Possible values: `<= 1`
    ///
    /// Default value: `1`
    ///
    /// An alternative to sampling with temperature, called nucleus sampling,
    /// where the model considers the results of the tokens with top_p probability mass.
    /// So 0.1 means only the tokens comprising the top 10% probability mass are considered.
    /// We generally recommend altering this or `temperature` but not both.
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
}

impl FIMCompletionRequestBuilder {
    fn validate(&self) -> Result<(), String> {
        if let Some(temperature) = self.temperature.flatten()
            && !(0.0..=2.0).contains(&temperature)
        {
            return Err("temperature must be between 0 and 2".to_string());
        }
        if let Some(logprobs) = self.logprobs.flatten()
            && logprobs > 20
        {
            return Err("logprobs must be <= 20".to_string());
        }

        if let Some(top_p) = self.top_p.flatten()
            && !(0.0..=1.0).contains(&top_p)
        {
            return Err("top_p must be between 0 and 1".to_string());
        }

        if let Some(stream) = self.stream.flatten()
            && !stream
            && self.stream_options.is_some()
        {
            return Err("stream_options cannot be set when stream is false".to_string());
        }

        if let Some(stop) = self.stop.as_ref().and_then(|s| s.as_ref())
            && let Stop::Many(values) = stop
            && values.len() > 16
        {
            return Err("a maximum of 16 stop sequences are allowed".to_string());
        }

        Ok(())
    }
}

/// Completion choice the model generated for the input prompt.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct CompletionChoice {
    /// Possible values: [`stop`, `length`, `content_filter`, `insufficient_system_resource`]
    ///
    /// The reason the model stopped generating tokens.
    /// This will be `stop` if the model hit a natural stop point or a provided stop sequence,
    /// `length` if the maximum number of tokens specified in the request was reached,
    /// `content_filter` if content was omitted due to a flag from our content filters,
    /// or `insufficient_system_resource` if the request is interrupted due to insufficient resource of the inference system.
    pub finish_reason: FinishReason,
    pub index: u64,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Logprobs>,
}

/// Completion finish reason.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ContentFilter,
    InsufficientSystemResources,
}

/// Logprob details for completion tokens.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct Logprobs {
    pub text_offset: Vec<u64>,
    pub token_logprobs: Vec<f64>,
    pub tokens: Vec<String>,
    pub top_logprobs: Option<Vec<HashMap<String, f64>>>,
}
/// Streaming completion choice.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct CompletionChoiceStream {
    pub finish_reason: Option<FinishReason>,
    pub index: u64,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Logprobs>,
}
/// Streaming FIM completion response (SSE chunks).
pub type CompletionStream = ChatGeneric<CompletionChoiceStream>;
/// Stream item produced by FIM completion streaming.
pub type CompletionStreamItem = Result<CompletionStream, DeepSeekError>;
/// Blocking iterator over FIM completion streaming chunks.
pub struct CompletionStreamBlocking {
    rx: std::sync::mpsc::Receiver<CompletionStreamItem>,
}

impl Iterator for CompletionStreamBlocking {
    type Item = CompletionStreamItem;

    fn next(&mut self) -> Option<Self::Item> {
        self.rx.recv().ok()
    }
}
impl DeepSeekRequest for FIMCompletionRequest {
    type Response = Completion;
    type StreamItem = CompletionStreamItem;
    type BlockingStream = CompletionStreamBlocking;

    async fn send(self) -> Result<Self::Response, DeepSeekError> {
        let client = self.client.clone();
        api_post("/completions", &self, client).await
    }

    async fn stream(self) -> Result<mpsc::Receiver<Self::StreamItem>, DeepSeekError> {
        let mut request = self;
        request.stream = Some(true);

        let client = request.client.clone();
        let event_source = api_request_stream(
            Method::POST,
            "/completions",
            |builder| builder.json(&request),
            client,
        )
        .await?;

        Ok(consume_sse(event_source, |data| {
            serde_json::from_str::<CompletionStream>(&data)
                .map(Some)
                .map_err(|err| DeepSeekError::decode(err.to_string(), data))
        }))
    }

    fn stream_blocking(self) -> Result<CompletionStreamBlocking, DeepSeekError> {
        let rx = spawn_blocking_stream(self.stream())?;
        Ok(CompletionStreamBlocking { rx })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DEFAULT_BETA_BASE_URL;

    fn get_client() -> DeepSeekClient {
        DeepSeekClient::new(
            std::env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY is not set"),
            DEFAULT_BETA_BASE_URL.clone(),
        )
    }

    fn get_fim_builder() -> FIMCompletionRequestBuilder {
        FIMCompletionRequestBuilder::default()
            .client(get_client())
            .model("deepseek-v4-flash")
            .max_tokens(64_u32)
    }

    #[tokio::test]
    async fn test_fim_completion() {
        let fim_request = get_fim_builder()
            .prompt("def fib(a):")
            .suffix("    return fib(a-1) + fib(a-2)")
            .build()
            .unwrap();
        let response = fim_request.send().await.unwrap();
        println!("{:#?}", response);
        assert_eq!(response.object, "text_completion");
        assert_eq!(response.model, "deepseek-v4-flash");
        assert_eq!(response.choices.len(), 1);
    }

    #[tokio::test]
    async fn test_fim_completion_stream() {
        let fim_request = get_fim_builder()
            .prompt("def fib(a):")
            .suffix("    return fib(a-1) + fib(a-2)")
            .stream(true)
            .build()
            .unwrap();
        let mut stream = fim_request.stream().await.unwrap();
        while let Some(item) = stream.recv().await {
            match item {
                Ok(chunk) => println!("Received chunk: {:#?}", chunk),
                Err(err) => eprintln!("Stream error: {}", err),
            }
        }
    }

    #[tokio::test]
    async fn test_fim_completion_stream_blocking() {
        let fim_request = get_fim_builder()
            .prompt("def fib(a):")
            .suffix("    return fib(a-1) + fib(a-2)")
            .stream(true)
            .build()
            .unwrap();
        let stream = fim_request.stream_blocking().unwrap();
        for item in stream {
            match item {
                Ok(chunk) => println!("Received chunk: {:#?}", chunk),
                Err(err) => eprintln!("Stream error: {}", err),
            }
        }
    }
}
