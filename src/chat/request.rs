use super::*;
use derive_builder::Builder;

pub(crate) fn is_none_or_empty_stop(opt: &Option<Stop>) -> bool {
    opt.as_ref().map(|stop| stop.is_empty()).unwrap_or(true)
}

/// Chat completion request body.
#[derive(Clone, Debug, PartialEq, Serialize, Builder)]
#[builder(
    pattern = "owned",
    setter(into, strip_option),
    build_fn(validate = "Self::validate"),
    name = "ChatRequestBuilder"
)]
pub struct ChatRequest {
    #[serde(skip_serializing)]
    pub client: DeepSeekClient,

    /// A list of messages comprising the conversation so far.
    #[builder(setter(each(name = "message", into)))]
    pub messages: Vec<ChatMessage>,

    /// Possible values: \`deepseek-v4-flash\`, \`deepseek-v4-pro\`
    ///
    /// ID of the model to use.
    pub model: String,

    /// Controls the switch between thinking and non-thinking mode.
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<Thinking>,

    /// Possible values: [`high`, `max`]
    ///
    /// Controls the reasoning effort of the model.
    /// The default effort is `high` for regular requests;
    /// for some complex agent requests (such as Claude Code, OpenCode),
    /// effort is automatically set to `max`.
    /// For compatibility, `low` and `medium` are mapped to `high`,
    /// and `xhigh` is mapped to `max`.
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffort>,

    /// The maximum number of tokens that can be generated in the chat completion.
    ///
    /// The total length of input tokens and generated tokens is limited by the model's context length.
    ///
    /// For the value range and default value, please refer to the [documentation](https://api-docs.deepseek.com/quick_start/pricing).
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// An object specifying the format that the model must output.
    /// Setting to { "type": "json_object" } enables JSON Output,
    /// which guarantees the message the model generates is valid JSON.
    ///
    /// **Important**: When using JSON Output, you must also instruct the model to produce JSON yourself via a system or user message.
    /// Without this, the model may generate an unending stream of whitespace until the generation reaches the token limit, resulting in a long-running and seemingly "stuck" request. Also note that the message content may be partially cut off if finish_reason="length", which indicates the generation exceeded max_tokens or the conversation exceeded the max context length.
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,

    /// Up to 16 sequences where the API will stop generating further tokens.
    #[builder(default)]
    #[serde(skip_serializing_if = "is_none_or_empty_stop")]
    pub stop: Option<Stop>,

    /// If set, partial message deltas will be sent.
    /// Tokens will be sent as data-only server-sent events (SSE) as they become available,
    /// with the stream terminated by a \`data: \[DONE\]\` message.
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    /// Options for streaming response. Only set this when you set `stream: true`.
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,

    /// Possible values: `<= 2`
    ///
    /// Default value: `1`
    ///
    /// What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
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
    ///
    /// We generally recommend altering this or `temperature` but not both.
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,

    /// A list of tools the model may call. Currently, only functions are supported as a tool.
    /// Use this to provide a list of functions the model may generate JSON inputs for.
    /// A max of 128 functions are supported.
    #[builder(default, setter(each(name = "tool", into)))]
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Tool>,

    /// Controls which (if any) tool is called by the model.
    /// `none` means the model will not call any tool and instead generates a message.
    /// `auto` means the model can pick between generating a message or calling one or more tools.
    /// `required` means the model must call one or more tools.
    /// Specifying a particular tool via `{"type": "function", "function": {"name": "my_function"}}` forces the model to call that tool.
    /// `none` is the default when no tools are present. `auto` is the default if tools are present.
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    /// Whether to return log probabilities of the output tokens or not.
    /// If true, returns the log probabilities of each output token returned in the `content` of `message`.
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,

    /// Possible values: `<= 20`
    ///
    /// An integer between 0 and 20 specifying the number of most likely tokens to return at each token position,
    /// each with an associated log probability. `logprobs` must be set to `true` if this parameter is used.
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,

    /// A custom `user_id`. Allowed character set is `[a-zA-Z0-9\-_]`, with a maximum length of 512.
    /// Do not include user privacy information in the `user_id`.

    /// `user_id` can be used to distinguish user identities on your side to help us with content safety review.
    /// `user_id` can be used for KVCache isolation for privacy management.
    /// `user_id` can be used for scheduling isolation of users on your business side.
    /// For more details on the `user_id` parameter, please refer to [Rate Limit & Isolation](https://api-docs.deepseek.com/quick_start/rate_limit)
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
}
/// Chat message variants.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum ChatMessage {
    System {
        /// The contents of the system message.
        content: String,
        /// An optional name for the participant. Provides the model information to differentiate between participants of the same role.
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    User {
        /// The contents of the user message.
        content: String,
        /// An optional name for the participant. Provides the model information to differentiate between participants of the same role.
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    Assistant {
        /// The contents of the assistant message.
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<String>,
        /// An optional name for the participant. Provides the model information to differentiate between participants of the same role.
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,

        #[serde(skip_serializing_if = "super::is_none_or_empty_vec")]
        tool_calls: Option<Vec<super::response::ToolCall>>,
    },
    Tool {
        /// The contents of the tool message.
        content: String,
        /// Tool call that this message is responding to.
        tool_call_id: String,
    },
}
/// Reasoning effort hints for the model.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    High,
    Max,
}
/// Response format configuration.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct ResponseFormat {
    /// Default value: `text`
    /// Must be one of `text` or `json_object`.
    #[serde(rename = "type")]
    pub(crate) typ: ResponseFormatType,
}
/// Supported response format types.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ResponseFormatType {
    Text,
    JsonObject,
}

impl ResponseFormat {
    pub fn text() -> Self {
        ResponseFormat {
            typ: ResponseFormatType::Text,
        }
    }

    pub fn json_object() -> Self {
        ResponseFormat {
            typ: ResponseFormatType::JsonObject,
        }
    }
}

/// Stop sequences for generation.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(untagged)]
pub enum Stop {
    One(String),
    Many(Vec<String>),
}

impl Stop {
    fn is_empty(&self) -> bool {
        match self {
            Stop::One(value) => value.is_empty(),
            Stop::Many(values) => values.is_empty(),
        }
    }
}

impl From<String> for Stop {
    fn from(value: String) -> Self {
        Stop::One(value)
    }
}

impl From<&str> for Stop {
    fn from(value: &str) -> Self {
        Stop::One(value.to_string())
    }
}

impl<T> From<Vec<T>> for Stop
where
    T: Into<String>,
{
    fn from(values: Vec<T>) -> Self {
        Stop::Many(values.into_iter().map(Into::into).collect())
    }
}
/// Streaming options for SSE responses.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct StreamOptions {
    /// If set, an additional chunk will be streamed before the \`data: \[DONE\]\` message.
    /// The `usage` field on this chunk shows the token usage statistics for the entire request,
    /// and the `choices` field will always be an empty array.
    /// All other chunks will also include a `usage` field, but with a null value.
    pub include_usage: bool,
}
/// Tool definition used by the model.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Tool {
    /// The type of the tool. Currently, only `function` is supported.
    #[serde(rename = "type")]
    pub typ: ToolType,
    pub function: ToolFunctionDefinition,
}

impl Tool {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Option<serde_json::Value>,
    ) -> Self {
        Tool {
            typ: ToolType::Function,
            function: ToolFunctionDefinition {
                name: name.into(),
                description: description.into(),
                parameters,
            },
        }
    }
}

/// Tool type.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolType {
    Function,
}

/// Tool function definition.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct ToolFunctionDefinition {
    /// A description of what the function does,
    /// used by the model to choose when and how to call the function.
    pub description: String,
    /// The name of the function to be called. Must be a-z, A-Z, 0-9,
    /// or contain underscores and dashes, with a maximum length of 64.
    pub name: String,
    /// The parameters the functions accepts, described as a JSON Schema object.
    /// See the [Tool Calls Guide](https://api-docs.deepseek.com/guides/tool_calls) for examples,
    /// and the [JSON Schema reference](https://json-schema.org/understanding-json-schema/) for documentation about the format.
    ///
    /// Omitting `parameters` defines a function with an empty parameter list.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}
/// Tool choice configuration.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ToolChoice {
    /// Possible values: [`none`, `auto`, `required`]
    Simple(ChatToolChoice),
    /// {"type":"function","function":{...}}
    Named(ChatNamedToolChoice),
}

impl ToolChoice {
    pub fn named(function: serde_json::Value) -> Self {
        ToolChoice::Named(ChatNamedToolChoice {
            typ: ToolType::Function,
            function,
        })
    }

    pub fn none() -> Self {
        ToolChoice::Simple(ChatToolChoice::None)
    }

    pub fn auto() -> Self {
        ToolChoice::Simple(ChatToolChoice::Auto)
    }

    pub fn required() -> Self {
        ToolChoice::Simple(ChatToolChoice::Required)
    }
}

/// Tool choice values.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatToolChoice {
    None,
    Auto,
    Required,
}
/// Named tool choice configuration.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct ChatNamedToolChoice {
    /// Possible values: \`function\`
    ///
    /// The type of the tool. Currently, only `function` is supported.
    #[serde(rename = "type")]
    pub typ: ToolType,

    pub function: serde_json::Value,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct Thinking {
    /// Possible values: [`enabled`, `disabled`]
    ///
    /// Default value: `enabled`
    ///
    /// If set to `enabled`, then use thinking mode. If set to `disabled`, then use non-thinking model.
    #[serde(rename = "type")]
    pub(crate) typ: ThinkingType,
}

impl Thinking {
    pub fn enabled() -> Self {
        Thinking {
            typ: ThinkingType::Enabled,
        }
    }

    pub fn disabled() -> Self {
        Thinking {
            typ: ThinkingType::Disabled,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ThinkingType {
    Enabled,
    Disabled,
}

impl ChatRequestBuilder {
    fn validate(&self) -> Result<(), String> {
        // derive_builder + strip_option makes Option<T> fields become Option<Option<T>> here;
        // flatten() treats "unset" and "explicit None" uniformly for validation.
        if let Some(temperature) = self.temperature.flatten()
            && !(0.0..=2.0).contains(&temperature)
        {
            return Err("temperature must be between 0 and 2".to_string());
        }

        if let Some(top_p) = self.top_p.flatten()
            && !(0.0..=1.0).contains(&top_p)
        {
            return Err("top_p must be between 0 and 1".to_string());
        }

        if let Some(top_logprobs) = self.top_logprobs.flatten() {
            if top_logprobs > 20 {
                return Err("top_logprobs must be <= 20".to_string());
            }
            if self.logprobs.flatten() != Some(true) {
                return Err("top_logprobs requires logprobs=true".to_string());
            }
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

        if let Some(user_id) = self.user_id.as_ref().and_then(|u| u.as_ref()) {
            if user_id.len() > 512 {
                return Err("user_id must be at most 512 characters".to_string());
            }
            if !user_id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
            {
                return Err("user_id must only contain [a-zA-Z0-9\\-_]".to_string());
            }
        }

        Ok(())
    }
}
