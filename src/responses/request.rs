use super::*;
use derive_builder::Builder;

/// Responses request body for the OpenAI Responses API format.
#[derive(Clone, Debug, PartialEq, Serialize, Builder)]
#[builder(
    pattern = "owned",
    setter(into, strip_option),
    build_fn(validate = "Self::validate"),
    name = "ResponsesRequestBuilder"
)]
pub struct ResponsesRequest {
    #[serde(skip_serializing)]
    pub client: DeepSeekClient,

    /// ID of the model to use. The Responses API currently only supports `deepseek-v4-flash`.
    pub model: String,

    /// The input to the model. Either a plain string (treated as a single `user` message),
    /// or a list of input items.
    ///
    /// At least one of `input` and `instructions` is required.
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<Input>,

    /// A system-level instruction, inserted as the first system message of the model's context.
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,

    /// Configuration of the thinking mode.
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Reasoning>,

    /// An upper bound for the number of tokens that can be generated in the response,
    /// including both the visible output tokens and the reasoning tokens.
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,

    /// If set to `true`, the response is streamed as semantic server-sent events.
    /// The final event is `response.completed` / `response.incomplete` / `response.failed`
    /// (there is no `data: [DONE]` message).
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    /// Possible values: `<= 2`
    ///
    /// Default value: `1`
    ///
    /// What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output
    /// more random, while lower values like 0.2 will make it more focused and deterministic.
    /// Has no effect in thinking mode.
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    /// Possible values: `<= 1`
    ///
    /// Default value: `1`
    ///
    /// An alternative to sampling with temperature, called nucleus sampling.
    /// Has no effect in thinking mode.
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,

    /// Configuration of the text output.
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<Text>,

    /// A list of tools the model may call. Function names must be non-empty, at most 128 characters,
    /// match `^[a-zA-Z0-9_-]+$`, and be unique across all tools.
    /// Besides `function`, the built-in `web_search` tool is supported and executed on the server side.
    #[builder(default, setter(each(name = "tool", into)))]
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Tool>,

    /// Controls which (if any) tool is called by the model.
    ///
    /// `none` means the model will not call any tool and instead generates a message.
    /// `auto` (default) means the model can pick between generating a message or calling one or more tools.
    /// `required` means the model must call one or more tools.
    ///
    /// Specifying a particular tool via `{"type": "function", "name": "my_function"}` forces the model
    /// to call that tool.
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    /// Possible values: `<= 20`
    ///
    /// An integer between 0 and 20 specifying the number of most likely tokens to return at each token
    /// position, each with an associated log probability.
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,

    /// A custom end-user identifier, with allowed character set `[a-zA-Z0-9\-_]` and a maximum length
    /// of 512. Do not include user privacy information.
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

/// The input to the model. Either a plain string (treated as a single `user` message),
/// or a list of input items.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub enum Input {
    /// A plain string, treated as a single `user` message.
    TextInput(String),
    /// A list of input items.
    InputItemList(Vec<InputItem>),
}

impl From<String> for Input {
    fn from(value: String) -> Self {
        Input::TextInput(value)
    }
}

impl From<&str> for Input {
    fn from(value: &str) -> Self {
        Input::TextInput(value.to_string())
    }
}

impl From<Vec<InputItem>> for Input {
    fn from(value: Vec<InputItem>) -> Self {
        Input::InputItemList(value)
    }
}

/// A single input item of a Responses request.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct InputItem {
    /// The type of the input item. For `message` items, this field can be omitted if `role` is present.
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub typ: Option<InputItemType>,

    /// For `message` items. The role of the message author. `developer` is treated as `system`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<InputRole>,

    /// For `message` items, the message content, either a plain string or a list of content parts.
    /// For `reasoning` items, a list of `reasoning_text` content parts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<InputContent>,

    /// For `function_call` / `function_call_output` items. The ID pairing a function call with its output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,

    /// For `function_call` items. The name of the function to call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// For `function_call` items. The arguments to call the function with, in JSON format.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,

    /// For `function_call_output` items. The output of the function call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
}

impl InputItem {
    /// Build a `user` message input item from plain text.
    pub fn user(content: impl Into<String>) -> Self {
        InputItem {
            typ: None,
            role: Some(InputRole::User),
            content: Some(InputContent::Text(content.into())),
            call_id: None,
            name: None,
            arguments: None,
            output: None,
        }
    }

    /// Build an `assistant` message input item from plain text.
    pub fn assistant(content: impl Into<String>) -> Self {
        InputItem {
            typ: None,
            role: Some(InputRole::Assistant),
            content: Some(InputContent::Text(content.into())),
            call_id: None,
            name: None,
            arguments: None,
            output: None,
        }
    }

    /// Build a `function_call` input item.
    pub fn function_call(
        call_id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        InputItem {
            typ: Some(InputItemType::FunctionCall),
            role: None,
            content: None,
            call_id: Some(call_id.into()),
            name: Some(name.into()),
            arguments: Some(arguments.into()),
            output: None,
        }
    }

    /// Build a `function_call_output` input item.
    pub fn function_call_output(call_id: impl Into<String>, output: impl Into<String>) -> Self {
        InputItem {
            typ: Some(InputItemType::FunctionCallOutput),
            role: None,
            content: None,
            call_id: Some(call_id.into()),
            name: None,
            arguments: None,
            output: Some(output.into()),
        }
    }
}

/// The type of an input item.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum InputItemType {
    Message,
    FunctionCall,
    FunctionCallOutput,
    Reasoning,
    WebSearchCall,
    /// An unrecognized input item type is ignored by the server.
    #[serde(other)]
    Unknown,
}

/// The role of a message input item.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum InputRole {
    User,
    Assistant,
    System,
    /// Treated as `system` by the server.
    Developer,
    #[serde(other)]
    Unknown,
}

/// Message content, either a plain string or a list of content parts.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub enum InputContent {
    /// A plain string message body.
    Text(String),
    /// A list of content parts.
    Parts(Vec<InputContentPart>),
}

impl From<String> for InputContent {
    fn from(value: String) -> Self {
        InputContent::Text(value)
    }
}

impl From<&str> for InputContent {
    fn from(value: &str) -> Self {
        InputContent::Text(value.to_string())
    }
}

/// A content part of a message input item.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputContentPart {
    InputText { text: String },
    OutputText { text: String },
}

/// Configuration of the thinking mode.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Reasoning {
    /// Controls the thinking mode toggle and the thinking effort.
    pub effort: ReasoningEffort,
}

impl Reasoning {
    pub fn new(effort: ReasoningEffort) -> Self {
        Reasoning { effort }
    }
}

/// Thinking effort levels.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    /// Disables thinking mode.
    None,
    Minimal,
    Low,
    Medium,
    High,
    /// Mapped to effort `high`.
    #[serde(rename = "xhigh")]
    XHigh,
    /// Enables thinking mode with effort `max`.
    #[serde(rename = "max")]
    Max,
}

/// Configuration of the text output.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Text {
    /// The output format.
    pub format: TextFormat,
}

impl Text {
    pub fn new(format: TextFormat) -> Self {
        Text { format }
    }
}

/// The output format of the response.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TextFormat {
    /// Plain text (default).
    Text,
    /// JSON mode.
    JsonObject,
    /// Structured output conforming to the given JSON Schema.
    JsonSchema {
        /// The name of the schema. Required when `type` is `json_schema`.
        name: String,
        /// The JSON Schema that the output must conform to.
        schema: serde_json::Value,
    },
}

impl TextFormat {
    pub fn text() -> Self {
        TextFormat::Text
    }

    pub fn json_object() -> Self {
        TextFormat::JsonObject
    }

    pub fn json_schema(name: impl Into<String>, schema: serde_json::Value) -> Self {
        TextFormat::JsonSchema {
            name: name.into(),
            schema,
        }
    }
}

/// A tool the model may call.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Tool {
    /// The type of the tool.
    #[serde(rename = "type")]
    pub typ: ToolType,

    /// For `function` tools. The name of the function. Must be non-empty, at most 128 characters,
    /// match `^[a-zA-Z0-9_-]+$`, and be unique across all tools.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// For `function` tools. A description of what the function does.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// The parameters the function accepts, described as a JSON Schema object.
    ///
    /// Omitting `parameters` defines a function with an empty parameter list.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

impl Tool {
    /// Build a `function` tool.
    pub fn function(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Option<serde_json::Value>,
    ) -> Self {
        Tool {
            typ: ToolType::Function,
            name: Some(name.into()),
            description: Some(description.into()),
            parameters,
        }
    }

    /// Build a server-side `web_search` tool.
    pub fn web_search() -> Self {
        Tool {
            typ: ToolType::WebSearch,
            name: None,
            description: None,
            parameters: None,
        }
    }
}

/// Tool type.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolType {
    Function,
    WebSearch,
    #[serde(rename = "web_search_2025_08_26")]
    WebSearch2025_08_26,
}

/// Tool choice configuration.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ToolChoice {
    /// Possible values: [`none`, `auto`, `required`]
    Mode(ToolChoiceMode),
    /// A specific tool, e.g. `{"type": "function", "name": "my_function"}`.
    Named(NamedToolChoice),
}

impl ToolChoice {
    pub fn none() -> Self {
        ToolChoice::Mode(ToolChoiceMode::None)
    }

    pub fn auto() -> Self {
        ToolChoice::Mode(ToolChoiceMode::Auto)
    }

    pub fn required() -> Self {
        ToolChoice::Mode(ToolChoiceMode::Required)
    }

    pub fn named(name: impl Into<String>) -> Self {
        ToolChoice::Named(NamedToolChoice {
            typ: ToolType::Function,
            name: Some(name.into()),
        })
    }

    pub fn web_search() -> Self {
        ToolChoice::Named(NamedToolChoice {
            typ: ToolType::WebSearch,
            name: None,
        })
    }
}

/// Tool choice modes.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoiceMode {
    None,
    Auto,
    Required,
}

/// A named tool choice.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct NamedToolChoice {
    /// Possible values: [`function`, `web_search`, `web_search_2025_08_26`]
    #[serde(rename = "type")]
    pub typ: ToolType,
    /// The name of the function to call. Required when `type` is `function`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl ResponsesRequestBuilder {
    fn validate(&self) -> Result<(), String> {
        if self.input.as_ref().and_then(|o| o.as_ref()).is_none()
            && self
                .instructions
                .as_ref()
                .and_then(|o| o.as_ref())
                .is_none()
        {
            return Err("at least one of `input` and `instructions` is required".to_string());
        }

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

        if let Some(top_logprobs) = self.top_logprobs.flatten()
            && top_logprobs > 20
        {
            return Err("top_logprobs must be <= 20".to_string());
        }

        if let Some(user) = self.user.as_ref().and_then(|u| u.as_ref()) {
            if user.len() > 512 {
                return Err("user must be at most 512 characters".to_string());
            }
            if !user
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
            {
                return Err("user must only contain [a-zA-Z0-9\\-_]".to_string());
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn client() -> DeepSeekClient {
        DeepSeekClient::new(
            std::env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY is not set"),
            crate::DEFAULT_BASE_URL.clone(),
        )
    }

    #[test]
    fn input_serializes_as_string_or_list() {
        let text = Input::TextInput("Hi".to_string());
        assert_eq!(serde_json::to_value(text).unwrap(), json!("Hi"));

        let items = Input::InputItemList(vec![InputItem::user("Hi")]);
        assert_eq!(
            serde_json::to_value(items).unwrap(),
            json!([{"role": "user", "content": "Hi"}])
        );
    }

    #[test]
    fn reasoning_effort_serializes_effort_values() {
        assert_eq!(
            serde_json::to_value(ReasoningEffort::None).unwrap(),
            json!("none")
        );
        assert_eq!(
            serde_json::to_value(ReasoningEffort::XHigh).unwrap(),
            json!("xhigh")
        );
        assert_eq!(
            serde_json::to_value(ReasoningEffort::Max).unwrap(),
            json!("max")
        );
    }

    #[test]
    fn tool_type_serializes_web_search_names() {
        assert_eq!(
            serde_json::to_value(ToolType::WebSearch).unwrap(),
            json!("web_search")
        );
        assert_eq!(
            serde_json::to_value(ToolType::WebSearch2025_08_26).unwrap(),
            json!("web_search_2025_08_26")
        );
    }

    #[test]
    fn text_format_serializes_json_schema() {
        let format = TextFormat::json_schema(
            "math_response",
            json!({"type": "object", "properties": {"answer": {"type": "number"}}}),
        );
        assert_eq!(
            serde_json::to_value(format).unwrap(),
            json!({
                "type": "json_schema",
                "name": "math_response",
                "schema": {"type": "object", "properties": {"answer": {"type": "number"}}}
            })
        );
    }

    #[test]
    fn tool_choice_serializes_mode_and_named() {
        assert_eq!(
            serde_json::to_value(ToolChoice::auto()).unwrap(),
            json!("auto")
        );
        assert_eq!(
            serde_json::to_value(ToolChoice::named("get_weather")).unwrap(),
            json!({"type": "function", "name": "get_weather"})
        );
    }

    #[test]
    fn request_serializes_full_payload() {
        let req = ResponsesRequestBuilder::default()
            .client(client())
            .model("deepseek-v4-flash")
            .input("Hi")
            .instructions("You are a helpful assistant.")
            .reasoning(Reasoning::new(ReasoningEffort::Low))
            .max_output_tokens(256_u32)
            .temperature(0.7_f64)
            .tool(Tool::function("get_weather", "Get the weather", None))
            .build()
            .unwrap();

        let value = serde_json::to_value(&req).unwrap();
        assert_eq!(value.get("model"), Some(&json!("deepseek-v4-flash")));
        assert_eq!(value.get("input"), Some(&json!("Hi")));
        assert_eq!(value.get("reasoning"), Some(&json!({"effort": "low"})));
        assert_eq!(value.get("client"), None);
    }

    #[test]
    fn builder_validation_rejects_invalid_values() {
        let base = || {
            ResponsesRequestBuilder::default()
                .client(client())
                .model("deepseek-v4-flash")
        };

        assert!(base().build().is_err(), "no input nor instructions");

        assert!(
            base().input("Hi").temperature(2.5_f64).build().is_err(),
            "temperature out of range"
        );
        assert!(
            base().input("Hi").top_p(1.5_f64).build().is_err(),
            "top_p out of range"
        );
        assert!(
            base().input("Hi").top_logprobs(21_u32).build().is_err(),
            "top_logprobs out of range"
        );
        assert!(
            base().input("Hi").user("not allowed!").build().is_err(),
            "user charset"
        );
        assert!(
            base().instructions("sys").build().is_ok(),
            "instructions alone is valid"
        );
    }

    #[test]
    fn deserialize_unknown_enum_variants() {
        let typ: InputItemType = serde_json::from_value(json!("file_search")).unwrap();
        assert_eq!(typ, InputItemType::Unknown);

        let role: InputRole = serde_json::from_value(json!("bot")).unwrap();
        assert_eq!(role, InputRole::Unknown);
    }
}
