# deepseek-sdk

DeepSeek API client for Rust.

## Features

- Chat completions (`/chat/completions`)
- FIM completions (beta, `/beta/completions`)
- Responses API (OpenAI Responses format, `/responses`)
- List models (`/models`)
- Account balance (`/user/balance`)
- Streaming via async receiver or blocking iterator

## Install

Add to `Cargo.toml`:

```toml
deepseek-sdk = "0.2"
```

By default all API modules are enabled. To compile only what you need:

```toml
deepseek-sdk = { version = "0.2", default-features = false, features = ["chat", "models"] }
```

| Feature | API | Enabled by default |
|---------|-----|--------------------|
| `chat` | Chat completions | Yes |
| `completion` | FIM + Beta chat completions (implies `chat`) | Yes |
| `responses` | Responses API (implies `chat`) | Yes |
| `models` | List models | Yes |
| `balance` | Account balance | Yes |

## API Key

Set your API key before running examples:

```bash
export DEEPSEEK_API_KEY="sk-..."
```

## Quick Start (Chat)

```rust
use deepseek_sdk::chat::request::{ChatMessage, ChatRequestBuilder, Thinking};
use deepseek_sdk::{DeepSeekClient, DeepSeekRequest, DEFAULT_BASE_URL};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
 let req = ChatRequestBuilder::default()
  .client(DeepSeekClient::new("sk-...", DEFAULT_BASE_URL.clone()))
  .model("deepseek-v4-flash")
  .message(ChatMessage::User {
   content: "Hi".to_string(),
   name: None,
  })
  .thinking(Thinking::disabled())
  .max_tokens(1024)
  .build()?;

 let resp = req.send().await?;
 println!("{:#?}", resp);
 Ok(())
}
```

## Async Streaming

```rust
use deepseek_sdk::chat::request::{ChatMessage, ChatRequestBuilder, Thinking};
use deepseek_sdk::{DeepSeekClient, DeepSeekRequest, DEFAULT_BASE_URL};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
 let req = ChatRequestBuilder::default()
  .client(DeepSeekClient::new("sk-...", DEFAULT_BASE_URL.clone()))
  .model("deepseek-v4-flash")
  .message(ChatMessage::User { content: "Hi".into(), name: None })
  .thinking(Thinking::disabled())
  .build()?;

 let mut rx = req.stream().await?;
 while let Some(item) = rx.recv().await {
  let chunk = item?;
  for choice in chunk.choices {
   if let Some(delta) = choice.delta.content {
    print!("{delta}");
   }
  }
 }
 Ok(())
}
```

## Blocking Streaming

```rust
use deepseek_sdk::chat::request::{ChatMessage, ChatRequestBuilder, Thinking};
use deepseek_sdk::{DeepSeekClient, DeepSeekRequest, DEFAULT_BASE_URL};

fn main() -> Result<(), Box<dyn std::error::Error>> {
 let req = ChatRequestBuilder::default()
  .client(DeepSeekClient::new("sk-...", DEFAULT_BASE_URL.clone()))
  .model("deepseek-v4-flash")
  .message(ChatMessage::User { content: "Hi".into(), name: None })
  .thinking(Thinking::disabled())
  .build()?;

 let stream = req.stream_blocking()?;
 for item in stream {
  let chunk = item?;
  for choice in chunk.choices {
   if let Some(delta) = choice.delta.content {
    print!("{delta}");
   }
  }
 }
 Ok(())
}
```

## FIM Completion (Beta)

FIM uses the beta base URL.

```rust
use deepseek_sdk::completion::fim::FIMCompletionRequestBuilder;
use deepseek_sdk::{DeepSeekClient, DeepSeekRequest, DEFAULT_BETA_BASE_URL};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
 let req = FIMCompletionRequestBuilder::default()
  .client(DeepSeekClient::new("sk-...", DEFAULT_BETA_BASE_URL.clone()))
  .model("deepseek-v4-pro")
  .prompt("def fib(n):")
  .suffix("    return fib(n-1) + fib(n-2)")
  .max_tokens(128)
  .build()?;

 let resp = req.send().await?;
 println!("{:#?}", resp);
 Ok(())
}
```

## Responses API (OpenAI Responses format)

The Responses API currently supports the `deepseek-v4-flash` model and uses the same base URL.

```rust
use deepseek_sdk::responses::request::{ReasoningEffort, Reasoning, ResponsesRequestBuilder};
use deepseek_sdk::{DeepSeekClient, DeepSeekRequest, DEFAULT_BASE_URL};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
 let req = ResponsesRequestBuilder::default()
  .client(DeepSeekClient::new("sk-...", DEFAULT_BASE_URL.clone()))
  .model("deepseek-v4-flash")
  .instructions("You are a helpful assistant.")
  .input("Hi, how are you?")
  .reasoning(Reasoning::new(ReasoningEffort::Low))
  .build()?;

 let resp = req.send().await?;
 println!("{}", resp.output_text());
 Ok(())
}
```

### Streaming Responses

DeepSeek streams the Responses API as semantic server-sent events. The final event is
`response.completed` / `response.incomplete` / `response.failed` — there is no `data: [DONE]`.

```rust
use deepseek_sdk::responses::request::{ResponsesRequestBuilder, ResponsesStreamEvent};
use deepseek_sdk::{DeepSeekClient, DeepSeekRequest, DEFAULT_BASE_URL};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
 let req = ResponsesRequestBuilder::default()
  .client(DeepSeekClient::new("sk-...", DEFAULT_BASE_URL.clone()))
  .model("deepseek-v4-flash")
  .input("Hi, how are you?")
  .build()?;

 let mut rx = req.stream().await?;
 while let Some(item) = rx.recv().await {
  let event = item?;
  if let Some(delta) = event.delta() {
   print!("{delta}");
  }
  if event.is_terminal() {
   let response = event.response().expect("terminal event carries the response");
   println!("\nstatus: {:?}, usage: {:?}", response.status, response.usage.total_tokens);
  }
 }
 Ok(())
}
```

## List Models

```rust
use deepseek_sdk::models::Models;
use deepseek_sdk::{DeepSeekClient, DEFAULT_BASE_URL};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
 let client = DeepSeekClient::new("sk-...", DEFAULT_BASE_URL.clone());
 let models = Models::list(client).await?;
 println!("{:#?}", models);
 Ok(())
}
```

## Balance

```rust
use deepseek_sdk::balance::Balance;
use deepseek_sdk::{DeepSeekClient, DEFAULT_BASE_URL};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
 let client = DeepSeekClient::new("sk-...", DEFAULT_BASE_URL.clone());
 let balance = Balance::get(client).await?;
 println!("{:#?}", balance);
 Ok(())
}
```

## Error Handling

All requests return `DeepSeekError` on failure, covering:

- API error payloads (`Api`)
- HTTP errors (`Http`)
- Decode errors (`Decode`)
- Transport failures (`Transport`)

All public error enums are marked `#[non_exhaustive]` — new variants may be added without a breaking semver change.

## License

MIT
