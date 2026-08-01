//! OpenAI Responses API (`/responses`) request/response models and streaming.
//!
//! This module contains the data structures for the Responses API format and
//! re-exports streaming helpers from the request implementation.
use crate::DeepSeekClient;
use serde::{Deserialize, Serialize};

pub mod request;
pub mod response;
pub mod stream;

pub use stream::{ResponsesStreamBlocking, ResponsesStreamEvent, ResponsesStreamItem};
