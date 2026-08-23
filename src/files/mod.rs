//! Files API (`/files`) for uploading and managing images.
//!
//! This module provides functions to upload, list, retrieve, and delete image files
//! that can be referenced by `file_id` in chat completion requests with the
//! `deepseek-v4-flash-vision-exp` model.
//!
//! Supported formats: JPEG, PNG, GIF, and WebP.
//!
//! See the [Files API guide](https://api-docs.deepseek.com/guides/files_api) for details.
//!
//! # Example
//!
//! ```ignore
//! use deepseek_sdk::{DeepSeekClient, DEFAULT_BASE_URL};
//! use deepseek_sdk::files::{upload_file_from_path, list_files};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let client = DeepSeekClient::new("sk-...", DEFAULT_BASE_URL.clone());
//!
//! // Upload an image
//! let file = upload_file_from_path(&client, "image.jpg", None).await?;
//! println!("Uploaded: {}", file.id);
//!
//! // List all files
//! let files = list_files(&client, None).await?;
//! for f in &files.data {
//!     println!("{} - {} bytes", f.filename, f.bytes);
//! }
//! # Ok(()) }
//! ```

pub mod delete;
pub mod list;
pub mod retrieve;
pub mod types;
pub mod upload;

pub use delete::delete_file;
pub use list::list_files;
pub use retrieve::retrieve_file;
pub use types::*;
pub use upload::{upload_file, upload_file_from_path};
