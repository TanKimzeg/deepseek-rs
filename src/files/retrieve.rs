use super::types::FileObject;
use crate::{DeepSeekClient, DeepSeekError, api_get};

/// Retrieve information about a specific file.
///
/// # Arguments
/// * `client` - The DeepSeek API client.
/// * `file_id` - The ID of the file to retrieve (format: `file-api-...`).
///
/// # Errors
/// Returns `DeepSeekError` if the request fails.
pub async fn retrieve_file(
    client: &DeepSeekClient,
    file_id: &str,
) -> Result<FileObject, DeepSeekError> {
    api_get(&format!("/files/{file_id}"), client.clone()).await
}
