use super::types::FileDeleteResponse;
use crate::{DeepSeekClient, DeepSeekError, api_delete};

/// Delete a file.
///
/// # Arguments
/// * `client` - The DeepSeek API client.
/// * `file_id` - The ID of the file to delete (format: `file-api-...`).
///
/// # Errors
/// Returns `DeepSeekError` if the request fails.
pub async fn delete_file(
    client: &DeepSeekClient,
    file_id: &str,
) -> Result<FileDeleteResponse, DeepSeekError> {
    api_delete(&format!("/files/{file_id}"), client.clone()).await
}
