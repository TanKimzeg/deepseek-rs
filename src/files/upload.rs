use super::types::{FileExpiration, FileObject};
use crate::{DeepSeekClient, DeepSeekError, api_post_multipart};
use reqwest::multipart;

/// Upload an image file that can later be referenced by its `file_id` in chat completion requests.
///
/// Supported formats: JPEG, PNG, GIF, and WebP. Maximum file size: 64 MiB.
///
/// See the [Files API guide](https://api-docs.deepseek.com/guides/files_api) for details.
///
/// # Arguments
/// * `client` - The DeepSeek API client.
/// * `filename` - The filename to use for the uploaded file.
/// * `data` - The raw bytes of the file to upload.
/// * `expiration` - Optional expiration settings. If `None`, the file is stored permanently.
///
/// # Errors
/// Returns `DeepSeekError` if the upload fails.
pub async fn upload_file(
    client: &DeepSeekClient,
    filename: &str,
    data: Vec<u8>,
    expiration: Option<FileExpiration>,
) -> Result<FileObject, DeepSeekError> {
    let part = multipart::Part::bytes(data)
        .file_name(filename.to_string())
        .mime_str(mime_from_filename(filename))?;

    let mut form = multipart::Form::new()
        .text("purpose", "user_data".to_string())
        .part("file", part);

    if let Some(exp) = expiration {
        form = form
            .text("expires_after[anchor]", exp.anchor)
            .text("expires_after[seconds]", exp.seconds.to_string());
    }

    api_post_multipart("/files", form, client.clone()).await
}

/// Upload an image file from a path.
///
/// This is a convenience wrapper around [`upload_file`] that reads the file from disk.
///
/// # Arguments
/// * `client` - The DeepSeek API client.
/// * `path` - Path to the image file on disk.
/// * `expiration` - Optional expiration settings.
///
/// # Errors
/// Returns `DeepSeekError` if the file cannot be read or the upload fails.
pub async fn upload_file_from_path(
    client: &DeepSeekClient,
    path: impl AsRef<std::path::Path>,
    expiration: Option<FileExpiration>,
) -> Result<FileObject, DeepSeekError> {
    let path = path.as_ref();
    let filename = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("image.jpg")
        .to_string();
    let data = tokio::fs::read(path).await?;
    upload_file(client, &filename, data, expiration).await
}

/// Infer a MIME type from a filename extension.
///
/// Falls back to `application/octet-stream` for unknown or missing extensions.
fn mime_from_filename(filename: &str) -> &'static str {
    let ext = filename
        .rsplit_once('.')
        .map(|(_, ext)| ext)
        .unwrap_or_default()
        .to_ascii_lowercase();
    match ext.as_str() {
        "jpg" | "jpeg" => "image/jpeg",
        "png" => "image/png",
        "gif" => "image/gif",
        "webp" => "image/webp",
        _ => "application/octet-stream",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::request::{ChatMessage, ChatRequestBuilder, UserContent, UserContentPart};
    use crate::files::{FileListParams, delete_file, list_files, retrieve_file};
    use crate::{DEFAULT_BASE_URL, DeepSeekClient, DeepSeekRequest};

    #[test]
    fn mime_inference_from_extension() {
        assert_eq!(mime_from_filename("a.png"), "image/png");
        assert_eq!(mime_from_filename("b.JPG"), "image/jpeg");
        assert_eq!(mime_from_filename("c.jpeg"), "image/jpeg");
        assert_eq!(mime_from_filename("d.gif"), "image/gif");
        assert_eq!(mime_from_filename("e.webp"), "image/webp");
        assert_eq!(mime_from_filename("f.txt"), "application/octet-stream");
        assert_eq!(mime_from_filename("noext"), "application/octet-stream");
    }

    fn get_client() -> DeepSeekClient {
        DeepSeekClient::new(
            std::env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY is not set"),
            DEFAULT_BASE_URL.clone(),
        )
    }

    fn crc32(data: &[u8]) -> u32 {
        let mut table = [0u32; 256];
        for i in 0..256u32 {
            let mut c = i;
            for _ in 0..8 {
                c = if c & 1 != 0 {
                    0xEDB8_8320 ^ (c >> 1)
                } else {
                    c >> 1
                };
            }
            table[i as usize] = c;
        }
        let mut crc = 0xFFFF_FFFFu32;
        for &b in data {
            crc = table[((crc ^ b as u32) & 0xFF) as usize] ^ (crc >> 8);
        }
        crc ^ 0xFFFF_FFFF
    }

    fn png_chunk(chunk_type: &[u8; 4], data: &[u8]) -> Vec<u8> {
        let mut crc_input = Vec::with_capacity(4 + data.len());
        crc_input.extend_from_slice(chunk_type);
        crc_input.extend_from_slice(data);

        let mut chunk = Vec::with_capacity(12 + data.len());
        chunk.extend_from_slice(&(data.len() as u32).to_be_bytes());
        chunk.extend_from_slice(chunk_type);
        chunk.extend_from_slice(data);
        chunk.extend_from_slice(&crc32(&crc_input).to_be_bytes());
        chunk
    }

    /// Generate a valid 1x1 transparent RGBA PNG without external dependencies.
    fn minimal_png() -> Vec<u8> {
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&1u32.to_be_bytes()); // width
        ihdr.extend_from_slice(&1u32.to_be_bytes()); // height
        ihdr.push(8); // bit depth
        ihdr.push(6); // color type: RGBA
        ihdr.push(0); // compression
        ihdr.push(0); // filter
        ihdr.push(0); // interlace

        // zlib stream wrapping a stored (uncompressed) deflate block that holds
        // one scanline: filter byte 0 followed by a single RGBA pixel.
        let raw_scanline = [0u8, 0, 0, 0, 0];
        let len = raw_scanline.len() as u16;
        let mut idat = vec![0x78, 0x01, 0x01];
        idat.extend_from_slice(&len.to_le_bytes());
        idat.extend_from_slice(&(!len).to_le_bytes());
        idat.extend_from_slice(&raw_scanline);
        idat.extend_from_slice(&0x0005_0001u32.to_be_bytes()); // adler32 of zeros

        let mut png = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        png.extend_from_slice(&png_chunk(b"IHDR", &ihdr));
        png.extend_from_slice(&png_chunk(b"IDAT", &idat));
        png.extend_from_slice(&png_chunk(b"IEND", &[]));
        png
    }

    #[tokio::test]
    async fn files_upload_retrieve_use_and_delete_lifecycle() {
        let client = get_client();
        let image = minimal_png();

        let uploaded = upload_file(&client, "opencode-test.png", image.clone(), None)
            .await
            .expect("upload should succeed");
        assert!(uploaded.id.starts_with("file-api-"));
        assert_eq!(uploaded.purpose, "user_data");
        assert_eq!(uploaded.bytes, image.len() as u64);
        assert_eq!(uploaded.filename, "opencode-test.png");
        assert!(uploaded.expires_at.is_none());

        let retrieved = retrieve_file(&client, &uploaded.id)
            .await
            .expect("retrieve should succeed");
        assert_eq!(retrieved.id, uploaded.id);

        let listed = list_files(
            &client,
            Some(FileListParams {
                purpose: Some("user_data".to_string()),
                ..Default::default()
            }),
        )
        .await
        .expect("list should succeed");
        assert!(listed.data.iter().any(|f| f.id == uploaded.id));

        // Reference the uploaded file in a vision request before deleting it.
        let req = ChatRequestBuilder::default()
            .client(client.clone())
            .model("deepseek-v4-flash-vision-exp")
            .message(ChatMessage::User {
                content: UserContent::Parts(vec![
                    UserContentPart::text("Reply with exactly one word: OK"),
                    UserContentPart::file_id(uploaded.id.clone()),
                ]),
                name: None,
            })
            .max_tokens(16_u32)
            .build()
            .expect("request should build");
        let resp = req.send().await.expect("vision request should succeed");
        assert!(!resp.choices.is_empty());

        let deleted = delete_file(&client, &uploaded.id)
            .await
            .expect("delete should succeed");
        assert!(deleted.deleted);

        let gone = retrieve_file(&client, &uploaded.id).await;
        assert!(gone.is_err(), "deleted file should not be retrievable");
    }

    #[tokio::test]
    async fn files_upload_with_expiration_sets_expires_at() {
        let client = get_client();

        let uploaded = upload_file(
            &client,
            "opencode-expiring-test.png",
            minimal_png(),
            Some(FileExpiration::created_at(3600)),
        )
        .await
        .expect("upload should succeed");

        assert!(uploaded.expires_at.is_some(), "expires_at should be set");

        delete_file(&client, &uploaded.id)
            .await
            .expect("cleanup delete should succeed");
    }
}
