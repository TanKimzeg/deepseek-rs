use serde::{Deserialize, Serialize};

/// A file object returned by the Files API.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct FileObject {
    /// The file identifier, of the form `file-api-...`.
    pub id: String,
    /// The object type, which is always `file`.
    pub object: String,
    /// The size of the file in bytes.
    pub bytes: u64,
    /// The Unix timestamp (in seconds) of when the file was created.
    pub created_at: u64,
    /// The name of the file.
    pub filename: String,
    /// The intended purpose of the file.
    pub purpose: String,
    /// The Unix timestamp (in seconds) of when the file expires.
    /// Only present when an expiration was set at upload time.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<u64>,
}

/// Response from deleting a file.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct FileDeleteResponse {
    /// The ID of the deleted file.
    pub id: String,
    /// The object type, which is always `file`.
    pub object: String,
    /// Whether the file was successfully deleted.
    pub deleted: bool,
}

/// Response from listing files.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct FileListResponse {
    /// The object type, which is always `list`.
    pub object: String,
    /// The list of file objects.
    pub data: Vec<FileObject>,
    /// The ID of the first file in the list.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_id: Option<String>,
    /// The ID of the last file in the list.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_id: Option<String>,
    /// Whether there are more files beyond this page.
    pub has_more: bool,
}

/// Query parameters for listing files.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct FileListParams {
    /// A `file_id` cursor for pagination. Returns files listed after this one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub after: Option<String>,
    /// The number of files to return (1-1000, default 1000).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
    /// Sort order by creation time: `asc` (default) or `desc`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub order: Option<String>,
    /// Only return files with the given purpose.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub purpose: Option<String>,
}

/// Expiration settings for file upload.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct FileExpiration {
    /// The anchor for the expiration. Must be `created_at`.
    pub anchor: String,
    /// The lifetime of the file in seconds (3600-2592000).
    pub seconds: u32,
}

impl FileExpiration {
    /// Create an expiration anchor with a duration in seconds.
    pub fn created_at(seconds: u32) -> Self {
        FileExpiration {
            anchor: "created_at".to_string(),
            seconds,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn file_object_deserializes_from_api_response() {
        let json = json!({
            "id": "file-api-0a1b2c3d4e5f60718293a4b5c6d7e8f9",
            "object": "file",
            "bytes": 102400,
            "created_at": 1700000000,
            "filename": "image.jpg",
            "purpose": "user_data"
        });
        let file: FileObject = serde_json::from_value(json).unwrap();
        assert_eq!(file.id, "file-api-0a1b2c3d4e5f60718293a4b5c6d7e8f9");
        assert_eq!(file.object, "file");
        assert_eq!(file.bytes, 102400);
        assert_eq!(file.filename, "image.jpg");
        assert!(file.expires_at.is_none());
    }

    #[test]
    fn file_object_with_expiration_deserializes() {
        let json = json!({
            "id": "file-api-abc",
            "object": "file",
            "bytes": 1024,
            "created_at": 1700000000,
            "filename": "test.png",
            "purpose": "user_data",
            "expires_at": 1700003600
        });
        let file: FileObject = serde_json::from_value(json).unwrap();
        assert_eq!(file.expires_at, Some(1700003600));
    }

    #[test]
    fn file_delete_response_deserializes() {
        let json = json!({
            "id": "file-api-abc",
            "object": "file",
            "deleted": true
        });
        let resp: FileDeleteResponse = serde_json::from_value(json).unwrap();
        assert_eq!(resp.id, "file-api-abc");
        assert!(resp.deleted);
    }

    #[test]
    fn file_list_response_deserializes() {
        let json = json!({
            "object": "list",
            "data": [
                {
                    "id": "file-api-001",
                    "object": "file",
                    "bytes": 1024,
                    "created_at": 1700000000,
                    "filename": "a.jpg",
                    "purpose": "user_data"
                },
                {
                    "id": "file-api-002",
                    "object": "file",
                    "bytes": 2048,
                    "created_at": 1700000001,
                    "filename": "b.png",
                    "purpose": "user_data"
                }
            ],
            "first_id": "file-api-001",
            "last_id": "file-api-002",
            "has_more": false
        });
        let resp: FileListResponse = serde_json::from_value(json).unwrap();
        assert_eq!(resp.object, "list");
        assert_eq!(resp.data.len(), 2);
        assert_eq!(resp.first_id.as_deref(), Some("file-api-001"));
        assert!(!resp.has_more);
    }

    #[test]
    fn file_expiration_created_at() {
        let exp = FileExpiration::created_at(7200);
        assert_eq!(exp.anchor, "created_at");
        assert_eq!(exp.seconds, 7200);
    }

    #[test]
    fn file_list_params_serializes_only_set_fields() {
        let params = FileListParams {
            limit: Some(50),
            order: Some("desc".to_string()),
            ..Default::default()
        };
        let value = serde_json::to_value(&params).unwrap();
        assert_eq!(value, json!({"limit": 50, "order": "desc"}));
    }

    #[test]
    fn file_expiration_serializes_anchor_and_seconds() {
        let value = serde_json::to_value(FileExpiration::created_at(3600)).unwrap();
        assert_eq!(value, json!({"anchor": "created_at", "seconds": 3600}));
    }
}
