use super::types::{FileListParams, FileListResponse};
use crate::{DeepSeekClient, DeepSeekError, api_get};

/// List files that belong to the user, with cursor-based pagination.
///
/// See the [Files API guide](https://api-docs.deepseek.com/guides/files_api) for details.
///
/// # Arguments
/// * `client` - The DeepSeek API client.
/// * `params` - Optional query parameters for filtering and pagination.
///
/// # Errors
/// Returns `DeepSeekError` if the request fails.
pub async fn list_files(
    client: &DeepSeekClient,
    params: Option<FileListParams>,
) -> Result<FileListResponse, DeepSeekError> {
    let route = build_list_route(params.as_ref());
    api_get(&route, client.clone()).await
}

/// Build the `/files` route with optional query parameters.
///
/// Values are percent-encoded so reserved characters (`&`, `=`, spaces, ...)
/// cannot corrupt the query string.
fn build_list_route(params: Option<&FileListParams>) -> String {
    let Some(p) = params else {
        return "/files".to_string();
    };

    let mut query = Vec::new();
    if let Some(after) = &p.after {
        query.push(format!("after={}", percent_encode(after)));
    }
    if let Some(limit) = p.limit {
        query.push(format!("limit={limit}"));
    }
    if let Some(order) = &p.order {
        query.push(format!("order={}", percent_encode(order)));
    }
    if let Some(purpose) = &p.purpose {
        query.push(format!("purpose={}", percent_encode(purpose)));
    }

    if query.is_empty() {
        "/files".to_string()
    } else {
        format!("/files?{}", query.join("&"))
    }
}

/// Percent-encode a query-string value (RFC 3986).
///
/// Unreserved characters (alphanumerics and `-._~`) are kept as-is; every
/// other byte is emitted as `%XX`.
fn percent_encode(value: &str) -> String {
    let mut encoded = String::with_capacity(value.len());
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                encoded.push(byte as char);
            }
            _ => {
                encoded.push('%');
                encoded.push_str(&format!("{byte:02X}"));
            }
        }
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_is_plain_path_without_params() {
        assert_eq!(build_list_route(None), "/files");
        assert_eq!(build_list_route(Some(&FileListParams::default())), "/files");
    }

    #[test]
    fn route_includes_only_set_params() {
        let params = FileListParams {
            limit: Some(50),
            purpose: Some("user_data".to_string()),
            ..Default::default()
        };
        assert_eq!(
            build_list_route(Some(&params)),
            "/files?limit=50&purpose=user_data"
        );
    }

    #[test]
    fn route_includes_all_params_in_field_order() {
        let params = FileListParams {
            after: Some("file-api-001".to_string()),
            limit: Some(10),
            order: Some("desc".to_string()),
            purpose: Some("user_data".to_string()),
        };
        assert_eq!(
            build_list_route(Some(&params)),
            "/files?after=file-api-001&limit=10&order=desc&purpose=user_data"
        );
    }

    #[test]
    fn route_percent_encodes_reserved_characters() {
        let params = FileListParams {
            after: Some("id&x=1 y".to_string()),
            ..Default::default()
        };
        assert_eq!(
            build_list_route(Some(&params)),
            "/files?after=id%26x%3D1%20y"
        );
    }

    #[test]
    fn percent_encode_keeps_unreserved_and_encodes_rest() {
        assert_eq!(percent_encode("file-api_1.0~ok"), "file-api_1.0~ok");
        assert_eq!(percent_encode("a b"), "a%20b");
        assert_eq!(percent_encode("&=?#%"), "%26%3D%3F%23%25");
        assert_eq!(percent_encode(""), "");
        // Non-ASCII bytes are encoded individually.
        assert_eq!(percent_encode("图"), "%E5%9B%BE");
    }
}
