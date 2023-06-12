use fs2::FileExt;
use rand::{distributions::Alphanumeric, thread_rng, Rng};
use reqwest::{
    header::{
        HeaderMap, HeaderName, HeaderValue, InvalidHeaderValue, ToStrError, AUTHORIZATION,
        CONTENT_RANGE, RANGE, USER_AGENT,
    },
    Client, Error as ReqwestError,
};
use std::num::ParseIntError;
use std::path::PathBuf;
use std::sync::Arc;
use thiserror::Error;
use tokio::io::{AsyncSeekExt, AsyncWriteExt, SeekFrom};
use tokio::sync::{AcquireError, Semaphore, TryAcquireError};

const VERSION: &str = env!("CARGO_PKG_VERSION");
const NAME: &str = env!("CARGO_PKG_NAME");

#[derive(Debug, Error)]
pub enum ApiError {
    #[error("Header {0} is missing")]
    MissingHeader(HeaderName),
    #[error("Header {0} is invalid")]
    InvalidHeader(HeaderName),
    #[error("Invalid header value {0}")]
    InvalidHeaderValue(#[from] InvalidHeaderValue),
    #[error("header value is not a string")]
    ToStr(#[from] ToStrError),
    #[error("request error: {0}")]
    RequestError(#[from] ReqwestError),
    #[error("Cannot parse int")]
    ParseIntError(#[from] ParseIntError),
    #[error("I/O error {0}")]
    IoError(#[from] std::io::Error),
    #[error("Too many retries: {0}")]
    TooManyRetries(Box<ApiError>),
    #[error("Try acquire: {0}")]
    TryAcquireError(#[from] TryAcquireError),
    #[error("Acquire: {0}")]
    AcquireError(#[from] AcquireError),
}

pub struct Repo {
    repo_id: String,
    revision: String,
}

impl Repo {
    pub fn new(repo_id: String) -> Self {
        Self::with_revision(repo_id, "main".to_string())
    }
    pub fn with_revision(repo_id: String, revision: String) -> Self {
        Self { repo_id, revision }
    }

    pub fn folder_name(&self) -> String {
        self.repo_id.replace('/', "--")
    }
}

pub struct ApiBuilder {
    endpoint: String,
    url_template: String,
    cache_dir: PathBuf,
    token: Option<String>,
    max_files: usize,
    chunk_size: usize,
    parallel_failures: usize,
    max_retries: usize,
}

impl ApiBuilder {
    pub fn new() -> Self {
        let cache_dir = match std::env::var("HF_HOME") {
            Ok(home) => home.into(),
            Err(_) => {
                let mut cache = dirs::home_dir().expect("Cache directory cannot be found");
                cache.push("huggingface");
                cache.push("hub");
                cache
            }
        };
        let mut token_filename = cache_dir.clone();
        token_filename.push(".token");
        let token = match std::fs::read_to_string(token_filename) {
            Ok(token_content) => {
                let token_content = token_content.trim();
                if token_content.len() > 0 {
                    Some(token_content.to_string())
                } else {
                    None
                }
            }
            Err(_) => None,
        };

        Self {
            endpoint: "https://huggingface.co".to_string(),
            url_template: "{endpoint}/{repo_id}/resolve/{revision}/{filename}".to_string(),
            cache_dir,
            token,
            max_files: 100,
            chunk_size: 10_000_000,
            parallel_failures: 0,
            max_retries: 0,
        }
    }

    pub fn with_cache_dir(mut self, cache_dir: &PathBuf) -> Self {
        self.cache_dir = cache_dir.clone();
        self
    }

    fn build_headers(&self) -> Result<HeaderMap, ApiError> {
        let mut headers = HeaderMap::new();
        let user_agent = format!("unkown/None; {NAME}/{VERSION}; rust/unknown");
        headers.insert(USER_AGENT, HeaderValue::from_str(&user_agent)?);
        if let Some(token) = &self.token {
            headers.insert(
                AUTHORIZATION,
                HeaderValue::from_str(&format!("Bearer {token}"))?,
            );
        }
        Ok(headers)
    }

    pub fn build(self) -> Result<Api, ApiError> {
        let headers = self.build_headers()?;
        let client = Client::builder().default_headers(headers).build()?;
        Ok(Api {
            endpoint: self.endpoint,
            url_template: self.url_template,
            cache_dir: self.cache_dir,
            client,
            max_files: self.max_files,
            chunk_size: self.chunk_size,
            parallel_failures: self.parallel_failures,
            max_retries: self.max_retries,
        })
    }
}

#[derive(Debug)]
struct Metadata {
    commit_hash: String,
    etag: String,
    size: usize,
}

pub struct Api {
    endpoint: String,
    url_template: String,
    cache_dir: PathBuf,
    client: Client,
    max_files: usize,
    chunk_size: usize,
    parallel_failures: usize,
    max_retries: usize,
}

fn temp_filename() -> PathBuf {
    let s: String = rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(7)
        .map(char::from)
        .collect();
    let mut path = std::env::temp_dir();
    path.push(s);
    path
}

fn symlink_or_rename(src: &PathBuf, dst: &PathBuf) -> Result<(), std::io::Error> {
    #[cfg(target_os = "windows")]
    std::os::windows::fs::symlink_file(src, dst)?;

    #[cfg(target_os = "unix")]
    std::os::unix::fs::symlink(src, dst)?;

    #[cfg(not(any(target_os = "unix", target_os = "windows")))]
    std::fs::rename(src, dst)?;

    Ok(())
}

fn jitter() -> usize {
    thread_rng().gen_range(0..=500)
}

fn exponential_backoff(base_wait_time: usize, n: usize, max: usize) -> usize {
    (base_wait_time + n.pow(2) + jitter()).min(max)
}

impl Api {
    fn url(&self, repo: &Repo, filename: &str) -> String {
        let endpoint = &self.endpoint;
        let repo_id = &repo.repo_id;
        let revision = &repo.revision;
        self.url_template
            .replace("{endpoint}", endpoint)
            .replace("{repo_id}", repo_id)
            .replace("{revision}", revision)
            .replace("{filename}", filename)
    }

    async fn metadata(&self, url: &str) -> Result<Metadata, ApiError> {
        let response = self
            .client
            .get(url)
            .header(RANGE, "bytes=0-0")
            .send()
            .await?;
        let headers = response.headers();

        let content_range = headers
            .get(CONTENT_RANGE)
            .ok_or(ApiError::MissingHeader(CONTENT_RANGE))?
            .to_str()?;

        let size = content_range
            .split("/")
            .last()
            .ok_or(ApiError::InvalidHeader(CONTENT_RANGE))?
            .parse()?;

        let header_commit = HeaderName::from_static("x-repo-commit");

        let header_linked_etag = HeaderName::from_static("x-linked-etag");
        let header_etag = HeaderName::from_static("etag");

        let etag = match headers.get(&header_linked_etag) {
            Some(etag) => etag,
            None => headers
                .get(&header_etag)
                .ok_or(ApiError::MissingHeader(header_etag))?,
        };
        // Cleaning extra quotes
        let etag = etag.to_str()?.to_string().replace('"', "");

        Ok(Metadata {
            commit_hash: headers
                .get(&header_commit)
                .ok_or(ApiError::MissingHeader(header_commit))?
                .to_str()?
                .to_string(),
            etag,
            size,
        })
    }

    async fn download_tempfile(&self, url: &str, length: usize) -> Result<PathBuf, ApiError> {
        let mut handles = vec![];
        let semaphore = Arc::new(Semaphore::new(self.max_files));
        let parallel_failures_semaphore = Arc::new(Semaphore::new(self.parallel_failures));
        let filename = temp_filename();

        let chunk_size = self.chunk_size;
        for start in (0..length).step_by(chunk_size) {
            let url = url.to_string();
            let filename = filename.clone();
            let client = self.client.clone();

            let stop = std::cmp::min(start + chunk_size - 1, length);
            let permit = semaphore.clone().acquire_owned().await?;
            let parallel_failures = self.parallel_failures;
            let max_retries = self.max_retries;
            let parallel_failures_semaphore = parallel_failures_semaphore.clone();
            handles.push(tokio::spawn(async move {
                let mut chunk = Self::download_chunk(&client, &url, &filename, start, stop).await;
                let mut i = 0;
                if parallel_failures > 0 {
                    while let Err(dlerr) = chunk {
                        let parallel_failure_permit =
                            parallel_failures_semaphore.clone().try_acquire_owned()?;

                        let wait_time = exponential_backoff(300, i, 10_000);
                        tokio::time::sleep(tokio::time::Duration::from_millis(wait_time as u64))
                            .await;

                        chunk = Self::download_chunk(&client, &url, &filename, start, stop).await;
                        i += 1;
                        if i > max_retries {
                            return Err(ApiError::TooManyRetries(dlerr.into()));
                        }
                        drop(parallel_failure_permit);
                    }
                }
                drop(permit);
                chunk
            }));
        }

        // Output the chained result
        let results: Vec<Result<Result<(), ApiError>, tokio::task::JoinError>> =
            futures::future::join_all(handles).await;
        let results: Result<(), ApiError> = results.into_iter().flatten().collect();
        results?;
        Ok(filename)
    }

    async fn download_chunk(
        client: &reqwest::Client,
        url: &str,
        filename: &PathBuf,
        start: usize,
        stop: usize,
    ) -> Result<(), ApiError> {
        // Process each socket concurrently.
        let range = format!("bytes={start}-{stop}");
        let mut file = tokio::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .open(filename)
            .await?;
        file.seek(SeekFrom::Start(start as u64)).await?;
        let response = client
            .get(url)
            .header(RANGE, range)
            .send()
            .await?
            .error_for_status()?;
        let content = response.bytes().await?;
        file.write_all(&content).await?;
        Ok(())
    }

    pub async fn download(&self, repo: &Repo, filename: &str) -> Result<PathBuf, ApiError> {
        let url = self.url(repo, filename);
        let metadata = self.metadata(&url).await?;

        let mut folder = self.cache_dir.clone();
        folder.push(repo.folder_name());

        let mut blob_path = folder.clone();
        blob_path.push("blobs");
        std::fs::create_dir_all(&blob_path).ok();
        blob_path.push(&metadata.etag);

        let file1 = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&blob_path)?;
        file1.lock_exclusive()?;

        let tmp_filename = self.download_tempfile(&url, metadata.size).await?;
        std::fs::copy(tmp_filename, &blob_path)?;

        let mut pointer_path = folder.clone();
        pointer_path.push("snapshots");
        pointer_path.push(&metadata.commit_hash);
        std::fs::create_dir_all(&pointer_path).ok();
        pointer_path.push(filename);

        symlink_or_rename(&blob_path, &pointer_path)?;

        Ok(pointer_path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{distributions::Alphanumeric, Rng}; // 0.8

    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        pub fn new() -> Self {
            let s: String = rand::thread_rng()
                .sample_iter(&Alphanumeric)
                .take(7)
                .map(char::from)
                .collect();
            let mut path = std::env::temp_dir();
            path.push(s);
            std::fs::create_dir(&path).unwrap();
            Self { path }
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.path).unwrap()
        }
    }

    #[tokio::test]
    async fn simple() {
        let tmp = TempDir::new();
        let api = ApiBuilder::new().with_cache_dir(&tmp.path).build().unwrap();
        let repo = Repo::new("julien-c/dummy-unknown".to_string());
        let downloaded_path = api.download(&repo, "config.json").await.unwrap();
        assert!(downloaded_path.exists());
    }
}
