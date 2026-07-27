//! A safe wrapper around `llama_model`.
use std::borrow::Cow;
use std::ffi::{c_char, CStr, CString};
use std::os::raw::c_int;
use std::path::Path;
use std::ptr::NonNull;
use std::str::Utf8Error;

use crate::chat_parser::{LlamaChatParams, LlamaGenerationParams};
use crate::context::params::LlamaContextParams;
use crate::context::LlamaContext;
use crate::llama_backend::LlamaBackend;
use crate::model::params::LlamaModelParams;
use crate::sampling::LlamaSampler;
use crate::token::LlamaToken;
use crate::vocab::LlamaVocab;
use crate::{
    ApplyChatTemplateError, ApplyChatTemplateErrorFull, ChatTemplateError, LlamaContextLoadError,
    LlamaLoraAdapterInitError, LlamaModelLoadError, MetaValError, NewLlamaChatMessageError,
    StringToTokenError, TokenToStringError,
};

pub mod params;
// For backwards compat.
pub use crate::vocab::{LlamaTokenTypeFromIntError, VocabType};

/// A safe wrapper around `llama_model`.
#[derive(Debug)]
#[repr(transparent)]
#[allow(clippy::module_name_repetitions)]
pub struct LlamaModel {
    pub(crate) model: NonNull<llama_cpp_sys_2::llama_model>,
}

/// A safe wrapper around `llama_lora_adapter`.
#[derive(Debug)]
#[repr(transparent)]
#[allow(clippy::module_name_repetitions)]
pub struct LlamaLoraAdapter {
    pub(crate) lora_adapter: NonNull<llama_cpp_sys_2::llama_adapter_lora>,
}

/// A performance-friendly wrapper around [`LlamaModel::chat_template`] which is then
/// fed into [`LlamaModel::apply_chat_template`] to convert a list of messages into an LLM
/// prompt. Internally the template is stored as a `CString` to avoid round-trip conversions
/// within the FFI.
#[derive(Eq, PartialEq, Clone, PartialOrd, Ord, Hash)]
pub struct LlamaChatTemplate(CString);

impl LlamaChatTemplate {
    /// Create a new template from a string. This can either be the name of a llama.cpp [chat template](https://github.com/ggerganov/llama.cpp/blob/8a8c4ceb6050bd9392609114ca56ae6d26f5b8f5/src/llama-chat.cpp#L27-L61)
    /// like "chatml" or "llama3" or an actual Jinja template for llama.cpp to interpret.
    pub fn new(template: &str) -> Result<Self, std::ffi::NulError> {
        Ok(Self(CString::new(template)?))
    }

    /// Accesses the template as a c string reference.
    pub fn as_c_str(&self) -> &CStr {
        &self.0
    }

    /// Attempts to convert the `CString` into a Rust str reference.
    pub fn to_str(&self) -> Result<&str, Utf8Error> {
        self.0.to_str()
    }

    /// Convenience method to create an owned String.
    pub fn to_string(&self) -> Result<String, Utf8Error> {
        self.to_str().map(str::to_string)
    }
}

impl std::fmt::Debug for LlamaChatTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// A Safe wrapper around `llama_chat_message`
#[derive(Debug, Eq, PartialEq, Clone)]
pub struct LlamaChatMessage {
    pub(crate) role: CString,
    pub(crate) content: CString,
    pub(crate) reasoning_content: CString,
    pub(crate) tool_name: CString,
    pub(crate) tool_call_id: CString,
    pub(crate) tool_calls: Vec<LlamaChatToolCall>,
}

impl LlamaChatMessage {
    /// Create a new `LlamaChatMessage`
    ///
    /// # Errors
    /// If either of ``role`` or ``content`` contain null bytes.
    pub fn new(role: String, content: String) -> Result<Self, NewLlamaChatMessageError> {
        Ok(Self {
            role: CString::new(role)?,
            content: CString::new(content)?,
            reasoning_content: CString::default(),
            tool_name: CString::default(),
            tool_call_id: CString::default(),
            tool_calls: Vec::default(),
        })
    }

    /// Sets the reasoning content for the chat message.
    ///
    /// # Errors
    /// If ``reasoning_content`` contains null bytes.
    pub fn with_reasoning_content(
        mut self,
        reasoning_content: String,
    ) -> Result<Self, NewLlamaChatMessageError> {
        self.reasoning_content = CString::new(reasoning_content)?;
        Ok(self)
    }

    /// Sets the tool name for the chat message.
    ///
    /// # Errors
    /// If ``tool_name`` contains null bytes.
    pub fn with_tool_name(mut self, tool_name: String) -> Result<Self, NewLlamaChatMessageError> {
        self.tool_name = CString::new(tool_name)?;
        Ok(self)
    }

    /// Sets the tool call ID for the chat message.
    ///
    /// # Errors
    /// If ``tool_call_id`` contains null bytes.
    pub fn with_tool_call_id(
        mut self,
        tool_call_id: String,
    ) -> Result<Self, NewLlamaChatMessageError> {
        self.tool_call_id = CString::new(tool_call_id)?;
        Ok(self)
    }

    /// Sets nested tool calls for the chat message.
    pub fn with_tool_calls(mut self, tool_calls: &[LlamaChatToolCall]) -> Self {
        self.tool_calls = tool_calls.to_vec();
        self
    }

    /// Gets the role of the chat message as a string slice.
    pub fn role<'a>(&'a self) -> Cow<'a, str> {
        self.role.to_string_lossy()
    }

    /// Gets the content of the chat message as a string slice.
    pub fn content<'a>(&'a self) -> Cow<'a, str> {
        self.content.to_string_lossy()
    }

    /// Gets the reasoning content of the chat message as a string slice.
    pub fn reasoning_content<'a>(&'a self) -> Cow<'a, str> {
        self.reasoning_content.to_string_lossy()
    }

    /// Gets the tool name of the chat message as a string slice.
    pub fn tool_name<'a>(&'a self) -> Cow<'a, str> {
        self.tool_name.to_string_lossy()
    }

    /// Gets the tool call ID of the chat message as a string slice.
    pub fn tool_call_id<'a>(&'a self) -> Cow<'a, str> {
        self.tool_call_id.to_string_lossy()
    }

    /// Gets the tool calls of the chat message as a slice of `LlamaChatToolCall`.
    pub fn tool_calls(&self) -> &[LlamaChatToolCall] {
        &self.tool_calls
    }
}

/// A Safe wrapper around `llama_chat_tool`
#[derive(Debug, Eq, PartialEq, Clone)]
pub struct LlamaChatTool {
    /// Name of the tool.
    pub(crate) name: CString,
    /// Description of the tool.
    pub(crate) description: CString,
    /// Stringified JSON schema for the tool's parameters.
    pub(crate) parameters: CString,
}

impl LlamaChatTool {
    /// Creates a new `LlamaChatTool`
    ///
    /// # Errors
    /// if `name`, `description`, or `parameters` contain null bytes.
    pub fn new(
        name: String,
        description: String,
        parameters: String,
    ) -> Result<Self, std::ffi::NulError> {
        Ok(Self {
            name: CString::new(name)?,
            description: CString::new(description)?,
            parameters: CString::new(parameters)?,
        })
    }

    /// Gets the name of the tool as a string slice.
    pub fn name<'a>(&'a self) -> Cow<'a, str> {
        self.name.to_string_lossy()
    }

    /// Gets the description of the tool as a string slice.
    pub fn description<'a>(&'a self) -> Cow<'a, str> {
        self.description.to_string_lossy()
    }

    /// Gets the parameters of the tool as a string slice.
    pub fn parameters<'a>(&'a self) -> Cow<'a, str> {
        self.parameters.to_string_lossy()
    }
}

/// A wrapper around `llama_chat_tool_call`
///
/// This is used for nested tool calls within an assistant message (i.e. when a model calls multiple tools in a single turn).
#[derive(Debug, Eq, PartialEq, Clone)]
pub struct LlamaChatToolCall {
    /// Name of the tool being called.
    pub(crate) name: CString,
    /// Stringified JSON arguments for the tool call.
    pub(crate) arguments: CString,
    /// Identifier for this tool call.
    pub(crate) id: CString,
}

impl LlamaChatToolCall {
    /// Creates a new `LlamaChatToolCall`
    ///
    /// # Errors
    /// if `name`, `arguments`, or `id` contain null bytes.
    pub fn new(name: String, arguments: String, id: String) -> Result<Self, std::ffi::NulError> {
        Ok(Self {
            name: CString::new(name)?,
            arguments: CString::new(arguments)?,
            id: CString::new(id)?,
        })
    }

    /// Gets the name of the tool call as a string slice.
    pub fn name<'a>(&'a self) -> Cow<'a, str> {
        self.name.to_string_lossy()
    }

    /// Gets the arguments of the tool call as a string slice.
    pub fn arguments<'a>(&'a self) -> Cow<'a, str> {
        self.arguments.to_string_lossy()
    }

    /// Gets the id of the tool call as a string slice.
    pub fn id<'a>(&'a self) -> Cow<'a, str> {
        self.id.to_string_lossy()
    }
}

/// The Rope type that's used within the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeType {
    Norm,
    NeoX,
    MRope,
    Vision,
}

unsafe impl Send for LlamaModel {}

unsafe impl Sync for LlamaModel {}

#[allow(deprecated)]
impl LlamaModel {
    /// Get the model's vocabulary.
    #[must_use]
    pub fn vocab(&self) -> LlamaVocab<'_> {
        let ptr = unsafe { llama_cpp_sys_2::llama_model_get_vocab(self.model.as_ptr()) };
        LlamaVocab::new(ptr).expect("model must have vocabulary")
    }

    /// get the number of tokens the model was trained on
    ///
    /// # Panics
    ///
    /// If the number of tokens the model was trained on does not fit into an `u32`. This should be impossible on most
    /// platforms due to llama.cpp returning a `c_int` (i32 on most platforms) which is almost certainly positive.
    #[must_use]
    pub fn n_ctx_train(&self) -> u32 {
        let n_ctx_train = unsafe { llama_cpp_sys_2::llama_n_ctx_train(self.model.as_ptr()) };
        u32::try_from(n_ctx_train).expect("n_ctx_train fits into an u32")
    }

    /// Get all tokens in the model.
    pub fn tokens(
        &self,
        decode_special: bool,
    ) -> impl Iterator<Item = (LlamaToken, Result<String, TokenToStringError>)> + '_ {
        (0..self.n_vocab())
            .map(LlamaToken::new)
            .map(move |llama_token| {
                let mut decoder = encoding_rs::UTF_8.new_decoder();
                (
                    llama_token,
                    self.token_to_piece(llama_token, &mut decoder, decode_special, None),
                )
            })
    }

    /// Get the beginning of stream token.
    #[must_use]
    pub fn token_bos(&self) -> LlamaToken {
        let token = unsafe { llama_cpp_sys_2::llama_vocab_bos(self.vocab_ptr()) };
        LlamaToken(token)
    }

    /// Get the end of stream token.
    #[must_use]
    pub fn token_eos(&self) -> LlamaToken {
        let token = unsafe { llama_cpp_sys_2::llama_vocab_eos(self.vocab_ptr()) };
        LlamaToken(token)
    }

    /// Get the newline token.
    #[must_use]
    pub fn token_nl(&self) -> LlamaToken {
        let token = unsafe { llama_cpp_sys_2::llama_vocab_nl(self.vocab_ptr()) };
        LlamaToken(token)
    }

    /// Check if a token represents the end of generation (end of turn, end of sequence, etc.)
    #[must_use]
    pub fn is_eog_token(&self, token: LlamaToken) -> bool {
        unsafe { llama_cpp_sys_2::llama_vocab_is_eog(self.vocab_ptr(), token.0) }
    }

    /// Get the decoder start token.
    #[must_use]
    pub fn decode_start_token(&self) -> LlamaToken {
        let token =
            unsafe { llama_cpp_sys_2::llama_model_decoder_start_token(self.model.as_ptr()) };
        LlamaToken(token)
    }

    /// Get the separator token (SEP).
    #[must_use]
    pub fn token_sep(&self) -> LlamaToken {
        let token = unsafe { llama_cpp_sys_2::llama_vocab_sep(self.vocab_ptr()) };
        LlamaToken(token)
    }

    /// Get the end of turn token.
    #[must_use]
    pub fn token_eot(&self) -> LlamaToken {
        let token = unsafe { llama_cpp_sys_2::llama_vocab_eot(self.vocab_ptr()) };
        LlamaToken(token)
    }

    /// Get the padding token.
    #[must_use]
    pub fn token_pad(&self) -> LlamaToken {
        let token = unsafe { llama_cpp_sys_2::llama_vocab_pad(self.vocab_ptr()) };
        LlamaToken(token)
    }

    /// Get the mask token.
    #[must_use]
    pub fn token_mask(&self) -> LlamaToken {
        let token = unsafe { llama_cpp_sys_2::llama_vocab_mask(self.vocab_ptr()) };
        LlamaToken(token)
    }

    /// Check if a token is a control token.
    #[must_use]
    pub fn is_control_token(&self, token: LlamaToken) -> bool {
        unsafe { llama_cpp_sys_2::llama_vocab_is_control(self.vocab_ptr(), token.0) }
    }

    /// Check if the vocabulary inherently expects a Beginning Of Stream (BOS) token to be added.
    #[must_use]
    pub fn should_add_bos_token(&self) -> bool {
        unsafe { llama_cpp_sys_2::llama_vocab_get_add_bos(self.vocab_ptr()) }
    }

    /// Check if the vocabulary inherently expects an End Of Stream (EOS) token to be added.
    #[must_use]
    pub fn should_add_eos_token(&self) -> bool {
        unsafe { llama_cpp_sys_2::llama_vocab_get_add_eos(self.vocab_ptr()) }
    }

    /// Check if the vocabulary inherently expects a Separator (SEP) token to be added.
    #[must_use]
    pub fn should_add_sep_token(&self) -> bool {
        unsafe { llama_cpp_sys_2::llama_vocab_get_add_sep(self.vocab_ptr()) }
    }

    /// Get the Fill-In-the-Middle (FIM) prefix token.
    #[must_use]
    pub fn token_fim_pre(&self) -> LlamaToken {
        let token = unsafe { llama_cpp_sys_2::llama_vocab_fim_pre(self.vocab_ptr()) };
        LlamaToken(token)
    }

    /// Get the Fill-In-the-Middle (FIM) suffix token.
    #[must_use]
    pub fn token_fim_suf(&self) -> LlamaToken {
        let token = unsafe { llama_cpp_sys_2::llama_vocab_fim_suf(self.vocab_ptr()) };
        LlamaToken(token)
    }

    /// Get the Fill-In-the-Middle (FIM) middle token.
    #[must_use]
    pub fn token_fim_mid(&self) -> LlamaToken {
        let token = unsafe { llama_cpp_sys_2::llama_vocab_fim_mid(self.vocab_ptr()) };
        LlamaToken(token)
    }

    /// Get the Fill-In-the-Middle (FIM) padding token.
    #[must_use]
    pub fn token_fim_pad(&self) -> LlamaToken {
        let token = unsafe { llama_cpp_sys_2::llama_vocab_fim_pad(self.vocab_ptr()) };
        LlamaToken(token)
    }

    /// Get the Fill-In-the-Middle (FIM) rep token.
    #[must_use]
    pub fn token_fim_rep(&self) -> LlamaToken {
        let token = unsafe { llama_cpp_sys_2::llama_vocab_fim_rep(self.vocab_ptr()) };
        LlamaToken(token)
    }

    /// Get the Fill-In-the-Middle (FIM) separator token.
    #[must_use]
    pub fn token_fim_sep(&self) -> LlamaToken {
        let token = unsafe { llama_cpp_sys_2::llama_vocab_fim_sep(self.vocab_ptr()) };
        LlamaToken(token)
    }

    /// Convert single token to a string.
    ///
    /// # Errors
    ///
    /// See [`TokenToStringError`] for more information.
    #[deprecated(since = "0.1.0", note = "Use `token_to_piece` instead")]
    pub fn token_to_str(
        &self,
        token: LlamaToken,
        special: Special,
    ) -> Result<String, TokenToStringError> {
        // TODO lsptrip None is acutally not quite the origignal behavior of this function,
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        self.token_to_piece(
            token,
            &mut decoder,
            matches!(special, Special::Tokenize),
            None,
        )
    }

    /// Convert single token to bytes.
    ///
    /// # Errors
    /// See [`TokenToStringError`] for more information.
    ///
    /// # Panics
    /// If a [`TokenToStringError::InsufficientBufferSpace`] error returned by
    /// [`Self::token_to_bytes_with_size`] contains a positive nonzero value. This should never
    /// happen.
    #[deprecated(since = "0.1.0", note = "Use `token_to_piece_bytes` instead")]
    pub fn token_to_bytes(
        &self,
        token: LlamaToken,
        special: Special,
    ) -> Result<Vec<u8>, TokenToStringError> {
        // TODO lsptrip None is acutally not quite the origignal behavior of this function,
        match self.token_to_piece_bytes(token, 8, matches!(special, Special::Tokenize), None) {
            Err(TokenToStringError::InsufficientBufferSpace(i)) => self.token_to_piece_bytes(
                token,
                (-i).try_into().expect("Error buffer size is positive"),
                matches!(special, Special::Tokenize),
                None,
            ),
            x => x,
        }
    }

    /// Convert a vector of tokens to a single string.
    ///
    /// # Errors
    ///
    /// See [`TokenToStringError`] for more information.
    #[deprecated(
        since = "0.1.0",
        note = "Use `token_to_piece` for each token individually instead"
    )]
    pub fn tokens_to_str(
        &self,
        tokens: &[LlamaToken],
        special: Special,
    ) -> Result<String, TokenToStringError> {
        let mut builder: Vec<u8> = Vec::with_capacity(tokens.len() * 4);
        for piece in tokens
            .iter()
            .copied()
            .map(|t| self.token_to_piece_bytes(t, 8, matches!(special, Special::Tokenize), None))
        {
            builder.extend_from_slice(&piece?);
        }
        Ok(String::from_utf8(builder)?)
    }

    /// Convert a string to a Vector of tokens.
    ///
    /// # Errors
    ///
    /// - if [`str`] contains a null byte.
    ///
    /// # Panics
    ///
    /// - if there is more than [`usize::MAX`] [`LlamaToken`]s in [`str`].
    ///
    ///
    /// ```no_run
    /// use llama_cpp_2::model::LlamaModel;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use std::path::Path;
    /// use llama_cpp_2::model::AddBos;
    /// let backend = llama_cpp_2::llama_backend::LlamaBackend::init()?;
    /// let model = LlamaModel::load_from_file(&backend, Path::new("path/to/model"), &Default::default())?;
    /// let tokens = model.str_to_token("Hello, World!", AddBos::Always)?;
    /// # Ok(())
    /// # }
    pub fn str_to_token(
        &self,
        str: &str,
        add_bos: AddBos,
    ) -> Result<Vec<LlamaToken>, StringToTokenError> {
        let add_bos = match add_bos {
            AddBos::Always => true,
            AddBos::Never => false,
        };

        let tokens_estimation = std::cmp::max(8, (str.len() / 2) + usize::from(add_bos));
        let mut buffer: Vec<LlamaToken> = Vec::with_capacity(tokens_estimation);

        let c_string = CString::new(str)?;
        let buffer_capacity =
            c_int::try_from(buffer.capacity()).expect("buffer capacity should fit into a c_int");

        let size = unsafe {
            llama_cpp_sys_2::llama_tokenize(
                self.vocab_ptr(),
                c_string.as_ptr(),
                c_int::try_from(c_string.as_bytes().len())?,
                buffer.as_mut_ptr().cast::<llama_cpp_sys_2::llama_token>(),
                buffer_capacity,
                add_bos,
                true,
            )
        };

        // if we fail the first time we can resize the vector to the correct size and try again. This should never fail.
        // as a result - size is guaranteed to be positive here.
        let size = if size.is_negative() {
            buffer.reserve_exact(usize::try_from(-size).expect("usize's are larger "));
            unsafe {
                llama_cpp_sys_2::llama_tokenize(
                    self.vocab_ptr(),
                    c_string.as_ptr(),
                    c_int::try_from(c_string.as_bytes().len())?,
                    buffer.as_mut_ptr().cast::<llama_cpp_sys_2::llama_token>(),
                    -size,
                    add_bos,
                    true,
                )
            }
        } else {
            size
        };

        let size = usize::try_from(size).expect("size is positive and usize ");

        // Safety: `size` < `capacity` and llama-cpp has initialized elements up to `size`
        unsafe { buffer.set_len(size) }
        Ok(buffer)
    }

    /// Get the type of a token.
    ///
    /// # Panics
    ///
    /// If the token type is not known to this library.
    #[must_use]
    pub fn token_attr(&self, LlamaToken(id): LlamaToken) -> LlamaTokenAttrs {
        let token_type = unsafe { llama_cpp_sys_2::llama_vocab_get_attr(self.vocab_ptr(), id) };
        LlamaTokenAttrs::try_from(token_type).expect("token type is valid")
    }

    /// Get the internal string representation of a token.
    ///
    /// Returns `None` if the token text is null or not valid UTF-8.
    #[must_use]
    pub fn token_get_text(&self, token: LlamaToken) -> Option<&std::ffi::CStr> {
        let text_ptr = unsafe { llama_cpp_sys_2::llama_vocab_get_text(self.vocab_ptr(), token.0) };
        if text_ptr.is_null() {
            None
        } else {
            Some(unsafe { std::ffi::CStr::from_ptr(text_ptr) })
        }
    }

    /// Get the score of a token.
    #[must_use]
    pub fn token_get_score(&self, token: LlamaToken) -> f32 {
        unsafe { llama_cpp_sys_2::llama_vocab_get_score(self.vocab_ptr(), token.0) }
    }

    /// Convert a token to a string using the underlying llama.cpp `llama_token_to_piece` function.
    ///
    /// This is the new default function for token decoding and provides direct access to
    /// the llama.cpp token decoding functionality without any special logic or filtering.
    ///
    /// Decoding raw string requires using an decoder, tokens from language models may not always map
    /// to full characters depending on the encoding so stateful decoding is required, otherwise partial strings may be lost!
    /// Invalid characters are mapped to REPLACEMENT CHARACTER making the method safe to use even if the model inherently produces
    /// garbage.
    ///
    /// # Errors
    ///
    /// - if the token type is unknown
    ///
    /// # Panics
    ///
    /// - if the returned size from llama-cpp does not fit into a [`usize`]. (this should never happen)
    pub fn token_to_piece(
        &self,
        token: LlamaToken,
        decoder: &mut encoding_rs::Decoder,
        special: bool,
        lstrip: Option<NonZeroU16>,
    ) -> Result<String, TokenToStringError> {
        let bytes = match self.token_to_piece_bytes(token, 8, special, lstrip) {
            // when there is insufficient space `token_to_piece` will return a negative number with the size that would have been returned
            // https://github.com/abetlen/llama-cpp-python/blob/c37132bac860fcc333255c36313f89c4f49d4c8d/llama_cpp/llama_cpp.py#L3461
            Err(TokenToStringError::InsufficientBufferSpace(i)) => self.token_to_piece_bytes(
                token,
                (-i).try_into().expect("Error buffer size is positive"),
                special,
                lstrip,
            ),
            x => x,
        }?;
        // here the assumption is that each byte from the output may map to at most one output charakter
        let mut output_piece = String::with_capacity(bytes.len());
        // _result only tells if there is nothing more in the input, or if the output was full
        // but further decoding will happen on the next interation anyway
        let (_result, _somesize, _truthy) =
            decoder.decode_to_string(&bytes, &mut output_piece, false);
        Ok(output_piece)
    }

    /// Raw token decoding to bytes, use if you want to handle the decoding model output yourself
    ///
    /// Convert a token to bytes using the underlying llama.cpp `llama_token_to_piece` function. This is mostly
    /// a thin wrapper around `llama_token_to_piece` function, that handles rust <-> c type conversions while
    /// letting the caller handle errors. For a safer inteface returing rust strings directly use `token_to_piece` instead!
    ///
    /// # Errors
    ///
    /// - if the token type is unknown
    /// - the resultant token is larger than `buffer_size`.
    ///
    /// # Panics
    ///
    /// - if `buffer_size` does not fit into a [`c_int`].
    /// - if the returned size from llama-cpp does not fit into a [`usize`]. (this should never happen)
    pub fn token_to_piece_bytes(
        &self,
        token: LlamaToken,
        buffer_size: usize,
        special: bool,
        lstrip: Option<NonZeroU16>,
    ) -> Result<Vec<u8>, TokenToStringError> {
        let string = CString::new(vec![b'*'; buffer_size]).expect("no null");
        let len = string.as_bytes().len();
        let len = c_int::try_from(len).expect("length fits into c_int");
        let buf = string.into_raw();
        let lstrip = lstrip.map_or(0, |it| i32::from(it.get()));
        let size = unsafe {
            llama_cpp_sys_2::llama_token_to_piece(
                self.vocab_ptr(),
                token.0,
                buf,
                len,
                lstrip,
                special,
            )
        };

        match size {
            0 => Err(TokenToStringError::UnknownTokenType),
            i if i.is_negative() => Err(TokenToStringError::InsufficientBufferSpace(i)),
            size => {
                let string = unsafe { CString::from_raw(buf) };
                let mut bytes = string.into_bytes();
                let len = usize::try_from(size).expect("size is positive and fits into usize");
                bytes.truncate(len);
                Ok(bytes)
            }
        }
    }

    /// Convert a token to a string with a specified buffer size.
    ///
    /// Generally you should use [`LlamaModel::token_to_str`] as it is able to decode tokens with
    /// any length.
    ///
    /// # Errors
    ///
    /// - if the token type is unknown
    /// - the resultant token is larger than `buffer_size`.
    /// - the string returend by llama-cpp is not valid utf8.
    ///
    /// # Panics
    ///
    /// - if `buffer_size` does not fit into a [`c_int`].
    /// - if the returned size from llama-cpp does not fit into a [`usize`]. (this should never happen)
    #[deprecated(since = "0.1.0", note = "Use `token_to_piece` instead")]
    pub fn token_to_str_with_size(
        &self,
        token: LlamaToken,
        buffer_size: usize,
        special: Special,
    ) -> Result<String, TokenToStringError> {
        let bytes = self.token_to_piece_bytes(
            token,
            buffer_size,
            matches!(special, Special::Tokenize),
            None,
        )?;
        Ok(String::from_utf8(bytes)?)
    }

    /// Convert a token to bytes with a specified buffer size.
    ///
    /// Generally you should use [`LlamaModel::token_to_bytes`] as it is able to handle tokens of
    /// any length.
    ///
    /// # Errors
    ///
    /// - if the token type is unknown
    /// - the resultant token is larger than `buffer_size`.
    ///
    /// # Panics
    ///
    /// - if `buffer_size` does not fit into a [`c_int`].
    /// - if the returned size from llama-cpp does not fit into a [`usize`]. (this should never happen)
    #[deprecated(since = "0.1.0", note = "Use `token_to_piece_bytes` instead")]
    pub fn token_to_bytes_with_size(
        &self,
        token: LlamaToken,
        buffer_size: usize,
        special: Special,
        lstrip: Option<NonZeroU16>,
    ) -> Result<Vec<u8>, TokenToStringError> {
        if token == self.token_nl() {
            return Ok(b"\n".to_vec());
        }

        // unsure what to do with this in the face of the 'special' arg + attr changes
        let attrs = self.token_attr(token);
        if attrs.is_empty()
            || attrs
                .intersects(LlamaTokenAttr::Unknown | LlamaTokenAttr::Byte | LlamaTokenAttr::Unused)
            || attrs.contains(LlamaTokenAttr::Control)
                && (token == self.token_bos() || token == self.token_eos())
        {
            return Ok(Vec::new());
        }

        let special = match special {
            Special::Tokenize => true,
            Special::Plaintext => false,
        };

        let string = CString::new(vec![b'*'; buffer_size]).expect("no null");
        let len = string.as_bytes().len();
        let len = c_int::try_from(len).expect("length fits into c_int");
        let buf = string.into_raw();
        let lstrip = lstrip.map_or(0, |it| i32::from(it.get()));
        let size = unsafe {
            llama_cpp_sys_2::llama_token_to_piece(
                self.vocab_ptr(),
                token.0,
                buf,
                len,
                lstrip,
                special,
            )
        };

        match size {
            0 => Err(TokenToStringError::UnknownTokenType),
            i if i.is_negative() => Err(TokenToStringError::InsufficientBufferSpace(i)),
            size => {
                let string = unsafe { CString::from_raw(buf) };
                let mut bytes = string.into_bytes();
                let len = usize::try_from(size).expect("size is positive and fits into usize");
                bytes.truncate(len);
                Ok(bytes)
            }
        }
    }
    /// The number of tokens the model was trained on.
    ///
    /// This returns a `c_int` for maximum compatibility. Most of the time it can be cast to an i32
    /// without issue.
    #[must_use]
    pub fn n_vocab(&self) -> i32 {
        self.vocab().n_tokens()
    }

    /// This returns a `c_int` for maximum compatibility. Most of the time it can be cast to an i32
    /// without issue.
    #[must_use]
    pub fn n_embd(&self) -> c_int {
        unsafe { llama_cpp_sys_2::llama_n_embd(self.model.as_ptr()) }
    }

    /// The model's *output* embedding width (`n_embd_out`). This is the width
    /// llama.cpp actually extracts embeddings at — `n_embd` and `n_embd_out`
    /// diverge when `{arch}.embedding_length_out` is present (deepstack models
    /// like qwen3vl). Returns a `c_int` for maximum compatibility.
    #[must_use]
    pub fn n_embd_out(&self) -> c_int {
        unsafe { llama_cpp_sys_2::llama_model_n_embd_out(self.model.as_ptr()) }
    }

    /// The model's classification output width (`n_cls_out`, default 1) — the
    /// width of a RANK-pooled embeddings read (llama.h:1029).
    #[must_use]
    pub fn n_cls_out(&self) -> u32 {
        unsafe { llama_cpp_sys_2::llama_model_n_cls_out(self.model.as_ptr()) }
    }

    /// Returns the total size of all the tensors in the model in bytes.
    pub fn size(&self) -> u64 {
        unsafe { llama_cpp_sys_2::llama_model_size(self.model.as_ptr()) }
    }

    /// Returns the number of parameters in the model.
    pub fn n_params(&self) -> u64 {
        unsafe { llama_cpp_sys_2::llama_model_n_params(self.model.as_ptr()) }
    }

    /// Returns whether the model is a recurrent network (Mamba, RWKV, etc)
    pub fn is_recurrent(&self) -> bool {
        unsafe { llama_cpp_sys_2::llama_model_is_recurrent(self.model.as_ptr()) }
    }

    /// Returns whether the model is a hybrid network (Jamba, Granite, Qwen3xx, etc)
    ///
    /// Hybrid models have both attention layers and recurrent/SSM layers.
    /// They require special handling for state checkpointing.
    pub fn is_hybrid(&self) -> bool {
        unsafe { llama_cpp_sys_2::llama_model_is_hybrid(self.model.as_ptr()) }
    }

    /// Returns the number of layers within the model.
    pub fn n_layer(&self) -> u32 {
        // It's never possible for this to panic because while the API interface is defined as an int32_t,
        // the field it's accessing is a uint32_t.
        u32::try_from(unsafe { llama_cpp_sys_2::llama_model_n_layer(self.model.as_ptr()) }).unwrap()
    }

    /// Returns the number of attention heads within the model.
    pub fn n_head(&self) -> u32 {
        // It's never possible for this to panic because while the API interface is defined as an int32_t,
        // the field it's accessing is a uint32_t.
        u32::try_from(unsafe { llama_cpp_sys_2::llama_model_n_head(self.model.as_ptr()) }).unwrap()
    }

    /// Returns the number of KV attention heads.
    pub fn n_head_kv(&self) -> u32 {
        // It's never possible for this to panic because while the API interface is defined as an int32_t,
        // the field it's accessing is a uint32_t.
        u32::try_from(unsafe { llama_cpp_sys_2::llama_model_n_head_kv(self.model.as_ptr()) })
            .unwrap()
    }

    /// Get metadata value as a string by key name
    pub fn meta_val_str(&self, key: &str) -> Result<String, MetaValError> {
        let key_cstring = CString::new(key)?;
        let key_ptr = key_cstring.as_ptr();

        extract_meta_string(
            |buf_ptr, buf_len| unsafe {
                llama_cpp_sys_2::llama_model_meta_val_str(
                    self.model.as_ptr(),
                    key_ptr,
                    buf_ptr,
                    buf_len,
                )
            },
            256,
        )
    }

    /// Get the number of metadata key/value pairs
    pub fn meta_count(&self) -> i32 {
        unsafe { llama_cpp_sys_2::llama_model_meta_count(self.model.as_ptr()) }
    }

    /// Get metadata key name by index
    pub fn meta_key_by_index(&self, index: i32) -> Result<String, MetaValError> {
        extract_meta_string(
            |buf_ptr, buf_len| unsafe {
                llama_cpp_sys_2::llama_model_meta_key_by_index(
                    self.model.as_ptr(),
                    index,
                    buf_ptr,
                    buf_len,
                )
            },
            256,
        )
    }

    /// Get metadata value as a string by index
    pub fn meta_val_str_by_index(&self, index: i32) -> Result<String, MetaValError> {
        extract_meta_string(
            |buf_ptr, buf_len| unsafe {
                llama_cpp_sys_2::llama_model_meta_val_str_by_index(
                    self.model.as_ptr(),
                    index,
                    buf_ptr,
                    buf_len,
                )
            },
            256,
        )
    }

    /// Returns the rope type of the model.
    pub fn rope_type(&self) -> Option<RopeType> {
        match unsafe { llama_cpp_sys_2::llama_model_rope_type(self.model.as_ptr()) } {
            llama_cpp_sys_2::LLAMA_ROPE_TYPE_NONE => None,
            llama_cpp_sys_2::LLAMA_ROPE_TYPE_NORM => Some(RopeType::Norm),
            llama_cpp_sys_2::LLAMA_ROPE_TYPE_NEOX => Some(RopeType::NeoX),
            llama_cpp_sys_2::LLAMA_ROPE_TYPE_MROPE => Some(RopeType::MRope),
            llama_cpp_sys_2::LLAMA_ROPE_TYPE_VISION => Some(RopeType::Vision),
            rope_type => {
                tracing::error!(rope_type = rope_type, "Unexpected rope type from llama.cpp");
                None
            }
        }
    }

    /// Get chat template from model by name. If the name parameter is None, the default chat template will be returned.
    ///
    /// You supply this into [`Self::apply_chat_template`] to get back a string with the appropriate template
    /// substitution applied to convert a list of messages into a prompt the LLM can use to complete
    /// the chat.
    ///
    /// You could also use an external jinja parser, like [minijinja](https://github.com/mitsuhiko/minijinja),
    /// to parse jinja templates not supported by the llama.cpp template engine.
    ///
    /// # Errors
    ///
    /// * If the model has no chat template by that name
    /// * If the chat template is not a valid [`CString`].
    pub fn chat_template(
        &self,
        name: Option<&str>,
    ) -> Result<LlamaChatTemplate, ChatTemplateError> {
        let name_cstr = name.map(CString::new);
        let name_ptr = match name_cstr {
            Some(Ok(name)) => name.as_ptr(),
            _ => std::ptr::null(),
        };
        let result =
            unsafe { llama_cpp_sys_2::llama_model_chat_template(self.model.as_ptr(), name_ptr) };

        // Convert result to Rust String if not null
        if result.is_null() {
            Err(ChatTemplateError::MissingTemplate)
        } else {
            let chat_template_cstr = unsafe { CStr::from_ptr(result) };
            let chat_template = CString::new(chat_template_cstr.to_bytes())?;
            Ok(LlamaChatTemplate(chat_template))
        }
    }

    /// Loads a model from a file.
    ///
    /// # Errors
    ///
    /// See [`LlamaModelLoadError`] for more information.
    #[tracing::instrument(skip_all, fields(params))]
    pub fn load_from_file(
        _: &LlamaBackend,
        path: impl AsRef<Path>,
        params: &LlamaModelParams,
    ) -> Result<Self, LlamaModelLoadError> {
        let path = path.as_ref();
        debug_assert!(Path::new(path).exists(), "{path:?} does not exist");
        let path = path
            .to_str()
            .ok_or(LlamaModelLoadError::PathToStrError(path.to_path_buf()))?;

        let cstr = CString::new(path)?;
        let llama_model =
            unsafe { llama_cpp_sys_2::llama_load_model_from_file(cstr.as_ptr(), params.params) };

        let model = NonNull::new(llama_model).ok_or(LlamaModelLoadError::NullResult)?;

        tracing::debug!(?path, "Loaded model");
        Ok(LlamaModel { model })
    }

    /// Initializes a lora adapter from a file.
    ///
    /// # Errors
    ///
    /// See [`LlamaLoraAdapterInitError`] for more information.
    pub fn lora_adapter_init(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<LlamaLoraAdapter, LlamaLoraAdapterInitError> {
        let path = path.as_ref();
        debug_assert!(Path::new(path).exists(), "{path:?} does not exist");

        let path = path
            .to_str()
            .ok_or(LlamaLoraAdapterInitError::PathToStrError(
                path.to_path_buf(),
            ))?;

        let cstr = CString::new(path)?;
        let adapter =
            unsafe { llama_cpp_sys_2::llama_adapter_lora_init(self.model.as_ptr(), cstr.as_ptr()) };

        let adapter = NonNull::new(adapter).ok_or(LlamaLoraAdapterInitError::NullResult)?;

        tracing::debug!(?path, "Initialized lora adapter");
        Ok(LlamaLoraAdapter {
            lora_adapter: adapter,
        })
    }

    /// Create a new context from this model.
    ///
    /// # Errors
    ///
    /// There is many ways this can fail. See [`LlamaContextLoadError`] for more information.
    // we intentionally do not derive Copy on `LlamaContextParams` to allow llama.cpp to change the type to be non-trivially copyable.
    #[allow(clippy::needless_pass_by_value)]
    pub fn new_context<'a>(
        &'a self,
        _: &LlamaBackend,
        params: LlamaContextParams,
    ) -> Result<LlamaContext<'a>, LlamaContextLoadError> {
        let context_params = params.context_params;
        let context = unsafe {
            llama_cpp_sys_2::llama_new_context_with_model(self.model.as_ptr(), context_params)
        };
        let context = NonNull::new(context).ok_or(LlamaContextLoadError::NullReturn)?;

        Ok(LlamaContext::new(self, context, params.embeddings()))
    }

    /// Create a new context bound to another context via llama.cpp's `ctx_other` field.
    ///
    /// This is required for MTP speculative decoding when the target model's
    /// architecture uses `LLM_ARCH_GEMMA4_ASSISTANT`, which asserts that the draft
    /// context references the target context so KV state can be shared.
    ///
    /// # Errors
    ///
    /// See [`LlamaContextLoadError`].
    #[allow(clippy::needless_pass_by_value)]
    pub fn new_context_with_ctx_other<'a>(
        &'a self,
        _: &LlamaBackend,
        params: LlamaContextParams,
        ctx_other: &LlamaContext<'_>,
    ) -> Result<LlamaContext<'a>, LlamaContextLoadError> {
        let mut context_params = params.context_params;
        context_params.ctx_other = ctx_other.context.as_ptr();
        let context = unsafe {
            llama_cpp_sys_2::llama_new_context_with_model(self.model.as_ptr(), context_params)
        };
        let context = NonNull::new(context).ok_or(LlamaContextLoadError::NullReturn)?;

        Ok(LlamaContext::new(self, context, params.embeddings()))
    }

    /// Creates a new context with backend samplers attached for specific sequences.
    ///
    /// Ownership of the samplers is transferred to the context, ensuring they remain
    /// alive for the context's lifetime. Only samplers that support backend execution
    /// (greedy, dist, temp, top_k, top_p, min_p, logit_bias) will run on the backend.
    ///
    /// # Arguments
    ///
    /// * `params` - Context parameters
    /// * `samplers` - Iterator of `(seq_id, sampler)` pairs where sampler must be a chain
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let sampler = LlamaSampler::chain([
    ///     LlamaSampler::min_p(0.01, 64),
    ///     LlamaSampler::temp(0.1),
    ///     LlamaSampler::dist(42),
    /// ], false);
    ///
    /// let ctx = model.new_context_with_samplers(
    ///     &backend,
    ///     ctx_params,
    ///     [(0, sampler)],
    /// )?;
    /// ```
    #[allow(clippy::needless_pass_by_value)]
    pub fn new_context_with_samplers<'a>(
        &'a self,
        _: &LlamaBackend,
        params: LlamaContextParams,
        samplers: impl IntoIterator<Item = (i32, LlamaSampler)>,
    ) -> Result<LlamaContext<'a>, LlamaContextLoadError> {
        let samplers: Vec<_> = samplers.into_iter().collect();
        let mut context_params = params.context_params;

        let mut sampler_configs: Vec<llama_cpp_sys_2::llama_sampler_seq_config> = samplers
            .iter()
            .map(
                |(seq_id, sampler)| llama_cpp_sys_2::llama_sampler_seq_config {
                    seq_id: *seq_id,
                    sampler: sampler.sampler,
                },
            )
            .collect();

        if !sampler_configs.is_empty() {
            context_params.samplers = sampler_configs.as_mut_ptr();
            context_params.n_samplers = sampler_configs.len();
        }

        let context = unsafe {
            llama_cpp_sys_2::llama_new_context_with_model(self.model.as_ptr(), context_params)
        };
        let context = NonNull::new(context).ok_or(LlamaContextLoadError::NullReturn)?;

        Ok(LlamaContext::with_samplers(
            self,
            context,
            params.embeddings(),
            samplers,
        ))
    }

    /// Apply the models chat template to some messages.
    /// See <https://github.com/ggerganov/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template>
    ///
    /// Unlike the llama.cpp `apply_chat_template` which just randomly uses the ChatML template when given
    /// a null pointer for the template, this requires an explicit template to be specified. If you want to
    /// use "chatml", then just do `LlamaChatTemplate::new("chatml")` or any other model name or template
    /// string.
    ///
    /// Use [`Self::chat_template`] to retrieve the template baked into the model (this is the preferred
    /// mechanism as using the wrong chat template can result in really unexpected responses from the LLM).
    ///
    /// You probably want to set `add_ass` to true so that the generated template string ends with a the
    /// opening tag of the assistant. If you fail to leave a hanging chat tag, the model will likely generate
    /// one into the output and the output may also have unexpected output aside from that.
    ///
    /// # Errors
    /// There are many ways this can fail. See [`ApplyChatTemplateError`] for more information.
    #[tracing::instrument(skip_all)]
    pub fn apply_chat_template(
        &self,
        tmpl: &LlamaChatTemplate,
        chat: &[LlamaChatMessage],
        add_ass: bool,
    ) -> Result<String, ApplyChatTemplateError> {
        // Buffer is twice the length of messages per their recommendation
        let message_length = chat.iter().fold(0, |acc, c| {
            acc + c.role.to_bytes().len() + c.content.to_bytes().len()
        });
        let mut buff: Vec<u8> = vec![0; message_length * 2];

        // Build our llama_cpp_sys_2 chat messages
        let chat: Vec<llama_cpp_sys_2::llama_chat_message> = chat
            .iter()
            .map(|c| llama_cpp_sys_2::llama_chat_message {
                role: c.role.as_ptr(),
                content: c.content.as_ptr(),
            })
            .collect();

        let tmpl_ptr = tmpl.0.as_ptr();

        let res = unsafe {
            llama_cpp_sys_2::llama_chat_apply_template(
                tmpl_ptr,
                chat.as_ptr(),
                chat.len(),
                add_ass,
                buff.as_mut_ptr().cast::<c_char>(),
                buff.len().try_into().expect("Buffer size exceeds i32::MAX"),
            )
        };

        if res < 0 {
            return Err(ApplyChatTemplateError::FfiError(res));
        }

        if res > buff.len().try_into().expect("Buffer size exceeds i32::MAX") {
            buff.resize(res.try_into().expect("res is negative"), 0);

            let res = unsafe {
                llama_cpp_sys_2::llama_chat_apply_template(
                    tmpl_ptr,
                    chat.as_ptr(),
                    chat.len(),
                    add_ass,
                    buff.as_mut_ptr().cast::<c_char>(),
                    buff.len().try_into().expect("Buffer size exceeds i32::MAX"),
                )
            };
            if res < 0 {
                return Err(ApplyChatTemplateError::FfiError(res));
            }
            assert_eq!(Ok(res), buff.len().try_into());
        }
        buff.truncate(res.try_into().expect("res is negative"));
        Ok(String::from_utf8(buff)?)
    }

    /// Apply the model's chat template with parameters.
    ///
    /// This uses `common_chat_templates_apply` under the hood (with jinja enabled),
    /// which will then try to init a model-tailored `LlamaChatParams`, or fall
    /// back to llama.cpp's new PEG parser (autoparser) under the hood.
    ///
    /// Either way, this supports parsing complex Jinja templates,
    /// as well as generating grammar constrained inputs and triggers [crate::chat_parser::LlamaGrammarTrigger] via a GGUF based grammar, or JSON schema format.
    ///
    /// The latter is useful for creating a sampler that lazily applies grammar constraints, e.g. to allow unrestricted reasoning, but grammar constrained output (see [common_sampler](https://github.com/ggml-org/llama.cpp/blob/571d0d540df04f25298d0e159e520d9fc62ed121/common/sampling.cpp#L111)
    /// and [common_sampler_init](https://github.com/ggml-org/llama.cpp/blob/571d0d540df04f25298d0e159e520d9fc62ed121/common/sampling.cpp#L187) for inspiration).
    ///
    /// Optional: You can provide an override to the model's baked in chat template by passing `Some(&LlamaChatTemple)`,
    /// otherwise the model's default template will be used instead (same template that can be derived from [`Self::chat_template`]).
    ///  
    /// Use [`LlamaGenerationParams`] to configure and build the generation parameters.
    ///
    /// # Errors
    /// There are many ways this can fail. See [`ApplyChatTemplateFullError`] for more information.
    #[tracing::instrument(skip_all)]
    pub fn apply_chat_template_with_params(
        &self,
        tmpl: Option<&LlamaChatTemplate>,
        params: &LlamaGenerationParams,
    ) -> Result<LlamaChatParams, ApplyChatTemplateErrorFull> {
        let mut gen_params_state = params.as_ptr()?;

        let out_chat_params = unsafe {
            llama_cpp_sys_2::common_chat_apply_template(
                self.model.as_ptr(),
                tmpl.map_or(ptr::null(), |t| t.as_c_str().as_ptr()),
                gen_params_state.get(),
            )
        };

        if out_chat_params.is_null() {
            Err(ApplyChatTemplateErrorFull::LlamaCppException)
        } else {
            match LlamaChatParams::new(out_chat_params) {
                Ok(params) => Ok(params),
                Err(e) => {
                    unsafe {
                        llama_cpp_sys_2::common_chat_params_free(out_chat_params);
                    }
                    Err(e.into())
                }
            }
        }
    }

    /// Formats a single chat message using the model's chat template.
    ///
    /// This is useful for evaluating formatting differences for partial updates.
    ///
    /// Optional: You can provide an override to the model's baked in chat template by passing `Some(&LlamaChatTemple)` to `tmpl`,
    /// otherwise the model's default template will be used instead (same template that can be derived from [`Self::chat_template`]).
    ///
    /// You probably want to set `add_ass` to true so that the generated string ends with the
    /// opening tag of the assistant. If you fail to leave a hanging chat tag, the model will likely generate
    /// one into the output and the output may also have unexpected output aside from that.
    #[tracing::instrument(skip_all)]
    pub fn chat_format_single(
        &self,
        tmpl: Option<&LlamaChatTemplate>,
        past_params: &LlamaGenerationParams,
        new_msg: &LlamaChatMessage,
        add_ass: bool,
        use_jinja: bool,
    ) -> Result<String, ApplyChatTemplateErrorFull> {
        let mut past_msg_state = past_params.as_ptr()?;

        let new_msg_gen_params = past_params.clone().with_messages(&[new_msg.clone()]);
        let mut new_msg_state = new_msg_gen_params.as_ptr()?;

        let formatted_str = unsafe {
            llama_cpp_sys_2::llama_rs_common_chat_format_single(
                self.model.as_ptr(),
                tmpl.map_or(ptr::null(), |t| t.as_c_str().as_ptr()),
                past_msg_state.get(),
                new_msg_state.get(),
                add_ass,
                use_jinja,
            )
        };

        if formatted_str.is_null() {
            Err(ApplyChatTemplateErrorFull::LlamaCppException)
        } else {
            let result = unsafe { std::ffi::CStr::from_ptr(formatted_str) }
                .to_string_lossy()
                .into_owned();
            unsafe { llama_cpp_sys_2::llama_rs_string_free(formatted_str) };
            Ok(result)
        }
    }
}

/// Generic helper function for extracting string values from the C API
/// This are specifically useful for the the metadata functions, where we pass in a buffer
/// to be populated by a string, not yet knowing if the buffer is large enough.
/// If the buffer was not large enough, we get the correct length back, which can be used to
/// construct a buffer of appropriate size.
fn extract_meta_string<F>(c_function: F, capacity: usize) -> Result<String, MetaValError>
where
    F: Fn(*mut c_char, usize) -> i32,
{
    let mut buffer = vec![0u8; capacity];

    // call the foreign function
    let result = c_function(buffer.as_mut_ptr().cast::<c_char>(), buffer.len());
    if result < 0 {
        return Err(MetaValError::NegativeReturn(result));
    }

    // check if the response fit in our buffer
    let returned_len = result as usize;
    if returned_len >= capacity {
        // buffer wasn't large enough, try again with the correct capacity.
        return extract_meta_string(c_function, returned_len + 1);
    }

    // verify null termination
    debug_assert_eq!(
        buffer.get(returned_len),
        Some(&0),
        "should end with null byte"
    );

    // resize, convert, and return
    buffer.truncate(returned_len);
    Ok(String::from_utf8(buffer)?)
}

impl Drop for LlamaModel {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys_2::llama_free_model(self.model.as_ptr()) }
    }
}
