use std::{
    borrow::Cow,
    ffi::{c_char, CStr, CString, NulError},
    ptr,
};

use llama_cpp_sys_2::{
    common_chat_msg_diffs, common_chat_msg_diffs_free, common_chat_msg_diffs_get_size,
    common_chat_msg_diffs_get_view, common_chat_params, common_chat_params_free,
    common_chat_params_get_grammar_trigger, common_chat_params_get_grammar_triggers_count,
    common_chat_params_get_message_delimiter, common_chat_params_get_message_delimiters_count,
    common_chat_params_get_preserved_token, common_chat_params_get_preserved_tokens_count,
    common_chat_params_get_view, common_chat_params_view, common_chat_templates_inputs,
    common_chat_templates_inputs_add_message, common_chat_templates_inputs_add_tool,
    common_chat_templates_inputs_add_tool_call_to_last_message,
    common_chat_templates_inputs_create, common_chat_templates_inputs_free, llama_rs_chat_parser,
    llama_rs_chat_parser_feed, llama_rs_chat_parser_free, llama_rs_chat_parser_init,
};

use crate::model::{LlamaChatMessage, LlamaChatTool, LlamaChatToolCall};
use crate::token::LlamaToken;

/// Errors that can occur when initializing the ChatParser
#[derive(Debug, thiserror::Error)]
pub enum ChatParserInitError {
    /// Failed to initialize parser parameters: C++ returned a null pointer.
    #[error("Failed to initialize parser parameters: C++ returned a null pointer")]
    NullParamsReturn,
    /// Failed to allocate initial chat message state.
    #[error("Failed to allocate initial chat message state")]
    NullStateReturn,
    /// Failed to convert a string to a CString.
    #[error("{0}")]
    NulError(#[from] NulError),
}
/// Errors that can occur while feeding tokens into the parser
#[derive(Debug, thiserror::Error)]
pub enum ChatParserFeedError {
    /// Invalid argument passed to the Llama.cpp parser.
    #[error("Invalid argument passed to the Llama.cpp parser")]
    InvalidArgument,
    /// Exception thrown by the Llama.cpp parser.
    #[error("Exception thrown by the Llama.cpp parser")]
    Exception,
    /// Failed to convert a string to a CString.
    #[error("{0}")]
    NulError(#[from] NulError),
    /// Failed to compute diffs for a chat message.
    #[error("{0}")]
    ChatDiffCreationError(#[from] ChatDiffCreationError),
}

/// A safe wrapping struct to use `common_chat_parse` and `common_chat_msg_diff::compute_diffs`.
#[derive(Debug, Clone)]
pub struct ChatParser {
    /// Pointer to the C++ parser state engine.
    ptr: *mut llama_rs_chat_parser,
}

// SAFETY: The underlying C++ structs are purely heap-allocated and do not use thread-local storage.
unsafe impl Send for ChatParser {}

impl ChatParser {
    /// Initializes a new ChatParser using the provided chat parameters.
    ///
    /// These parameters are typically obtained by calling `apply_chat_template_full`
    /// on the model.
    ///
    /// # Errors
    /// - Returns a [ChatParserInitError::NullStateReturn] if the underlying C++ allocations fail.
    /// - Returns a [ChatParserInitError::NullParamsReturn] if `generation_params.as_ptr()` fails.
    pub fn new(
        chat_params: &LlamaChatParams,
        generation_params: &LlamaGenerationParams,
    ) -> Result<Self, ChatParserInitError> {
        let mut gen_params_state = generation_params.as_ptr()?;
        let ptr = unsafe { llama_rs_chat_parser_init(chat_params.ptr, gen_params_state.get()) };
        if ptr.is_null() {
            return Err(ChatParserInitError::NullStateReturn);
        }
        Ok(Self { ptr })
    }

    /// Feeds a newly generated token piece (as bytes) into the parser.
    ///
    /// If the newly added bytes end in the middle of a multi-byte UTF-8 character,
    /// this method will safely buffer the bytes and return `Ok(None)` to wait for the
    /// rest of the character in the next token.
    ///
    /// # Errors
    /// Returns [ChatParserFeedError::InvalidArgument] if the input `piece` is null.
    /// Returns [ChatParserFeedError::Exception] if llama.cpp throws an exception.
    /// Returns [ChatParserFeedError::NulError] if `piece` contains null bytes.
    /// Returns [ChatParserFeedError::ChatDiffCreationError] if `diffs_ptr` is null/missing.
    pub fn feed(&mut self, piece: &str) -> Result<Vec<ChatDiff>, ChatParserFeedError> {
        self.feed_piece(piece, true)
    }

    /// Call this to "repair" the feed parser and get the tool call diff following the end of a tool call, ie EOS.
    pub fn finish(&mut self) -> Result<Vec<ChatDiff>, ChatParserFeedError> {
        self.feed_piece("", false)
    }

    /// Feeds a newly generated token piece (as bytes) into the parser.
    ///
    /// If the newly added bytes end in the middle of a multi-byte UTF-8 character,
    /// this method will safely buffer the bytes and return `Ok(None)` to wait for the
    /// rest of the character in the next token.
    ///
    /// # Errors
    /// Returns [ChatParserFeedError::InvalidArgument] if the input `piece` is null.
    /// Returns [ChatParserFeedError::Exception] if llama.cpp throws an exception.
    /// Returns [ChatParserFeedError::NulError] if `piece` contains null bytes.
    /// Returns [ChatParserFeedError::ChatDiffCreationError] if `diffs_ptr` is null/missing.
    fn feed_piece(
        &mut self,
        piece: &str,
        is_partial: bool,
    ) -> Result<Vec<ChatDiff>, ChatParserFeedError> {
        let mut diffs_ptr = ptr::null_mut();
        let diffs_res: llama_cpp_sys_2::llama_rs_status = unsafe {
            llama_rs_chat_parser_feed(
                self.ptr,
                CString::new(piece)?.as_ptr(),
                is_partial,
                &mut diffs_ptr,
            )
        };
        match diffs_res {
            llama_cpp_sys_2::LLAMA_RS_STATUS_OK => {
                let len = unsafe { common_chat_msg_diffs_get_size(diffs_ptr) };
                if len == 0 {
                    unsafe { common_chat_msg_diffs_free(diffs_ptr) };
                    return Ok(Vec::new());
                }

                let mut diffs = Vec::with_capacity(len);
                for i in 0..len {
                    match ChatDiff::new(diffs_ptr, i) {
                        Ok(diff) => diffs.push(diff),
                        Err(e) => {
                            unsafe {
                                common_chat_msg_diffs_free(diffs_ptr);
                            }
                            return Err(e.into());
                        }
                    }
                }
                unsafe {
                    common_chat_msg_diffs_free(diffs_ptr);
                }
                Ok(diffs)
            }
            llama_cpp_sys_2::LLAMA_RS_STATUS_INVALID_ARGUMENT => {
                return Err(ChatParserFeedError::InvalidArgument)
            }
            llama_cpp_sys_2::LLAMA_RS_STATUS_EXCEPTION | _ => {
                return Err(ChatParserFeedError::Exception)
            }
        }
    }
}

impl Drop for ChatParser {
    fn drop(&mut self) {
        unsafe { llama_rs_chat_parser_free(self.ptr) }
    }
}

/// Errors for creating a `ChatDiff`
#[derive(Debug, thiserror::Error)]
pub enum ChatDiffCreationError {
    /// This occurs if the diffs or out_view is null/missing.
    #[error("Invalid argument passed to the Llama.cpp parser")]
    InvalidArgument,

    /// Occurs if a valid input diff returns a view that is completely empty/null.
    #[error("Exception in Llama.cpp occured")]
    Exception,
}

/// A Safe wrapper around `common_chat_diff`.
#[derive(Debug, Clone)]
pub struct ChatDiff {
    content: Option<String>,
    reasoning: Option<String>,
    tool_call_index: Option<usize>,
    tool_call: Option<LlamaChatToolCall>,
}

impl ChatDiff {
    /// Creates a new [ChatDiff].
    ///
    /// # Errors
    /// - Returns [ChatDiffCreationError::InvalidArgument] if `diff` is null.
    /// - Returns [ChatDiffCreationError::Exception] if llama.cpp returns a null view for the given `diff`.
    pub fn new(
        diff: *mut common_chat_msg_diffs,
        index: usize,
    ) -> Result<Self, ChatDiffCreationError> {
        if diff.is_null() {
            return Err(ChatDiffCreationError::InvalidArgument);
        }

        let view = unsafe { common_chat_msg_diffs_get_view(diff, index) };

        let content = Self::get_opt_string(view.content);
        let reasoning = Self::get_opt_string(view.reasoning_content);
        let tool_call_name = Self::get_opt_string(view.tool_call_name);
        let tool_call_id = Self::get_opt_string(view.tool_call_id);
        let tool_call_arguments = Self::get_opt_string(view.tool_call_arguments);
        let tool_call_index = if view.tool_call_index == usize::MAX {
            None
        } else {
            Some(view.tool_call_index)
        };

        if content.is_none()
            && reasoning.is_none()
            && tool_call_name.is_none()
            && tool_call_id.is_none()
            && tool_call_arguments.is_none()
        {
            return Err(ChatDiffCreationError::Exception);
        }

        Ok(Self {
            content,
            reasoning,
            tool_call_index,
            tool_call: if tool_call_name.is_some()
                || tool_call_arguments.is_some()
                || tool_call_id.is_some()
            {
                match LlamaChatToolCall::new(
                    tool_call_name.unwrap_or_default(),
                    tool_call_arguments.unwrap_or_default(),
                    tool_call_id.unwrap_or_default(),
                ) {
                    Ok(tc) => Some(tc),
                    Err(_) => None,
                }
            } else {
                None
            },
        })
    }

    /// Gets the standard textual content delta.
    ///
    /// # Returns
    /// - `Some(Cow::Borrowed(&str))` if the `common_chat_diff` has content.
    /// - `None` if no content was generated.
    pub fn content(&self) -> Option<Cow<'_, str>> {
        self.content.as_deref().map(Cow::Borrowed)
    }

    /// Gets the reasoning content delta.
    ///
    /// # Returns
    /// - `Some(Cow::Borrowed(&str))` if the `common_chat_diff` has reasoning content.
    /// - `None` if no reasoning content was generated.
    pub fn reasoning(&self) -> Option<Cow<'_, str>> {
        self.reasoning.as_deref().map(Cow::Borrowed)
    }

    /// Gets the tool call delta.
    pub fn tool_call(&self) -> Option<LlamaChatToolCall> {
        self.tool_call.clone()
    }

    /// Gets the tool call index delta.
    pub fn tool_call_index(&self) -> Option<usize> {
        self.tool_call_index
    }

    fn get_opt_string(ptr: *const i8) -> Option<String> {
        unsafe {
            if ptr.is_null() {
                None
            } else {
                let res = CStr::from_ptr(ptr).to_string_lossy().into_owned();
                if res.is_empty() {
                    None
                } else {
                    Some(res)
                }
            }
        }
    }
}

/// Errors in creating a `LlamaChatParams`
#[derive(Debug, thiserror::Error)]
pub enum ChatParamsCreationError {
    /// Failed to create chat params view.
    #[error("Failed to create chat params view")]
    ViewCreationFailed,

    /// Invalid argument passed to the Llama.cpp parser
    #[error("Invalid argument passed to the Llama.cpp parser")]
    InvalidArgument,
}

/// Safe wrapper around `common_chat_params`.
#[derive(Debug, Clone)]
pub struct LlamaChatParams {
    /// Raw pointer to `common_chat_params`.
    pub ptr: *mut common_chat_params,
    /// Raw point to a view of the `common_chat_params`.
    pub view: common_chat_params_view,
}

impl LlamaChatParams {
    /// Creates a new `LlamaChatParams` from a raw pointer + a view to it.
    ///
    /// # Errors
    /// - Returns [ChatParamsCreationError::InvalidArgument] if `ptr` is null.
    pub fn new(ptr: *mut common_chat_params) -> Result<Self, ChatParamsCreationError> {
        if ptr.is_null() {
            return Err(ChatParamsCreationError::InvalidArgument);
        }
        let view = unsafe { common_chat_params_get_view(ptr) };
        Ok(Self { ptr, view })
    }

    /// Returns a string slice of the generated chat prompt.
    pub fn prompt(&self) -> &str {
        unsafe { CStr::from_ptr(self.view.prompt) }
            .to_str()
            .unwrap_or("")
    }

    /// Returns a safe Rust view of the chat params view.
    pub fn view(&self) -> LlamaChatParamsView {
        let get_cstring = |ptr: *const i8| -> CString {
            unsafe {
                if ptr.is_null() {
                    CString::default()
                } else {
                    CStr::from_ptr(ptr).to_owned()
                }
            }
        };
        let preserved_tokens = unsafe {
            let count = common_chat_params_get_preserved_tokens_count(self.ptr);
            let mut tokens = Vec::with_capacity(count);
            for i in 0..count {
                let token_ptr = common_chat_params_get_preserved_token(self.ptr, i);
                if !token_ptr.is_null() {
                    tokens.push(CStr::from_ptr(token_ptr).to_owned());
                }
            }
            tokens
        };
        let grammar_triggers = unsafe {
            let count = common_chat_params_get_grammar_triggers_count(self.ptr);
            let mut triggers = Vec::with_capacity(count);
            for i in 0..count {
                let view = common_chat_params_get_grammar_trigger(self.ptr, i);
                triggers.push(LlamaGrammarTrigger {
                    trigger_type: <llama_cpp_sys_2::llama_rs_common_grammar_trigger_type>::from(
                        view.type_ as u32,
                    )
                    .into(),
                    value: get_cstring(view.value),
                    token: LlamaToken::new(view.token),
                });
            }
            triggers
        };
        let message_delimiters = unsafe {
            let count = common_chat_params_get_message_delimiters_count(self.ptr);
            let mut delimiters = Vec::with_capacity(count);
            for i in 0..count {
                let view = common_chat_params_get_message_delimiter(self.ptr, i);
                let mut tokens = Vec::with_capacity(view.tokens_count);
                if !view.tokens.is_null() && view.tokens_count > 0 {
                    let slice = std::slice::from_raw_parts(view.tokens, view.tokens_count);
                    for &t in slice {
                        tokens.push(LlamaToken::new(t));
                    }
                }
                delimiters.push(LlamaChatMessageDelimiter {
                    role: <llama_cpp_sys_2::llama_rs_common_chat_role>::from(view.role as u32)
                        .into(),
                    delimiter: get_cstring(view.delimiter),
                    tokens,
                });
            }
            delimiters
        };
        LlamaChatParamsView {
            format: <llama_cpp_sys_2::llama_rs_common_chat_format>::from(self.view.format as u32)
                .into(),
            prompt: get_cstring(self.view.prompt),
            grammar: get_cstring(self.view.grammar),
            grammar_lazy: self.view.grammar_lazy,
            generation_prompt: get_cstring(self.view.generation_prompt),
            supports_thinking: self.view.supports_thinking,
            thinking_start_tag: get_cstring(self.view.thinking_start_tag),
            thinking_end_tag: get_cstring(self.view.thinking_end_tag),
            preserved_tokens,
            parser: get_cstring(self.view.parser),
            grammar_triggers,
            message_delimiters,
        }
    }
}

impl Drop for LlamaChatParams {
    fn drop(&mut self) {
        unsafe {
            common_chat_params_free(self.ptr);
        }
    }
}

/// Format variant for chat
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LlamaChatFormat {
    ///These are intended to be parsed by the PEG parser
    #[default]
    ContentOnly = 0,
    /// These are intended to be parsed by the PEG parser
    PegSimple = 1,
    /// These are intended to be parsed by the PEG parser
    PegNative = 2,
    /// These are intended to be parsed by the PEG parser
    PegGemma4 = 3,
    /// Not a format, just the # formats"]
    Count = 4,
}

impl From<llama_cpp_sys_2::llama_rs_common_chat_format> for LlamaChatFormat {
    fn from(value: llama_cpp_sys_2::llama_rs_common_chat_format) -> Self {
        match value {
            llama_cpp_sys_2::LLAMA_RS_COMMON_CHAT_FORMAT_CONTENT_ONLY => Self::ContentOnly,
            llama_cpp_sys_2::LLAMA_RS_COMMON_CHAT_FORMAT_PEG_SIMPLE => Self::PegSimple,
            llama_cpp_sys_2::LLAMA_RS_COMMON_CHAT_FORMAT_PEG_NATIVE => Self::PegNative,
            llama_cpp_sys_2::LLAMA_RS_COMMON_CHAT_FORMAT_PEG_GEMMA4 => Self::PegGemma4,
            llama_cpp_sys_2::LLAMA_RS_COMMON_CHAT_FORMAT_COUNT => Self::Count,
            _ => Self::default(),
        }
    }
}

impl Into<llama_cpp_sys_2::llama_rs_common_chat_format> for LlamaChatFormat {
    fn into(self) -> llama_cpp_sys_2::llama_rs_common_chat_format {
        match self {
            Self::ContentOnly => llama_cpp_sys_2::LLAMA_RS_COMMON_CHAT_FORMAT_CONTENT_ONLY,
            Self::PegSimple => llama_cpp_sys_2::LLAMA_RS_COMMON_CHAT_FORMAT_PEG_SIMPLE,
            Self::PegNative => llama_cpp_sys_2::LLAMA_RS_COMMON_CHAT_FORMAT_PEG_NATIVE,
            Self::PegGemma4 => llama_cpp_sys_2::LLAMA_RS_COMMON_CHAT_FORMAT_PEG_GEMMA4,
            Self::Count => llama_cpp_sys_2::LLAMA_RS_COMMON_CHAT_FORMAT_COUNT,
        }
    }
}

/// Safe struct for `common_chat_params_view`
#[derive(Debug, Clone)]
pub struct LlamaChatParamsView {
    /// Chat format
    pub format: LlamaChatFormat,
    /// Formatted prompt.
    pub prompt: CString,
    /// Grammar constraint.
    pub grammar: CString,
    /// Whether the grammar is lazy (will be triggered by tokens).
    pub grammar_lazy: bool,
    /// Generation prompt.
    pub generation_prompt: CString,
    /// Whether the model supports thinking.
    pub supports_thinking: bool,
    /// " e.g., \"<think>\""
    pub thinking_start_tag: CString,
    /// e.g., \"</think>\""
    pub thinking_end_tag: CString,
    /// Think tags, tool call tags, etc.
    pub preserved_tokens: Vec<CString>,
    /// Parser (used to load the PEG Arena).
    pub parser: CString,
    /// Grammar triggers (lazy triggers).
    pub grammar_triggers: Vec<LlamaGrammarTrigger>,
    /// Message delimiters
    pub message_delimiters: Vec<LlamaChatMessageDelimiter>,
}

/// Chat role enum mapping to llama.cpp's `common_chat_role`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamaChatRole {
    Unknown,
    System,
    Assistant,
    User,
    Tool,
}

impl From<llama_cpp_sys_2::llama_rs_common_chat_role> for LlamaChatRole {
    fn from(value: llama_cpp_sys_2::llama_rs_common_chat_role) -> Self {
        match value {
            llama_cpp_sys_2::LLAMA_RS_COMMON_CHAT_ROLE_UNKNOWN => Self::Unknown,
            llama_cpp_sys_2::LLAMA_RS_COMMON_CHAT_ROLE_SYSTEM => Self::System,
            llama_cpp_sys_2::LLAMA_RS_COMMON_CHAT_ROLE_ASSISTANT => Self::Assistant,
            llama_cpp_sys_2::LLAMA_RS_COMMON_CHAT_ROLE_USER => Self::User,
            llama_cpp_sys_2::LLAMA_RS_COMMON_CHAT_ROLE_TOOL => Self::Tool,
            _ => Self::Unknown,
        }
    }
}

/// A parsed message delimiter definition.
#[derive(Debug, Clone)]
pub struct LlamaChatMessageDelimiter {
    pub role: LlamaChatRole,
    pub delimiter: CString,
    pub tokens: Vec<LlamaToken>,
}

/// Grammar trigger type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamaGrammarTriggerType {
    /// Trigger on a specific token.
    Token,
    /// Trigger on a specific word.
    Word,
    /// Trigger on a regex pattern.
    Pattern,
    /// Trigger on a full regex pattern.
    PatternFull,
}

impl From<llama_cpp_sys_2::llama_rs_common_grammar_trigger_type> for LlamaGrammarTriggerType {
    fn from(value: llama_cpp_sys_2::llama_rs_common_grammar_trigger_type) -> Self {
        match value {
            llama_cpp_sys_2::LLAMA_RS_COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN => Self::Token,
            llama_cpp_sys_2::LLAMA_RS_COMMON_GRAMMAR_TRIGGER_TYPE_WORD => Self::Word,
            llama_cpp_sys_2::LLAMA_RS_COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN => Self::Pattern,
            llama_cpp_sys_2::LLAMA_RS_COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL => Self::PatternFull,
            _ => Self::Word, // fallback
        }
    }
}

/// Grammar trigger
#[derive(Debug, Clone)]
pub struct LlamaGrammarTrigger {
    /// Type of the trigger
    pub trigger_type: LlamaGrammarTriggerType,
    /// Value (string or regex)
    pub value: CString,
    /// Token ID
    pub token: LlamaToken,
}

/// Convenience struct which gets converted to `generation_params`.
/// See `generation_params` for defaults: https://github.com/ggml-org/llama.cpp/blob/32beb244f5c2ca91c583be15d4671643b54ba238/common/chat-auto-parser.h#L54
#[derive(Debug, Clone)]
pub struct LlamaGenerationParams {
    /// Message history in order.
    pub messages: Vec<LlamaChatMessage>,
    /// Tools to be aware of.
    pub tools: Vec<LlamaChatTool>,
    /// Add generation prompt to the prompt.
    /// - Defaults to `false`
    pub add_generation_prompt: bool,
    /// Enable thinking.
    /// - Defaults to `true`.
    pub enable_thinking: bool,

    /// Reasoning format.
    /// - Defaults to [LlamaReasoningFormat::AUTO].
    pub reasoning_format: LlamaReasoningFormat,

    /// Chat continuation.
    /// - Defaults to [LlamaChatContinuation::NONE].
    pub continue_final_message: LlamaChatContinuation,

    /// Stringified JSON object for Jinja kwargs
    pub extra_context: Option<CString>,

    /// Stringified JSON schema for constrained output
    /// - If grammar is set, this will be ignored.
    pub json_schema: Option<CString>,
    /// Grammar to use for constrained output
    /// - If this is set, `json_schema` will be ignored.
    pub grammar: Option<CString>,

    /// Enable parallel tool calls.
    /// - Defaults to `true`.
    pub parallel_tool_calls: bool,
    /// Add beginning of sentence token.
    /// - Defaults to `false`.
    pub add_bos: bool,
    /// Add end of sentence token.
    /// - Defaults to `false`.
    pub add_eos: bool,
}

/// Safe wrapper around `common_chat_templates_inputs`.
#[derive(Debug)]
pub struct LlamaGenerationParamsPtr {
    ptr: *mut common_chat_templates_inputs,
}

impl LlamaGenerationParamsPtr {
    /// Gets a mutable pointer to the raw common_chat_templates_inputs.
    pub fn get(&mut self) -> *mut common_chat_templates_inputs {
        self.ptr
    }
}

impl Drop for LlamaGenerationParamsPtr {
    fn drop(&mut self) {
        unsafe {
            if !self.ptr.is_null() {
                common_chat_templates_inputs_free(self.ptr);
            }
        }
    }
}

impl LlamaGenerationParams {
    /// Creates a new `LlamaGenerationParams` with the default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the message history to use for this generation.
    pub fn with_messages(mut self, messages: &[LlamaChatMessage]) -> Self {
        self.messages = messages.to_vec();
        self
    }

    /// Set the tools available to the model.
    pub fn with_tools(mut self, tools: &[LlamaChatTool]) -> Self {
        self.tools = tools.to_vec();
        self
    }

    /// Set whether to add the generation prompt.
    /// - Defaults to `false`
    pub fn with_add_generation_prompt(mut self, add_generation_prompt: bool) -> Self {
        self.add_generation_prompt = add_generation_prompt;
        self
    }

    /// Set whether to enable thinking.
    /// - Defaults to `true`
    pub fn with_enable_thinking(mut self, enable_thinking: bool) -> Self {
        self.enable_thinking = enable_thinking;
        self
    }

    /// Set the reasoning format.
    ///
    /// This is a legacy property that was mostly replaced by the `enable_thinking` boolean.
    /// It's only usage is for the [ChatParser] to determine whether to inline the reasoning output
    /// into the content field ([LlamaReasoningFormat::DEEPSEEK_LEGACY]), or to
    /// extract the reasoning from the output ([LlamaReasoningFormat::DEEPSEEK]/[LlamaReasoningFormat::AUTO]) and route it to the reasoning field.
    ///
    /// # Defaults
    ///
    /// - If `enable_thinking` is false, and this is left unset, this will default to [LlamaReasoningFormat::NONE].
    /// - If `enable_thinking` is true, and this is left unset, this will default to [LlamaReasoningFormat::AUTO]
    pub fn with_reasoning_format(mut self, reasoning_format: LlamaReasoningFormat) -> Self {
        self.reasoning_format = reasoning_format;
        self
    }

    /// Set the continuation type.
    ///
    /// This is used by the [ChatParser] to determine the proper routing of tokens following
    /// an interrupted generation.
    pub fn with_continue_final_message(mut self, continuation: LlamaChatContinuation) -> Self {
        self.continue_final_message = continuation;
        self
    }

    /// Set the extra context to use for this generation.
    /// - Defaults to `None`
    pub fn with_extra_context(mut self, extra_context: &str) -> Self {
        self.extra_context = Some(CString::new(extra_context).unwrap_or_default());
        self
    }

    /// Set the JSON schema to use for constrained output.
    /// - If grammar is set, this will be ignored.
    /// - Defaults to `None`
    pub fn with_json_schema(mut self, json_schema: &str) -> Self {
        self.json_schema = Some(CString::new(json_schema).unwrap_or_default());
        self
    }

    /// Set the grammar to use for constrained output.
    /// - If this is set, `json_schema` will be ignored.
    /// - Defaults to `None`
    pub fn with_grammar(mut self, grammar: &str) -> Self {
        self.grammar = Some(CString::new(grammar).unwrap_or_default());
        self
    }

    /// Set whether to enable parallel tool calls.
    /// - Defaults to `true`
    pub fn with_parallel_tool_calls(mut self, parallel_tool_calls: bool) -> Self {
        self.parallel_tool_calls = parallel_tool_calls;
        self
    }

    /// Set whether to add the beginning of sentence token.
    /// - Defaults to `false`
    pub fn with_add_bos(mut self, add_bos: bool) -> Self {
        self.add_bos = add_bos;
        self
    }

    /// Set whether to add the end of sentence token.
    /// - Defaults to `false`
    pub fn with_add_eos(mut self, add_eos: bool) -> Self {
        self.add_eos = add_eos;
        self
    }

    /// Creates a pointer to `common_chat_templates_inputs`.
    pub fn as_ptr(&self) -> Result<LlamaGenerationParamsPtr, NulError> {
        let get_opt_ptr = |cstr: &Option<CString>| -> *const c_char {
            match cstr {
                Some(c) => c.as_ptr(),
                None => ptr::null(),
            }
        };

        let get_ptr = |cstr: &CString| -> *const c_char {
            if cstr.is_empty() {
                return ptr::null();
            }
            cstr.as_ptr()
        };

        let reasoning_format: llama_cpp_sys_2::llama_rs_common_reasoning_format =
            if self.reasoning_format != LlamaReasoningFormat::default() {
                self.reasoning_format
            } else {
                if self.enable_thinking {
                    LlamaReasoningFormat::AUTO
                } else {
                    LlamaReasoningFormat::NONE
                }
            }
            .into();

        let continuation: llama_cpp_sys_2::llama_rs_common_chat_continuation =
            self.continue_final_message.into();

        let ptr = unsafe {
            common_chat_templates_inputs_create(
                self.add_generation_prompt,
                self.enable_thinking,
                reasoning_format as i32,
                continuation as i32,
                self.parallel_tool_calls,
                self.add_bos,
                self.add_eos,
                get_opt_ptr(&self.json_schema),
                get_opt_ptr(&self.grammar),
                get_opt_ptr(&self.extra_context),
            )
        };

        for msg in &self.messages {
            unsafe {
                common_chat_templates_inputs_add_message(
                    ptr,
                    get_ptr(&msg.role),
                    get_ptr(&msg.content),
                    get_ptr(&msg.reasoning_content),
                    get_ptr(&msg.tool_name),
                    get_ptr(&msg.tool_call_id),
                );
            }

            for tc in &msg.tool_calls {
                unsafe {
                    common_chat_templates_inputs_add_tool_call_to_last_message(
                        ptr,
                        get_ptr(&tc.name),
                        get_ptr(&tc.arguments),
                        get_ptr(&tc.id),
                    );
                }
            }
        }

        for tool in &self.tools {
            unsafe {
                common_chat_templates_inputs_add_tool(
                    ptr,
                    get_ptr(&tool.name),
                    get_ptr(&tool.description),
                    get_ptr(&tool.parameters),
                );
            }
        }

        Ok(LlamaGenerationParamsPtr { ptr })
    }
}

impl Default for LlamaGenerationParams {
    fn default() -> Self {
        Self {
            messages: Vec::<LlamaChatMessage>::default(),
            tools: Vec::<LlamaChatTool>::default(),
            add_generation_prompt: false,
            enable_thinking: true,
            reasoning_format: LlamaReasoningFormat::AUTO,
            continue_final_message: LlamaChatContinuation::NONE,
            extra_context: None,
            json_schema: None,
            grammar: None,
            parallel_tool_calls: true,
            add_bos: false,
            add_eos: false,
        }
    }
}

/// Chat continuation method provided via `with_continue_final_message`. Only used by [ChatParser].
///
/// This enum determines how content is resumed following a partial generation.
///
/// See the below for more details and inspiration:
/// - `chat_parse`: https://github.com/ggml-org/llama.cpp/blob/86a9c79f866799eb0e7e89c03578ccfbcc5d808e/common/chat.cpp#L2859
/// - `server-task.cpp`: https://github.com/ggml-org/llama.cpp/blob/86a9c79f866799eb0e7e89c03578ccfbcc5d808e/tools/server/server-task.cpp#L158
#[derive(Debug, Clone, Copy, Default)]
pub enum LlamaChatContinuation {
    /// Don't resume the final message (eg new prompt)
    #[default]
    NONE,
    /// Auto resume either reasoning or content generation.
    AUTO,
    /// Resume reasoning generation.
    REASONING,
    /// Resume content generation.
    CONTENT,
}

impl From<llama_cpp_sys_2::llama_rs_common_chat_continuation> for LlamaChatContinuation {
    fn from(value: llama_cpp_sys_2::llama_rs_common_chat_continuation) -> Self {
        match value {
            llama_cpp_sys_2::LLAMA_RS_COMMON_CHAT_CONTINUATION_NONE => Self::NONE,
            llama_cpp_sys_2::LLAMA_RS_COMMON_CHAT_CONTINUATION_AUTO => Self::AUTO,
            llama_cpp_sys_2::LLAMA_RS_COMMON_CHAT_CONTINUATION_REASONING => Self::REASONING,
            llama_cpp_sys_2::LLAMA_RS_COMMON_CHAT_CONTINUATION_CONTENT => Self::CONTENT,
            _ => Self::default(),
        }
    }
}

impl Into<llama_cpp_sys_2::llama_rs_common_chat_continuation> for LlamaChatContinuation {
    fn into(self) -> llama_cpp_sys_2::llama_rs_common_chat_continuation {
        match self {
            Self::NONE => llama_cpp_sys_2::LLAMA_RS_COMMON_CHAT_CONTINUATION_NONE,
            Self::AUTO => llama_cpp_sys_2::LLAMA_RS_COMMON_CHAT_CONTINUATION_AUTO,
            Self::REASONING => llama_cpp_sys_2::LLAMA_RS_COMMON_CHAT_CONTINUATION_REASONING,
            Self::CONTENT => llama_cpp_sys_2::LLAMA_RS_COMMON_CHAT_CONTINUATION_CONTENT,
        }
    }
}

/// Reasoning API response format (not to be confused as chat template's
/// reasoning format) only used by [ChatParser].
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum LlamaReasoningFormat {
    /// Skip reasoning extraction.
    NONE,
    /// Same as deepseek, using `message.reasoning_content`
    #[default]
    AUTO,
    /// Extract thinking tag contents and return as `message.reasoning_content`,
    /// or leave inline in <think> tags in stream mode
    DEEPSEEK_LEGACY,
    /// Extract thinking tag contents and return as `message.reasoning_content`,
    /// including in streaming deltas.
    DEEPSEEK,
}

impl From<llama_cpp_sys_2::llama_rs_common_reasoning_format> for LlamaReasoningFormat {
    fn from(value: llama_cpp_sys_2::llama_rs_common_reasoning_format) -> Self {
        match value {
            llama_cpp_sys_2::LLAMA_RS_COMMON_REASONING_FORMAT_NONE => Self::NONE,
            llama_cpp_sys_2::LLAMA_RS_COMMON_REASONING_FORMAT_AUTO => Self::AUTO,
            llama_cpp_sys_2::LLAMA_RS_COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY => {
                Self::DEEPSEEK_LEGACY
            }
            llama_cpp_sys_2::LLAMA_RS_COMMON_REASONING_FORMAT_DEEPSEEK => Self::DEEPSEEK,
            _ => Self::default(),
        }
    }
}

impl Into<llama_cpp_sys_2::llama_rs_common_reasoning_format> for LlamaReasoningFormat {
    fn into(self) -> llama_cpp_sys_2::llama_rs_common_reasoning_format {
        match self {
            Self::NONE => llama_cpp_sys_2::LLAMA_RS_COMMON_REASONING_FORMAT_NONE,
            Self::AUTO => llama_cpp_sys_2::LLAMA_RS_COMMON_REASONING_FORMAT_AUTO,
            Self::DEEPSEEK_LEGACY => {
                llama_cpp_sys_2::LLAMA_RS_COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY
            }
            Self::DEEPSEEK => llama_cpp_sys_2::LLAMA_RS_COMMON_REASONING_FORMAT_DEEPSEEK,
        }
    }
}
