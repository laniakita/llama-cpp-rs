use std::{
    borrow::Cow,
    ffi::{CStr, CString, FromBytesWithNulError, NulError},
    ptr,
};

use llama_cpp_sys_2::{
    llama_rs_chat_msg_diff_get_view, llama_rs_chat_msg_diff_view, llama_rs_chat_msg_diff_view_free,
    llama_rs_chat_msg_diff_view_init, llama_rs_chat_msg_diffs_len, llama_rs_chat_parser,
    llama_rs_chat_parser_feed, llama_rs_chat_parser_free, llama_rs_chat_parser_init,
    llama_rs_chat_template_generation_params, llama_rs_common_chat_continuation,
    llama_rs_common_chat_format, llama_rs_common_chat_msg_diffs,
    llama_rs_common_chat_msg_diffs_free, llama_rs_common_chat_msg_diffs_init,
    llama_rs_common_chat_params, llama_rs_common_chat_params_free,
    llama_rs_common_chat_params_view, llama_rs_common_chat_params_view_free,
    llama_rs_common_chat_params_view_init, llama_rs_common_chat_role,
    llama_rs_common_grammar_trigger_type, llama_rs_common_reasoning_format, llama_rs_status,
    LLAMA_RS_COMMON_CHAT_CONTINUATION_AUTO, LLAMA_RS_COMMON_CHAT_CONTINUATION_CONTENT,
    LLAMA_RS_COMMON_CHAT_CONTINUATION_NONE, LLAMA_RS_COMMON_CHAT_CONTINUATION_REASONING,
    LLAMA_RS_COMMON_CHAT_FORMAT_CONTENT_ONLY, LLAMA_RS_COMMON_CHAT_FORMAT_COUNT,
    LLAMA_RS_COMMON_CHAT_FORMAT_PEG_GEMMA4, LLAMA_RS_COMMON_CHAT_FORMAT_PEG_NATIVE,
    LLAMA_RS_COMMON_CHAT_FORMAT_PEG_SIMPLE, LLAMA_RS_COMMON_CHAT_ROLE_ASSISTANT,
    LLAMA_RS_COMMON_CHAT_ROLE_SYSTEM, LLAMA_RS_COMMON_CHAT_ROLE_TOOL,
    LLAMA_RS_COMMON_CHAT_ROLE_UNKNOWN, LLAMA_RS_COMMON_CHAT_ROLE_USER,
    LLAMA_RS_COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN,
    LLAMA_RS_COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL, LLAMA_RS_COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN,
    LLAMA_RS_COMMON_GRAMMAR_TRIGGER_TYPE_WORD, LLAMA_RS_COMMON_REASONING_FORMAT_AUTO,
    LLAMA_RS_COMMON_REASONING_FORMAT_DEEPSEEK, LLAMA_RS_COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY,
    LLAMA_RS_COMMON_REASONING_FORMAT_NONE, LLAMA_RS_STATUS_EXCEPTION,
    LLAMA_RS_STATUS_INVALID_ARGUMENT, LLAMA_RS_STATUS_OK,
};

use crate::{
    model::{LlamaChatMessageFull, LlamaChatTool, LlamaChatToolCall, LlamaModel},
    token::LlamaToken,
};

/// Errors that can occur when initializing the ChatParser
#[derive(Debug, thiserror::Error)]
pub enum ChatParserInitError {
    #[error("Failed to initialize parser parameters: C++ returned a null pointer")]
    NullParamsReturn,
    #[error("Failed to allocate initial chat message state")]
    NullStateReturn,
    #[error("{0}")]
    NulError(#[from] NulError),
}
/// Errors that can occur while feeding tokens into the parser
#[derive(Debug, thiserror::Error)]
pub enum ChatParserFeedError {
    #[error("Invalid argument passed to the Llama.cpp parser")]
    InvalidArgument,
    #[error("Exception thrown by the Llama.cpp parser")]
    Exception,

    #[error("{0}")]
    NulError(#[from] NulError),
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
    /// - Returns a [ChatParserInitError::] if the underlying C++ allocations fail.
    /// - Returns a [ChatParserInitError::FromBytesWithNulError] if `generation_params.as_ptr()` fails.
    pub fn new(
        chat_params: &LlamaChatParams,
        generation_params: &LlamaGenerationParams,
    ) -> Result<Self, ChatParserInitError> {
        // Under the hood, LlamaChatParams is now just a safe wrapper around
        // *mut llama_rs_common_chat_params. We pass its pointer down to the C++
        // engine initialization!
        let mut gen_params_state = generation_params.into_state()?;
        let ptr =
            unsafe { llama_rs_chat_parser_init(chat_params.ptr, &mut gen_params_state.params) };
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
    /// Returns `ChatParserFeedError` if the C++ engine fails.
    pub fn feed_piece<'a>(
        &'a mut self,
        piece: &str,
    ) -> Result<Vec<ChatDiff<'a>>, ChatParserFeedError> {
        let mut diffs_ptr = ptr::null_mut();
        let diffs_res: llama_rs_status = unsafe {
            llama_rs_chat_parser_feed(self.ptr, CString::new(piece)?.as_ptr(), &mut diffs_ptr)
        };
        match diffs_res {
            LLAMA_RS_STATUS_OK => {
                let len = unsafe { llama_rs_chat_msg_diffs_len(diffs_ptr) };
                if len == 0 {
                    unsafe { llama_rs_common_chat_msg_diffs_free(diffs_ptr) };
                    return Ok(Vec::new());
                }

                let mut diffs = Vec::with_capacity(len);
                for i in 0..len {
                    match ChatDiff::new(diffs_ptr, i) {
                        Ok(diff) => diffs.push(diff),
                        Err(e) => {
                            unsafe {
                                llama_rs_common_chat_msg_diffs_free(diffs_ptr);
                            }
                            return Err(e.into());
                        }
                    }
                }
                unsafe {
                    llama_rs_common_chat_msg_diffs_free(diffs_ptr);
                }
                Ok(diffs)
            }
            LLAMA_RS_STATUS_INVALID_ARGUMENT => return Err(ChatParserFeedError::InvalidArgument),
            LLAMA_RS_STATUS_EXCEPTION | _ => return Err(ChatParserFeedError::Exception),
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
    /// Llama.cpp returns this when it fails to create the diff view.
    #[error("Exception in Llama.cpp occured")]
    Exception,
}

/// A Safe wrapper around `common_chat_diff`.
#[derive(Debug, Clone)]
pub struct ChatDiff<'a> {
    _marker: std::marker::PhantomData<&'a ()>,
    view: *mut llama_rs_chat_msg_diff_view,
}

impl<'a> ChatDiff<'a> {
    /// Creates a new [ChatDiff].
    pub fn new(
        diff: *mut llama_rs_common_chat_msg_diffs,
        index: usize,
    ) -> Result<Self, ChatDiffCreationError> {
        let view = unsafe { llama_rs_chat_msg_diff_view_init() };
        let view_res: llama_rs_status =
            unsafe { llama_rs_chat_msg_diff_get_view(diff, index, view) };
        match view_res {
            LLAMA_RS_STATUS_OK => Ok(Self {
                _marker: std::marker::PhantomData,
                view,
            }),
            LLAMA_RS_STATUS_INVALID_ARGUMENT => {
                unsafe {
                    llama_rs_chat_msg_diff_view_free(view);
                }
                return Err(ChatDiffCreationError::InvalidArgument);
            }
            LLAMA_RS_STATUS_EXCEPTION | _ => {
                unsafe {
                    llama_rs_chat_msg_diff_view_free(view);
                }
                return Err(ChatDiffCreationError::Exception);
            }
        }
    }

    /// Gets the standard textual content delta.
    ///
    /// # Returns
    /// - `Some(Cow::Borrowed(&str))` if the `common_chat_diff` has content.
    /// - `None` if no content was generated.
    pub fn content(&self) -> Option<Cow<'a, str>> {
        Self::get_opt_string_cow(unsafe { (*self.view).content })
    }

    /// Gets the reasoning content delta.
    ///
    /// # Returns
    /// - `Some(Cow::Borrowed(&str))` if the `common_chat_diff` has reasoning content.
    /// - `None` if no reasoning content was generated.
    pub fn reasoning(&self) -> Option<Cow<'a, str>> {
        Self::get_opt_string_cow(unsafe { (*self.view).reasoning_content })
    }

    /// Gets the tool call delta.
    pub fn tool_call(&self) -> Option<LlamaChatToolCall> {
        unsafe {
            if let Some(name) = Self::get_opt_string_cow((*self.view).tool_call_name) {
                match LlamaChatToolCall::new(
                    &name,
                    &Self::get_opt_string_cow((*self.view).tool_call_arguments).unwrap_or_default(),
                    &Self::get_opt_string_cow((*self.view).tool_call_id).unwrap_or_default(),
                ) {
                    Ok(tc) => Some(tc),
                    Err(_) => None,
                }
            } else {
                None
            }
        }
    }

    fn get_opt_string_cow(ptr: *const i8) -> Option<Cow<'a, str>> {
        unsafe {
            if ptr.is_null() {
                None
            } else {
                let res: Cow<'a, str> =
                    Cow::Owned(CStr::from_ptr(ptr).to_string_lossy().into_owned());
                if res.is_empty() {
                    None
                } else {
                    Some(res)
                }
            }
        }
    }
}

impl<'a> Drop for ChatDiff<'a> {
    fn drop(&mut self) {
        unsafe {
            llama_rs_chat_msg_diff_view_free(self.view);
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ChatParamsCreationError {
    #[error("Failed to create chat params view")]
    ViewCreationFailed,
}

/// Safe wrapper around `common_chat_params`.
#[derive(Debug, Clone)]
pub struct LlamaChatParams {
    pub ptr: *mut llama_rs_common_chat_params,
    pub view: *mut llama_rs_common_chat_params_view,
}

impl LlamaChatParams {
    pub fn new(ptr: *mut llama_rs_common_chat_params) -> Result<Self, ChatParamsCreationError> {
        let view_ptr = unsafe { llama_rs_common_chat_params_view_init(ptr) };
        if view_ptr.is_null() {
            return Err(ChatParamsCreationError::ViewCreationFailed);
        }
        Ok(Self {
            ptr,
            view: view_ptr,
        })
    }

    /// Returns a string slice of the generated chat prompt.
    pub fn prompt(&self) -> &str {
        unsafe { CStr::from_ptr((*self.view).prompt) }
            .to_str()
            .unwrap_or("")
    }

    /// Returns a view of the chat params.
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
        let grammar_triggers = unsafe {
            if (*self.view).n_grammar_triggers > 0
                && !(*self.view).n_grammar_triggers < i32::MAX as usize
            {
                let triggers_slice = std::slice::from_raw_parts(
                    (*self.view).grammar_triggers,
                    (*self.view).n_grammar_triggers as usize,
                );
                triggers_slice
                    .iter()
                    .map(|t| LlamaGrammarTrigger {
                        trigger_type: t.type_.into(),
                        value: get_cstring(t.value),
                        token: LlamaToken(t.token),
                    })
                    .collect::<Vec<LlamaGrammarTrigger>>()
            } else {
                Vec::new()
            }
        };

        let message_delimiters = unsafe {
            if (*self.view).n_message_delimiters > 0
                && !(*self.view).n_message_delimiters < i32::MAX as usize
            {
                std::slice::from_raw_parts(
                    (*self.view).message_delimiters,
                    (*self.view).n_message_delimiters as usize,
                )
                .iter()
                .map(|md| LlamaChatMsgDelimiter {
                    role: md.role.into(),
                    delimiter: get_cstring(md.delimiter),
                    tokens: std::slice::from_raw_parts((*md).tokens, (*md).n_tokens)
                        .iter()
                        .map(|t| LlamaToken(*t))
                        .collect::<Vec<LlamaToken>>(),
                })
                .collect::<Vec<LlamaChatMsgDelimiter>>()
            } else {
                Vec::new()
            }
        };
        let preserved_tokens = unsafe {
            if (*self.view).n_preserved_tokens > 0
                && !(*self.view).n_preserved_tokens > i32::MAX as usize
            {
                std::slice::from_raw_parts(
                    (*self.view).preserved_tokens,
                    (*self.view).n_preserved_tokens as usize,
                )
                .iter()
                .map(|t| get_cstring(*t))
                .collect::<Vec<CString>>()
            } else {
                Vec::new()
            }
        };
        let additional_stops = unsafe {
            if (*self.view).n_additional_stops > 0
                && !(*self.view).n_additional_stops > i32::MAX as usize
            {
                std::slice::from_raw_parts(
                    (*self.view).additional_stops,
                    (*self.view).n_additional_stops as usize,
                )
                .iter()
                .map(|t| get_cstring(*t))
                .collect::<Vec<CString>>()
            } else {
                Vec::new()
            }
        };
        unsafe {
            LlamaChatParamsView {
                format: (*self.view).format.into(),
                prompt: get_cstring((*self.view).prompt),
                grammar: get_cstring((*self.view).grammar),
                grammar_lazy: (*self.view).grammar_lazy,
                generation_prompt: get_cstring((*self.view).generation_prompt),
                supports_thinking: (*self.view).supports_thinking,
                thinking_start_tag: get_cstring((*self.view).thinking_start_tag),
                thinking_end_tag: get_cstring((*self.view).thinking_end_tag),
                grammar_triggers,
                preserved_tokens,
                additional_stops,
                parser: get_cstring((*self.view).parser),
                message_delimiters,
            }
        }
    }
}

impl Drop for LlamaChatParams {
    fn drop(&mut self) {
        unsafe {
            llama_rs_common_chat_params_free(self.ptr);
            llama_rs_common_chat_params_view_free(self.view)
        }
    }
}

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

impl From<llama_rs_common_chat_format> for LlamaChatFormat {
    fn from(value: llama_rs_common_chat_format) -> Self {
        match value {
            LLAMA_RS_COMMON_CHAT_FORMAT_CONTENT_ONLY => Self::ContentOnly,
            LLAMA_RS_COMMON_CHAT_FORMAT_PEG_SIMPLE => Self::PegSimple,
            LLAMA_RS_COMMON_CHAT_FORMAT_PEG_NATIVE => Self::PegNative,
            LLAMA_RS_COMMON_CHAT_FORMAT_PEG_GEMMA4 => Self::PegGemma4,
            LLAMA_RS_COMMON_CHAT_FORMAT_COUNT => Self::Count,
            _ => Self::default(),
        }
    }
}

/// Enum for `common_grammar_trigger_type`
#[derive(Debug, Clone, Copy, Default)]
pub enum LlamaGrammarTriggerType {
    #[default]
    /// Trigger grammar at a token boundary.
    Token,
    /// Trigger grammar at a word boundary.
    Word,
    /// Trigger grammar at the start of a pattern.
    Pattern,
    /// Trigger grammar at the end of a pattern.
    PatternFull,
}

impl From<llama_rs_common_grammar_trigger_type> for LlamaGrammarTriggerType {
    fn from(value: llama_rs_common_grammar_trigger_type) -> Self {
        match value {
            LLAMA_RS_COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN => Self::Token,
            LLAMA_RS_COMMON_GRAMMAR_TRIGGER_TYPE_WORD => Self::Word,
            LLAMA_RS_COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN => Self::Pattern,
            LLAMA_RS_COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL => Self::PatternFull,
            _ => Self::default(),
        }
    }
}

/// Safe struct for `common_grammar_trigger`
#[derive(Debug, Clone)]
pub struct LlamaGrammarTrigger {
    /// The type of grammar trigger.
    pub trigger_type: LlamaGrammarTriggerType,
    /// The value of the grammar trigger.
    pub value: CString,
    /// The token that triggers the grammar.
    pub token: LlamaToken,
}
/// Enum for `common_chat_role`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LlamaChatRole {
    #[default]
    UNKNOWN,
    SYSTEM,
    ASSISTANT,
    USER,
    TOOL,
}

impl From<llama_rs_common_chat_role> for LlamaChatRole {
    fn from(value: llama_rs_common_chat_role) -> Self {
        match value {
            LLAMA_RS_COMMON_CHAT_ROLE_UNKNOWN => Self::UNKNOWN,
            LLAMA_RS_COMMON_CHAT_ROLE_SYSTEM => Self::SYSTEM,
            LLAMA_RS_COMMON_CHAT_ROLE_ASSISTANT => Self::ASSISTANT,
            LLAMA_RS_COMMON_CHAT_ROLE_USER => Self::USER,
            LLAMA_RS_COMMON_CHAT_ROLE_TOOL => Self::TOOL,
            _ => Self::UNKNOWN,
        }
    }
}

/// Safe struct for `common_chat_msg_delimiter`
#[derive(Debug, Clone)]
pub struct LlamaChatMsgDelimiter {
    pub role: LlamaChatRole,
    pub delimiter: CString,
    pub tokens: Vec<LlamaToken>,
}

#[derive(Debug, Clone)]
pub struct LlamaChatParamsView {
    pub format: LlamaChatFormat,
    pub prompt: CString,
    pub grammar: CString,
    pub grammar_lazy: bool,
    pub generation_prompt: CString,
    pub supports_thinking: bool,
    /// " e.g., \"<think>\""
    pub thinking_start_tag: CString,
    /// e.g., \"</think>\""
    pub thinking_end_tag: CString,
    pub grammar_triggers: Vec<LlamaGrammarTrigger>,
    pub preserved_tokens: Vec<CString>,
    pub additional_stops: Vec<CString>,
    pub parser: CString,
    pub message_delimiters: Vec<LlamaChatMsgDelimiter>,
}

/// Convenience struct which gets converted to `generation_params`.
/// See `generation_params` for defaults: https://github.com/ggml-org/llama.cpp/blob/32beb244f5c2ca91c583be15d4671643b54ba238/common/chat-auto-parser.h#L54
#[derive(Debug, Clone)]
pub struct LlamaGenerationParams {
    /// Message history in order.
    pub messages: Vec<LlamaChatMessageFull>,
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
    pub extra_context: Option<String>,

    /// Stringified JSON schema for constrained output
    /// - If grammar is set, this will be ignored.
    pub json_schema: Option<String>,
    /// Grammar to use for constrained output
    /// - If this is set, `json_schema` will be ignored.
    pub grammar: Option<String>,

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

impl LlamaGenerationParams {
    /// Use this to create a builder for `LlamaGenerationParams`.
    pub fn builder() -> LlamaGenerationParamsBuilder {
        LlamaGenerationParamsBuilder {
            messages: Vec::<LlamaChatMessageFull>::default(),
            tools: None,
            add_generation_prompt: false,
            enable_thinking: true,
            reasoning_format: None,
            continue_final_message: None,
            extra_context: None,
            json_schema: None,
            grammar: None,
            parallel_tool_calls: true,
            add_bos: false,
            add_eos: false,
        }
    }
}

#[derive(Debug)]
pub struct LlamaGenerationParamsState {
    pub params: llama_cpp_sys_2::llama_rs_chat_template_generation_params,
    _msgs_tool_calls: Vec<Vec<llama_cpp_sys_2::llama_rs_chat_tool_call>>,
    _msgs: Vec<llama_cpp_sys_2::llama_rs_chat_message>,
    _tools: Vec<llama_cpp_sys_2::llama_rs_chat_tool>,
    _extra_context: Option<CString>,
    _json_schema: Option<CString>,
    _grammar: Option<CString>,
}

impl LlamaGenerationParams {
    /// Creates a state object holding [llama_rs_chat_template_generation_params] and its owned strings/buffers.
    pub fn into_state(&self) -> Result<LlamaGenerationParamsState, std::ffi::NulError> {
        let mut msgs_tool_calls = Vec::new();
        let msgs = self
            .messages
            .iter()
            .map(|c| {
                let tool_calls = c
                    .tool_calls
                    .iter()
                    .map(|tc| llama_cpp_sys_2::llama_rs_chat_tool_call {
                        name: tc.name.as_ptr(),
                        id: tc.id.as_ptr(),
                        arguments: tc.arguments.as_ptr(),
                    })
                    .collect::<Vec<llama_cpp_sys_2::llama_rs_chat_tool_call>>();
                let tool_calls_ptr = tool_calls.as_ptr();
                msgs_tool_calls.push(tool_calls);
                llama_cpp_sys_2::llama_rs_chat_message {
                    role: c.role.as_ptr(),
                    content: c.content.as_ptr(),
                    reasoning_content: if c.reasoning_content.is_empty() {
                        ptr::null_mut()
                    } else {
                        c.reasoning_content.as_ptr()
                    },
                    tool_name: if c.tool_name.is_empty() {
                        ptr::null_mut()
                    } else {
                        c.tool_name.as_ptr()
                    },
                    tool_call_id: if c.tool_call_id.is_empty() {
                        ptr::null_mut()
                    } else {
                        c.tool_call_id.as_ptr()
                    },
                    tool_calls: tool_calls_ptr,
                    n_tool_calls: c.tool_calls.len(),
                }
            })
            .collect::<Vec<llama_cpp_sys_2::llama_rs_chat_message>>();
        let tools = self
            .tools
            .iter()
            .map(|t| llama_cpp_sys_2::llama_rs_chat_tool {
                name: t.name.as_ptr(),
                description: t.description.as_ptr(),
                parameters: t.parameters.as_ptr(),
            })
            .collect::<Vec<llama_cpp_sys_2::llama_rs_chat_tool>>();

        let mut params = llama_cpp_sys_2::llama_rs_chat_template_generation_params {
            messages: msgs.as_ptr(),
            n_messages: msgs.len(),
            tools: tools.as_ptr(),
            n_tools: tools.len(),
            add_generation_prompt: self.add_generation_prompt,
            enable_thinking: self.enable_thinking,
            reasoning_format: self.reasoning_format.into(),
            continue_final_message: self.continue_final_message.into(),
            extra_context: ptr::null(),
            json_schema: ptr::null(),
            grammar: ptr::null(),
            parallel_tool_calls: self.parallel_tool_calls,
            add_bos: self.add_bos,
            add_eos: self.add_eos,
        };

        let extra_context = match &self.extra_context {
            Some(ctx) => Some(CString::new(ctx.as_bytes())?),
            None => None,
        };
        let json_schema = match &self.json_schema {
            Some(js) => Some(CString::new(js.as_bytes())?),
            None => None,
        };
        let grammar = match &self.grammar {
            Some(grm) => Some(CString::new(grm.as_bytes())?),
            None => None,
        };

        params.extra_context = extra_context.as_ref().map_or(ptr::null(), |c| c.as_ptr());
        params.json_schema = json_schema.as_ref().map_or(ptr::null(), |c| c.as_ptr());
        params.grammar = grammar.as_ref().map_or(ptr::null(), |c| c.as_ptr());

        Ok(LlamaGenerationParamsState {
            params,
            _msgs_tool_calls: msgs_tool_calls,
            _msgs: msgs,
            _tools: tools,
            _extra_context: extra_context,
            _json_schema: json_schema,
            _grammar: grammar,
        })
    }
}

impl Default for LlamaGenerationParams {
    fn default() -> Self {
        Self {
            messages: Vec::<LlamaChatMessageFull>::default(),
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

/// Builder for `LlamaGenerationParams`.
#[derive(Debug, Clone)]
pub struct LlamaGenerationParamsBuilder {
    /// Message history in order.
    pub messages: Vec<LlamaChatMessageFull>,
    /// Tools to be aware of.
    pub tools: Option<Vec<LlamaChatTool>>,
    /// Add generation prompt to the prompt.
    /// - Defaults to `false`
    pub add_generation_prompt: bool,
    /// Enable thinking.
    /// - Defaults to `true`.
    pub enable_thinking: bool,

    /// Reasoning format.
    /// - Defaults to [LlamaReasoningFormat::AUTO].
    pub reasoning_format: Option<LlamaReasoningFormat>,

    /// Chat continuation.
    /// - Defaults to [LlamaChatContinuation::NONE].
    pub continue_final_message: Option<LlamaChatContinuation>,

    /// Stringified JSON object for Jinja kwargs
    pub extra_context: Option<String>,

    /// Stringified JSON schema for constrained output
    /// - If grammar is set, this will be ignored.
    pub json_schema: Option<String>,
    /// Grammar to use for constrained output
    /// - If this is set, `json_schema` will be ignored.
    pub grammar: Option<String>,

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

impl LlamaGenerationParamsBuilder {
    /// Set the message history to use for this generation.
    pub fn with_messages(mut self, messages: &[LlamaChatMessageFull]) -> Self {
        self.messages = messages.to_vec();
        self
    }

    /// Set the tools available to the model.
    pub fn with_tools(mut self, tools: &[LlamaChatTool]) -> Self {
        self.tools = Some(tools.to_vec());
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
        self.reasoning_format = Some(reasoning_format);
        self
    }

    /// Set the continuation type.
    ///
    /// This is used by the [ChatParser] to determine the proper routing of tokens following
    /// an interrupted generation.
    pub fn with_continue_final_message(mut self, continuation: LlamaChatContinuation) -> Self {
        self.continue_final_message = Some(continuation);
        self
    }

    /// Set the extra context to use for this generation.
    /// - Defaults to `None`
    pub fn with_extra_context(mut self, extra_context: &str) -> Self {
        self.extra_context = Some(extra_context.to_string());
        self
    }

    /// Set the JSON schema to use for constrained output.
    /// - If grammar is set, this will be ignored.
    /// - Defaults to `None`
    pub fn with_json_schema(mut self, json_schema: &str) -> Self {
        self.json_schema = Some(json_schema.to_string());
        self
    }

    /// Set the grammar to use for constrained output.
    /// - If this is set, `json_schema` will be ignored.
    /// - Defaults to `None`
    pub fn with_grammar(mut self, grammar: &str) -> Self {
        self.grammar = Some(grammar.to_string());
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

    /// Build the `LlamaGenerationParams`.
    pub fn build(self) -> LlamaGenerationParams {
        let reasoning_format = if let Some(rf) = self.reasoning_format {
            rf
        } else {
            if self.enable_thinking {
                LlamaReasoningFormat::AUTO
            } else {
                LlamaReasoningFormat::NONE
            }
        };

        LlamaGenerationParams {
            messages: self.messages,
            tools: self.tools.unwrap_or_default(),
            add_generation_prompt: self.add_generation_prompt,
            enable_thinking: self.enable_thinking,
            reasoning_format,
            continue_final_message: self.continue_final_message.unwrap_or_default(),
            extra_context: self.extra_context,
            json_schema: self.json_schema,
            grammar: self.grammar,
            parallel_tool_calls: self.parallel_tool_calls,
            add_bos: self.add_bos,
            add_eos: self.add_eos,
        }
    }
}

/// Chat continuation method provided via `with_continue_final_message`. Only used by [ChatParser].
#[derive(Debug, Clone, Copy, Default)]
pub enum LlamaChatContinuation {
    #[default]
    NONE,
    AUTO,
    REASONING,
    CONTENT,
}

impl From<llama_rs_common_chat_continuation> for LlamaChatContinuation {
    fn from(value: llama_rs_common_chat_continuation) -> Self {
        match value {
            LLAMA_RS_COMMON_CHAT_CONTINUATION_NONE => Self::NONE,
            LLAMA_RS_COMMON_CHAT_CONTINUATION_AUTO => Self::AUTO,
            LLAMA_RS_COMMON_CHAT_CONTINUATION_REASONING => Self::REASONING,
            LLAMA_RS_COMMON_CHAT_CONTINUATION_CONTENT => Self::CONTENT,
            _ => Self::default(),
        }
    }
}

impl Into<llama_rs_common_chat_continuation> for LlamaChatContinuation {
    fn into(self) -> llama_rs_common_chat_continuation {
        match self {
            Self::NONE => LLAMA_RS_COMMON_CHAT_CONTINUATION_NONE,
            Self::AUTO => LLAMA_RS_COMMON_CHAT_CONTINUATION_AUTO,
            Self::REASONING => LLAMA_RS_COMMON_CHAT_CONTINUATION_REASONING,
            Self::CONTENT => LLAMA_RS_COMMON_CHAT_CONTINUATION_CONTENT,
        }
    }
}

/// Reasoning API response format (not to be confused as chat template's
/// reasoning format) only used by [ChatParser].
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, Default)]
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

impl From<llama_rs_common_reasoning_format> for LlamaReasoningFormat {
    fn from(value: llama_rs_common_reasoning_format) -> Self {
        match value {
            LLAMA_RS_COMMON_REASONING_FORMAT_NONE => Self::NONE,
            LLAMA_RS_COMMON_REASONING_FORMAT_AUTO => Self::AUTO,
            LLAMA_RS_COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY => Self::DEEPSEEK_LEGACY,
            LLAMA_RS_COMMON_REASONING_FORMAT_DEEPSEEK => Self::DEEPSEEK,
            _ => Self::default(),
        }
    }
}

impl Into<llama_rs_common_reasoning_format> for LlamaReasoningFormat {
    fn into(self) -> llama_rs_common_reasoning_format {
        match self {
            Self::NONE => LLAMA_RS_COMMON_REASONING_FORMAT_NONE,
            Self::AUTO => LLAMA_RS_COMMON_REASONING_FORMAT_AUTO,
            Self::DEEPSEEK_LEGACY => LLAMA_RS_COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY,
            Self::DEEPSEEK => LLAMA_RS_COMMON_REASONING_FORMAT_DEEPSEEK,
        }
    }
}
