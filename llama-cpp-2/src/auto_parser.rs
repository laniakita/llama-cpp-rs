//! Provides a wrapper for the chat template autoparser.
//!
//! It also provides:
//! - Supporting structs for the template analysis.
//! - High-level structs for the generation params.
//!
use std::{ffi::CStr, ptr::null_mut};

use llama_cpp_sys_2::{
    llama_rs_analyze_content, llama_rs_analyze_reasoning, llama_rs_analyze_tools,
    llama_rs_autoparser, llama_rs_autoparser_analyze_template, llama_rs_autoparser_free,
    llama_rs_autoparser_init, llama_rs_call_id_position, llama_rs_common_chat_format,
    llama_rs_common_chat_template_free, llama_rs_common_chat_template_init,
    llama_rs_common_grammar_trigger_type, llama_rs_content_mode, llama_rs_reasoning_mode,
    llama_rs_template_analysis, llama_rs_template_analysis_free, llama_rs_tool_arguments_analysis,
    llama_rs_tool_format, llama_rs_tool_format_analysis, llama_rs_tool_function_analysis,
    llama_rs_tool_id_analysis, LLAMA_RS_CALL_ID_POSITION_BETWEEN_FUNC_AND_ARGS,
    LLAMA_RS_CALL_ID_POSITION_NONE, LLAMA_RS_CALL_ID_POSITION_POST_ARGS,
    LLAMA_RS_CALL_ID_POSITION_PRE_FUNC_NAME, LLAMA_RS_COMMON_CHAT_FORMAT_CONTENT_ONLY,
    LLAMA_RS_COMMON_CHAT_FORMAT_PEG_GEMMA4, LLAMA_RS_COMMON_CHAT_FORMAT_PEG_NATIVE,
    LLAMA_RS_COMMON_CHAT_FORMAT_PEG_SIMPLE, LLAMA_RS_COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN,
    LLAMA_RS_COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL, LLAMA_RS_COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN,
    LLAMA_RS_COMMON_GRAMMAR_TRIGGER_TYPE_WORD, LLAMA_RS_CONTENT_MODE_ALWAYS_WRAPPED,
    LLAMA_RS_CONTENT_MODE_PLAIN, LLAMA_RS_CONTENT_MODE_WRAPPED_WITH_REASONING,
    LLAMA_RS_REASONING_MODE_NONE, LLAMA_RS_REASONING_MODE_TAG_BASED,
    LLAMA_RS_REASONING_MODE_TOOLS_ONLY, LLAMA_RS_STATUS_INVALID_ARGUMENT,
    LLAMA_RS_TOOL_FORMAT_JSON_NATIVE, LLAMA_RS_TOOL_FORMAT_NONE,
    LLAMA_RS_TOOL_FORMAT_TAG_WITH_JSON, LLAMA_RS_TOOL_FORMAT_TAG_WITH_TAGGED,
};

use crate::{
    model::{LlamaChatMessage, LlamaChatTemplate, LlamaChatTool, LlamaModel},
    token::LlamaToken,
    AnalyzeTemplateError::{self},
    NewAutoParserError,
};

/// Auto parser for chat templates.
#[derive(Debug)]
pub struct AutoParser {
    ptr: *mut llama_rs_autoparser,
}

impl Drop for AutoParser {
    fn drop(&mut self) {
        unsafe {
            llama_rs_autoparser_free(self.ptr);
        }
    }
}

impl AutoParser {
    /// Creates a new AutoParser.
    ///
    /// ### Errors
    ///
    /// - `NewAutoParserError::NullResult` - if the AutoParser could not be created.
    pub fn new() -> Result<Self, NewAutoParserError> {
        let ptr = unsafe { llama_rs_autoparser_init() };
        if ptr.is_null() {
            return Err(NewAutoParserError::NullResult);
        }
        Ok(Self { ptr })
    }

    /// Analyzes a chat template.
    pub fn analyze_template(
        &self,
        model: &LlamaModel,
        template: &LlamaChatTemplate,
    ) -> Result<LlamaChatTemplateAnalysis, AnalyzeTemplateError> {
        let mut analysis = llama_rs_template_analysis {
            reasoning: llama_rs_analyze_reasoning {
                mode: LLAMA_RS_REASONING_MODE_NONE,
                start: null_mut(),
                end: null_mut(),
            },
            content: llama_rs_analyze_content {
                mode: LLAMA_RS_CONTENT_MODE_PLAIN,
                start: null_mut(),
                end: null_mut(),
                requires_nonnull_content: false,
            },
            tools: llama_rs_analyze_tools {
                format: llama_rs_tool_format_analysis {
                    mode: LLAMA_RS_TOOL_FORMAT_NONE,
                    section_start: null_mut(),
                    section_end: null_mut(),
                    per_call_start: null_mut(),
                    per_call_end: null_mut(),
                    fun_name_is_key: false,
                    tools_array_wrapped: false,
                    function_field: null_mut(),
                    name_field: null_mut(),
                    args_field: null_mut(),
                    id_field: null_mut(),
                    gen_id_field: null_mut(),
                    parameter_order: null_mut(),
                },
                function: llama_rs_tool_function_analysis {
                    name_prefix: null_mut(),
                    name_suffix: null_mut(),
                    close: null_mut(),
                },
                arguments: llama_rs_tool_arguments_analysis {
                    start: null_mut(),
                    end: null_mut(),
                    name_prefix: null_mut(),
                    name_suffix: null_mut(),
                    value_prefix: null_mut(),
                    value_suffix: null_mut(),
                    separator: null_mut(),
                },
                call_id: llama_rs_tool_id_analysis {
                    pos: LLAMA_RS_CALL_ID_POSITION_NONE,
                    prefix: null_mut(),
                    suffix: null_mut(),
                },
            },
        };

        let common_template = unsafe { template.to_common_chat_template(model)? };

        let res = unsafe {
            llama_rs_autoparser_analyze_template(self.ptr, common_template, &mut analysis)
        };

        if res < 0 {
            match res {
                LLAMA_RS_STATUS_INVALID_ARGUMENT => {
                    return Err(AnalyzeTemplateError::InvalidTemplate(template.clone()));
                }
                _ => {
                    return Err(AnalyzeTemplateError::ExceptionOccured);
                }
            }
        }
        let template_analysis: LlamaChatTemplateAnalysis = analysis.into();

        unsafe {
            llama_rs_template_analysis_free(&mut analysis);
            llama_rs_common_chat_template_free(common_template);
        }

        Ok(template_analysis)
    }
}

/// Chat template analysis.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct LlamaChatTemplateAnalysis {
    /// Reasoning analysis.
    pub reasoning: LlamaReasoningAnalysis,
    /// Content analysis.
    pub content: LlamaContentAnalysis,
    /// Tools analysis.
    pub tools: LlamaToolCallsAnalysis,
}

impl From<llama_rs_template_analysis> for LlamaChatTemplateAnalysis {
    fn from(analysis: llama_rs_template_analysis) -> Self {
        let get_option_string = |ptr: *mut i8| -> Option<String> {
            unsafe {
                if ptr.is_null() {
                    None
                } else {
                    Some(CStr::from_ptr(ptr).to_string_lossy().to_string())
                }
            }
        };
        let get_vec_from_ptr = |mut ptr: *mut *mut i8| -> Vec<String> {
            unsafe {
                if ptr.is_null() {
                    return Vec::new();
                }
                let mut res = Vec::new();
                while !(*ptr).is_null() {
                    let str_ptr = *ptr;
                    res.push(CStr::from_ptr(str_ptr).to_string_lossy().to_string());
                    ptr = ptr.add(1);
                }
                res
            }
        };

        let reasoning = LlamaReasoningAnalysis {
            mode: analysis.reasoning.mode.into(),
            start: get_option_string(analysis.reasoning.start),
            end: get_option_string(analysis.reasoning.end),
        };
        let content = LlamaContentAnalysis {
            mode: analysis.content.mode.into(),
            start: get_option_string(analysis.content.start),
            end: get_option_string(analysis.content.end),
            requires_nonnull_content: analysis.content.requires_nonnull_content,
        };
        let tools = LlamaToolCallsAnalysis {
            format: LlamaToolFormatAnalysis {
                mode: analysis.tools.format.mode.into(),
                section_start: get_option_string(analysis.tools.format.section_start),
                section_end: get_option_string(analysis.tools.format.section_end),
                per_call_start: get_option_string(analysis.tools.format.per_call_start),
                per_call_end: get_option_string(analysis.tools.format.per_call_end),
                fun_name_is_key: analysis.tools.format.fun_name_is_key,
                tools_array_wrapped: analysis.tools.format.tools_array_wrapped,
                function_field: unsafe {
                    CStr::from_ptr(analysis.tools.format.function_field)
                        .to_string_lossy()
                        .to_string()
                },
                name_field: unsafe {
                    CStr::from_ptr(analysis.tools.format.name_field)
                        .to_string_lossy()
                        .to_string()
                },
                args_field: unsafe {
                    CStr::from_ptr(analysis.tools.format.args_field)
                        .to_string_lossy()
                        .to_string()
                },
                id_field: get_option_string(analysis.tools.format.id_field),
                gen_id_field: get_option_string(analysis.tools.format.gen_id_field),
                parameter_order: get_vec_from_ptr(analysis.tools.format.parameter_order),
            },
            function: LlamaToolFunctionAnalysis {
                name_prefix: get_option_string(analysis.tools.function.name_prefix),
                name_suffix: get_option_string(analysis.tools.function.name_suffix),
                close: get_option_string(analysis.tools.function.close),
            },
            arguments: LlamaToolArgumentsAnalysis {
                start: get_option_string(analysis.tools.arguments.start),
                end: get_option_string(analysis.tools.arguments.end),
                name_prefix: get_option_string(analysis.tools.arguments.name_prefix),
                name_suffix: get_option_string(analysis.tools.arguments.name_suffix),
                value_prefix: get_option_string(analysis.tools.arguments.value_prefix),
                value_suffix: get_option_string(analysis.tools.arguments.value_suffix),
                separator: get_option_string(analysis.tools.arguments.separator),
            },
            call_id: LlamaToolIdAnalysis {
                pos: analysis.tools.call_id.pos.into(),
                prefix: get_option_string(analysis.tools.call_id.prefix),
                suffix: get_option_string(analysis.tools.call_id.suffix),
            },
        };

        Self {
            reasoning,
            content,
            tools,
        }
    }
}

// ============================================================================
// Reasoning, Content, and Tool analyzer structs
// ============================================================================

/// Reasoning analysis.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct LlamaReasoningAnalysis {
    /// Reasoning mode.
    pub mode: LLamaReasoningMode,
    /// Start marker.
    pub start: Option<String>,
    /// End marker.
    pub end: Option<String>,
}

/// Content analysis.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct LlamaContentAnalysis {
    /// Content mode.
    pub mode: LlamaContentMode,
    /// e.g., "<response>", ">>>all\n", ""
    pub start: Option<String>,
    /// e.g., "</response>", ""
    pub end: Option<String>,
    /// Whether the content must not be empty.
    pub requires_nonnull_content: bool,
}

/// Tools analysis.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct LlamaToolCallsAnalysis {
    /// Tool format analysis.
    pub format: LlamaToolFormatAnalysis,
    /// Tool function analysis.
    pub function: LlamaToolFunctionAnalysis,
    /// Tool arguments analysis.
    pub arguments: LlamaToolArgumentsAnalysis,
    /// Tool call ID analysis.
    pub call_id: LlamaToolIdAnalysis,
}

// ============================================================================
// Sub-structs for tool analysis
// ============================================================================

/// Tool format analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct LlamaToolFormatAnalysis {
    /// Tool format mode.
    pub mode: LlamaToolFormat,
    /// e.g., "<tool_call>", "[TOOL_CALLS]", ""
    pub section_start: Option<String>,
    /// e.g., "</tool_call>", ""
    pub section_end: Option<String>,
    /// e.g., "<|tool_call_begin|>", "" (for multi-call templates)
    pub per_call_start: Option<String>,
    /// e.g., "<|tool_call_end|>", ""
    pub per_call_end: Option<String>,

    /// In JSON format function name is JSON key, i.e. { "<funname>": { ...
    /// arguments ... } }
    pub fun_name_is_key: bool,
    /// Tool calls wrapped in JSON array [...]
    pub tools_array_wrapped: bool,

    /// Function field name.
    pub function_field: String,
    /// Name field name.
    pub name_field: String,
    /// Arguments field name.
    pub args_field: String,
    /// ID field name.
    pub id_field: Option<String>,
    /// Generated ID field name.
    pub gen_id_field: Option<String>,
    /// Parameter order.
    pub parameter_order: Vec<String>,
}
impl Default for LlamaToolFormatAnalysis {
    fn default() -> Self {
        Self {
            mode: LlamaToolFormat::None,
            section_start: None,
            section_end: None,
            per_call_start: None,
            per_call_end: None,
            fun_name_is_key: false,
            tools_array_wrapped: false,
            function_field: "function".to_string(),
            name_field: "name".to_string(),
            args_field: "arguments".to_string(),
            id_field: None,
            gen_id_field: None,
            parameter_order: Vec::new(),
        }
    }
}

/// Tool function analysis.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct LlamaToolFunctionAnalysis {
    /// e.g., "<function=", "\"name\": \"", "functions."
    pub name_prefix: Option<String>,
    /// e.g., ">", "\"", ":0"
    pub name_suffix: Option<String>,
    /// e.g., "</function>", "" (for tag-based)
    pub close: Option<String>,
}

/// Tool arguments analysis
#[derive(Debug, Clone, PartialEq, Default)]
pub struct LlamaToolArgumentsAnalysis {
    /// e.g., "<|tool_call_argument_begin|>", "<args>"
    pub start: Option<String>,
    /// e.g., "<|tool_call_argument_end|>", "</args>"
    pub end: Option<String>,
    /// e.g., "<param=", "<arg_key>", "\""
    pub name_prefix: Option<String>,
    /// e.g., ">", "</arg_key>", "\":"
    pub name_suffix: Option<String>,
    /// e.g., "", "<arg_value>", ""
    pub value_prefix: Option<String>,
    /// e.g., "</param>", "</arg_value>", ""
    pub value_suffix: Option<String>,
    /// e.g., "", "\n", ","
    pub separator: Option<String>,
}

/// Tool Id analysis.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct LlamaToolIdAnalysis {
    /// Tool id position.
    pub pos: LlamaCallIdPosition,
    /// e.g., "[CALL_ID]" (marker before call ID value)
    pub prefix: Option<String>,
    /// e.g., "" (marker after call ID value, before next section)
    pub suffix: Option<String>,
}

// ============================================================================
// Analysis Result Enums
// ============================================================================

/// Reasoning mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LLamaReasoningMode {
    #[default]
    /// No reasoning markers detected
    None = 0,
    /// Tag-based: Tag-based: <think>...</think> (start can be empty for
    /// delimiter-style)
    TagBased = 1,
    /// Only reason on tool calls, not on normal content
    ToolsOnly = 2,
}

impl From<llama_rs_reasoning_mode> for LLamaReasoningMode {
    fn from(value: llama_rs_reasoning_mode) -> Self {
        match value {
            LLAMA_RS_REASONING_MODE_NONE => Self::None,
            LLAMA_RS_REASONING_MODE_TAG_BASED => Self::TagBased,
            LLAMA_RS_REASONING_MODE_TOOLS_ONLY => Self::ToolsOnly,
            _ => Self::default(),
        }
    }
}

/// Content mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LlamaContentMode {
    #[default]
    /// No content markers
    Plain = 0,
    /// Content always wrapped with markers
    AlwaysWrapped = 1,
    /// Content wrapped only when reasoning present
    WrappedWithReasoning = 2,
}

impl From<llama_rs_content_mode> for LlamaContentMode {
    fn from(value: llama_rs_content_mode) -> Self {
        match value {
            LLAMA_RS_CONTENT_MODE_PLAIN => Self::Plain,
            LLAMA_RS_CONTENT_MODE_ALWAYS_WRAPPED => Self::AlwaysWrapped,
            LLAMA_RS_CONTENT_MODE_WRAPPED_WITH_REASONING => Self::WrappedWithReasoning,
            _ => Self::default(),
        }
    }
}

/// Call ID position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LlamaCallIdPosition {
    #[default]
    /// No call ID support detected
    PositionNone = 0,
    /// Call ID before function name: [CALL_ID]id[FUNC]name{args}
    PositionPreFuncName = 1,
    /// Call ID between function and args: [FUNC]name[CALL_ID]id{args}
    PositionBetweenFuncAndArgs = 2,
    /// Call ID after arguments: [FUNC]name{args}[CALL_ID]id
    PositionPostArgs = 3,
}

impl From<llama_rs_call_id_position> for LlamaCallIdPosition {
    fn from(value: llama_rs_call_id_position) -> Self {
        match value {
            LLAMA_RS_CALL_ID_POSITION_NONE => Self::PositionNone,
            LLAMA_RS_CALL_ID_POSITION_PRE_FUNC_NAME => Self::PositionPreFuncName,
            LLAMA_RS_CALL_ID_POSITION_BETWEEN_FUNC_AND_ARGS => Self::PositionBetweenFuncAndArgs,
            LLAMA_RS_CALL_ID_POSITION_POST_ARGS => Self::PositionPostArgs,
            _ => Self::default(),
        }
    }
}

/// Tool call format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LlamaToolFormat {
    #[default]
    /// No tool support detected
    None = 0,
    /// Pure JSON: {"name": "X", "arguments": {...}}
    JSONNative = 1,
    /// Tag-based with JSON args: <function=X>{...}</function>
    TagWithJSON = 2,
    /// Tag-based with tagged args: <param=key>value</param>
    TagWithTagged = 3,
}

impl From<llama_rs_tool_format> for LlamaToolFormat {
    fn from(value: llama_rs_tool_format) -> Self {
        match value {
            LLAMA_RS_TOOL_FORMAT_NONE => Self::None,
            LLAMA_RS_TOOL_FORMAT_JSON_NATIVE => Self::JSONNative,
            LLAMA_RS_TOOL_FORMAT_TAG_WITH_JSON => Self::TagWithJSON,
            LLAMA_RS_TOOL_FORMAT_TAG_WITH_TAGGED => Self::TagWithTagged,
            _ => Self::default(),
        }
    }
}

// ============================================================================
// Chat params
// ============================================================================

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
            messages: Vec::<LlamaChatMessage>::default(),
            tools: None,
            add_generation_prompt: false,
            enable_thinking: true,
            extra_context: None,
            json_schema: None,
            grammar: None,
            parallel_tool_calls: true,
            add_bos: false,
            add_eos: false,
        }
    }
}

impl Default for LlamaGenerationParams {
    fn default() -> Self {
        Self {
            messages: Vec::<LlamaChatMessage>::default(),
            tools: Vec::<LlamaChatTool>::default(),
            add_generation_prompt: false,
            enable_thinking: true,
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
    pub messages: Vec<LlamaChatMessage>,
    /// Tools to be aware of.
    pub tools: Option<Vec<LlamaChatTool>>,
    /// Add generation prompt to the prompt.
    /// - Defaults to `false`
    pub add_generation_prompt: bool,
    /// Enable thinking.
    /// - Defaults to `true`.
    pub enable_thinking: bool,

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
    pub fn with_messages(mut self, messages: &[LlamaChatMessage]) -> Self {
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

    /// Set the extra context to use for this generation.
    /// - Defaults to `None`
    pub fn with_extra_context(mut self, extra_context: String) -> Self {
        self.extra_context = Some(extra_context);
        self
    }

    /// Set the JSON schema to use for constrained output.
    /// - If grammar is set, this will be ignored.
    /// - Defaults to `None`
    pub fn with_json_schema(mut self, json_schema: String) -> Self {
        self.json_schema = Some(json_schema);
        self
    }

    /// Set the grammar to use for constrained output.
    /// - If this is set, `json_schema` will be ignored.
    /// - Defaults to `None`
    pub fn with_grammar(mut self, grammar: String) -> Self {
        self.grammar = Some(grammar);
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
        LlamaGenerationParams {
            messages: self.messages,
            tools: self.tools.unwrap_or_default(),
            add_generation_prompt: self.add_generation_prompt,
            enable_thinking: self.enable_thinking,
            extra_context: self.extra_context,
            json_schema: self.json_schema,
            grammar: self.grammar,
            parallel_tool_calls: self.parallel_tool_calls,
            add_bos: self.add_bos,
            add_eos: self.add_eos,
        }
    }
}

/// Safe struct for `common_chat_params`.
#[derive(Debug, Clone)]
pub struct LlamaChatParams {
    /// Chat format.
    pub common_chat_format: LlamaChatFormat,
    /// Formatted prompt.
    pub prompt: String,
    /// Grammar.
    pub grammar: Option<String>,
    /// Grammar lazy evaluation.
    pub grammar_lazy: bool,
    /// Generation prompt.
    pub generation_prompt: String,
    /// Whether the model supports thinking.
    pub supports_thinking: bool,
    /// The tag that marks the beginning of thinking.
    pub thinking_start_tag: Option<String>, // e.g., "<think>"
    /// The tag that marks the end of thinking.
    pub thinking_end_tag: Option<String>, // e.g., "</think>"
    /// What triggers the grammar to be evaluated.
    pub grammar_triggers: Vec<LlamaGrammarTrigger>,
    /// Tokens to preserve.
    pub preserved_tokens: Vec<String>,
    /// Additional tokens to stop generation at.
    pub additional_stops: Vec<String>,
    /// The parser to use.
    pub parser: String,
    /// Message spans.
    pub message_spans: Vec<LlamaChatMsgSpan>,
}

impl Default for LlamaChatParams {
    fn default() -> Self {
        Self {
            common_chat_format: LlamaChatFormat::default(),
            prompt: String::default(),
            grammar: None,
            grammar_lazy: false,
            generation_prompt: String::default(),
            supports_thinking: false,
            thinking_start_tag: None,
            thinking_end_tag: None,
            grammar_triggers: Vec::<LlamaGrammarTrigger>::default(),
            preserved_tokens: Vec::<String>::default(),
            additional_stops: Vec::<String>::default(),
            parser: String::default(),
            message_spans: Vec::<LlamaChatMsgSpan>::default(),
        }
    }
}

/// Safe enum for `common_chat_format`.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, Default)]
pub enum LlamaChatFormat {
    /// Chat format that only contains content.
    #[default]
    COMMON_CHAT_FORMAT_CONTENT_ONLY,
    /// Simple PEG based chat format.

    /// # Notes
    /// - Intended to be parsed by the PEG parser.
    COMMON_CHAT_FORMAT_PEG_SIMPLE,
    /// Native PEG based chat format.
    /// # Notes
    /// - Intended to be parsed by the PEG parser.
    COMMON_CHAT_FORMAT_PEG_NATIVE,
    /// Gemma4 PEG based chat format.
    /// # Notes
    /// - Intended to be parsed by the PEG parser.
    COMMON_CHAT_FORMAT_PEG_GEMMA4,

    /// Not a format, just the # formats.
    COMMON_CHAT_FORMAT_COUNT,
}

impl From<llama_rs_common_chat_format> for LlamaChatFormat {
    fn from(value: llama_rs_common_chat_format) -> Self {
        match value {
            LLAMA_RS_COMMON_CHAT_FORMAT_CONTENT_ONLY => Self::COMMON_CHAT_FORMAT_CONTENT_ONLY,
            LLAMA_RS_COMMON_CHAT_FORMAT_PEG_SIMPLE => Self::COMMON_CHAT_FORMAT_PEG_SIMPLE,
            LLAMA_RS_COMMON_CHAT_FORMAT_PEG_NATIVE => Self::COMMON_CHAT_FORMAT_PEG_NATIVE,
            LLAMA_RS_COMMON_CHAT_FORMAT_PEG_GEMMA4 => Self::COMMON_CHAT_FORMAT_PEG_GEMMA4,
            _ => Self::default(),
        }
    }
}

/// Enum for `common_grammar_trigger_type`
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy)]
pub enum LlamaGrammarTriggerType {
    /// Trigger grammar at a token boundary.
    COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN,
    /// Trigger grammar at a word boundary.
    COMMON_GRAMMAR_TRIGGER_TYPE_WORD,
    /// Trigger grammar at the start of a pattern.
    COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN,
    /// Trigger grammar at the end of a pattern.
    COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL,
}

impl From<llama_rs_common_grammar_trigger_type> for LlamaGrammarTriggerType {
    fn from(value: llama_rs_common_grammar_trigger_type) -> Self {
        match value {
            LLAMA_RS_COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN => Self::COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN,
            LLAMA_RS_COMMON_GRAMMAR_TRIGGER_TYPE_WORD => Self::COMMON_GRAMMAR_TRIGGER_TYPE_WORD,
            LLAMA_RS_COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN => {
                Self::COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN
            }
            LLAMA_RS_COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL => {
                Self::COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL
            }
            _ => Self::COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN,
        }
    }
}

/// Safe struct for `common_grammar_trigger`
#[derive(Debug, Clone)]
pub struct LlamaGrammarTrigger {
    /// The type of grammar trigger.
    pub trigger_type: LlamaGrammarTriggerType,
    /// The value of the grammar trigger.
    pub value: String,
    /// The token that triggers the grammar.
    pub token: LlamaToken,
}

/// Safe struct for `common_chat_msg_span`
#[derive(Debug, Clone)]
pub struct LlamaChatMsgSpan {
    /// The role of the message.
    pub role: String,
    /// The starting position of the message.
    pub pos: usize,
    /// The length of the message.
    pub len: usize,
}
