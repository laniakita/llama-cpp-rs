use std::{ffi::CStr, ptr::null_mut};

use llama_cpp_sys_2::{
    llama_rs_analyze_content, llama_rs_analyze_reasoning, llama_rs_analyze_tools,
    llama_rs_autoparser, llama_rs_autoparser_analyze_template, llama_rs_autoparser_free,
    llama_rs_autoparser_init, llama_rs_call_id_position, llama_rs_common_chat_template_free,
    llama_rs_common_chat_template_init, llama_rs_content_mode, llama_rs_reasoning_mode,
    llama_rs_template_analysis, llama_rs_template_analysis_free, llama_rs_tool_arguments_analysis,
    llama_rs_tool_format, llama_rs_tool_format_analysis, llama_rs_tool_function_analysis,
    llama_rs_tool_id_analysis, LLAMA_RS_CALL_ID_POSITION_BETWEEN_FUNC_AND_ARGS,
    LLAMA_RS_CALL_ID_POSITION_NONE, LLAMA_RS_CALL_ID_POSITION_POST_ARGS,
    LLAMA_RS_CALL_ID_POSITION_PRE_FUNC_NAME, LLAMA_RS_CONTENT_MODE_ALWAYS_WRAPPED,
    LLAMA_RS_CONTENT_MODE_PLAIN, LLAMA_RS_CONTENT_MODE_WRAPPED_WITH_REASONING,
    LLAMA_RS_REASONING_MODE_NONE, LLAMA_RS_REASONING_MODE_TAG_BASED,
    LLAMA_RS_REASONING_MODE_TOOLS_ONLY, LLAMA_RS_STATUS_INVALID_ARGUMENT,
    LLAMA_RS_TOOL_FORMAT_JSON_NATIVE, LLAMA_RS_TOOL_FORMAT_NONE,
    LLAMA_RS_TOOL_FORMAT_TAG_WITH_JSON, LLAMA_RS_TOOL_FORMAT_TAG_WITH_TAGGED,
};

use crate::{
    model::{LlamaChatTemplate, LlamaModel},
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
        template: LlamaChatTemplate,
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

        let common_template = {
            let mut decoder = encoding_rs::UTF_8.new_decoder();
            let bos_token_string =
                model.token_to_piece(model.token_bos(), &mut decoder, true, None)?;
            let bos_token = CStr::from_bytes_with_nul(bos_token_string.as_bytes())?;
            let eos_token_string =
                model.token_to_piece(model.token_eos(), &mut decoder, true, None)?;
            let eos_token = CStr::from_bytes_with_nul(eos_token_string.as_bytes())?;
            unsafe {
                llama_rs_common_chat_template_init(
                    template.as_c_str().as_ptr(),
                    bos_token.as_ptr(),
                    eos_token.as_ptr(),
                )
            }
        };

        let res = unsafe {
            llama_rs_autoparser_analyze_template(self.ptr, common_template, &mut analysis)
        };

        if res < 0 {
            match res {
                LLAMA_RS_STATUS_INVALID_ARGUMENT => {
                    return Err(AnalyzeTemplateError::InvalidTemplate(template));
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
