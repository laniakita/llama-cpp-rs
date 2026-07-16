#pragma once

#include "llama.cpp/include/llama.h"
#include <stdbool.h>
#include <stddef.h>

// ============================================================================
// Analysis Result Enums
// ============================================================================

/// Reasoning handling mode (derived from R1-R3 comparisons)
typedef enum llama_rs_reasoning_mode {
  /// No reasoning markers detected
  LLAMA_RS_REASONING_MODE_NONE = 0,
  /// Tag-based: Tag-based: <think>...</think> (start can be empty for
  /// delimiter-style)
  LLAMA_RS_REASONING_MODE_TAG_BASED = 1,
  /// Only reason on tool calls, not on normal content
  LLAMA_RS_REASONING_MODE_TOOLS_ONLY = 2,
} llama_rs_reasoning_mode;

/// Content wrapping mode (derived from C1 comparison)
typedef enum llama_rs_content_mode {
  /// No content markers
  LLAMA_RS_CONTENT_MODE_PLAIN = 0,
  /// Content always wrapped with markers
  LLAMA_RS_CONTENT_MODE_ALWAYS_WRAPPED = 1,
  /// Content wrapped only when reasoning present
  LLAMA_RS_CONTENT_MODE_WRAPPED_WITH_REASONING = 2,
} llama_rs_content_mode;

/// Call ID position in tool calls (for non-JSON formats)
typedef enum llama_rs_call_id_position {
  /// No call ID support detected
  LLAMA_RS_CALL_ID_POSITION_NONE = 0,
  /// Call ID before function name: [CALL_ID]id[FUNC]name{args}
  LLAMA_RS_CALL_ID_POSITION_PRE_FUNC_NAME = 1,
  /// Call ID between function and args: [FUNC]name[CALL_ID]id{args}
  LLAMA_RS_CALL_ID_POSITION_BETWEEN_FUNC_AND_ARGS = 2,
  /// Call ID after arguments: [FUNC]name{args}[CALL_ID]id
  LLAMA_RS_CALL_ID_POSITION_POST_ARGS = 3,
} llama_rs_call_id_position;

/// Tool call format classification (derived from T1-T5, A1-A3 comparisons)
typedef enum llama_rs_tool_format {
  /// No tool support detected
  LLAMA_RS_TOOL_FORMAT_NONE = 0,
  /// Pure JSON: {"name": "X", "arguments": {...}}
  LLAMA_RS_TOOL_FORMAT_JSON_NATIVE = 1,
  /// Tag-based with JSON args: <function=X>{...}</function>
  LLAMA_RS_TOOL_FORMAT_TAG_WITH_JSON = 2,
  /// Tag-based with tagged args: <param=key>value</param>
  LLAMA_RS_TOOL_FORMAT_TAG_WITH_TAGGED = 3,
} llama_rs_tool_format;

// ============================================================================
// Sub-structs for tool analysis
// ============================================================================

typedef struct llama_rs_tool_format_analysis {
  llama_rs_tool_format mode;
  /// e.g., "<tool_call>", "[TOOL_CALLS]", ""
  char *section_start;
  /// e.g., "</tool_call>", ""
  char *section_end;
  /// e.g., "<|tool_call_begin|>", "" (for multi-call templates)
  char *per_call_start;
  /// e.g., "<|tool_call_end|>", ""
  char *per_call_end;

  /// In JSON format function name is JSON key, i.e. { "<funname>": { ...
  /// arguments ... } }
  bool fun_name_is_key;
  /// Tool calls wrapped in JSON array [...]
  bool tools_array_wrapped;

  char *function_field;
  char *name_field;
  char *args_field;
  char *id_field;
  char *gen_id_field;
  char **parameter_order;
} llama_rs_tool_format_analysis;

typedef struct llama_rs_tool_function_analysis {
  /// e.g., "<function=", "\"name\": \"", "functions."
  char *name_prefix;
  /// e.g., ">", "\"", ":0"
  char *name_suffix;
  /// e.g., "</function>", "" (for tag-based)
  char *close;
} llama_rs_tool_function_analysis;

typedef struct llama_rs_tool_arguments_analysis {
  /// e.g., "<|tool_call_argument_begin|>", "<args>"
  char *start;
  /// e.g., "<|tool_call_argument_end|>", "</args>"
  char *end;
  /// e.g., "<param=", "<arg_key>", "\""
  char *name_prefix;
  /// e.g., ">", "</arg_key>", "\":"
  char *name_suffix;
  /// e.g., "", "<arg_value>", ""
  char *value_prefix;
  /// e.g., "</param>", "</arg_value>", ""
  char *value_suffix;
  /// e.g., "", "\n", ","
  char *separator;
} llama_rs_tool_arguments_analysis;

typedef struct llama_rs_tool_id_analysis {
  llama_rs_call_id_position pos;
  /// e.g., "[CALL_ID]" (marker before call ID value)
  char *prefix;
  /// e.g., "" (marker after call ID value, before next section)
  char *suffix;
} llama_rs_tool_id_analysis;

// ============================================================================
// Reasoning, Content, and Tool analyzer structs
// ============================================================================

typedef struct llama_rs_analyze_reasoning {
  llama_rs_reasoning_mode mode;
  char *start;
  char *end;
} llama_rs_analyze_reasoning;

typedef struct llama_rs_analyze_content {
  llama_rs_content_mode mode;
  /// e.g., "<response>", ">>>all\n", ""
  char *start;
  /// e.g., "</response>", ""
  char *end;
  bool requires_nonnull_content;
} llama_rs_analyze_content;

typedef struct llama_rs_analyze_tools {
  llama_rs_tool_format_analysis format;
  llama_rs_tool_function_analysis function;
  llama_rs_tool_arguments_analysis arguments;
  llama_rs_tool_id_analysis call_id;
} llama_rs_analyze_tools;

typedef struct llama_rs_template_analysis {
  llama_rs_analyze_reasoning reasoning;
  llama_rs_analyze_content content;
  llama_rs_analyze_tools tools;
} llama_rs_template_analysis;

// ============================================================================
// High-level params for parser generation
// ============================================================================

typedef struct llama_rs_chat_template_generation_params {
  const struct llama_rs_chat_message *messages;
  size_t n_messages;

  const struct llama_rs_chat_tool *tools;
  size_t n_tools;

  bool add_generation_prompt;
  bool enable_thinking;

  /// Stringified JSON object for Jinja kwargs
  const char *extra_context;
  /// Stringified JSON schema for constrained output
  const char *json_schema;
  const char *grammar;

  bool parallel_tool_calls;
  bool add_bos;
  bool add_eos;

} llama_rs_chat_template_generation_params;

/// tool.
typedef struct llama_rs_chat_tool {
  const char *name;
  const char *description;
  const char *parameters;
} llama_rs_chat_tool;

// Single tool call.
typedef struct llama_rs_chat_tool_call {
  const char *name;
  const char *arguments;
  const char *id;
} llama_rs_chat_tool_call;

// Single message.
typedef struct llama_rs_chat_message {
  const char *role;
  const char *content;
  const char *reasoning_content;
  const char *tool_name;
  const char *tool_call_id;

  /// Nested tool calls (e.g. assistant message may contain invoked tools).
  const struct llama_rs_chat_tool_call *tool_calls;
  size_t n_tool_calls;
} llama_rs_chat_message;

// ============================================================================
// Chat params result
// ============================================================================

typedef enum llama_rs_common_chat_format {
  LLAMA_RS_COMMON_CHAT_FORMAT_CONTENT_ONLY,

  /// These are intended to be parsed by the PEG parser
  LLAMA_RS_COMMON_CHAT_FORMAT_PEG_SIMPLE,
  LLAMA_RS_COMMON_CHAT_FORMAT_PEG_NATIVE,
  LLAMA_RS_COMMON_CHAT_FORMAT_PEG_GEMMA4,
  /// Not a format, just the # formats
  LLAMA_RS_COMMON_CHAT_FORMAT_COUNT,
} llama_rs_common_chat_format;

typedef enum llama_rs_common_grammar_trigger_type {
  LLAMA_RS_COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN,
  LLAMA_RS_COMMON_GRAMMAR_TRIGGER_TYPE_WORD,
  LLAMA_RS_COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN,
  LLAMA_RS_COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL,
} llama_rs_common_grammar_trigger_type;

typedef struct llama_rs_common_chat_params {
  enum llama_rs_common_chat_format format;
  char *prompt;
  char *grammar;
  bool grammar_lazy;
  char *generation_prompt;
  bool supports_thinking;
  /// e.g., "<think>"
  char *thinking_start_tag;
  /// e.g., "</think>"
  char *thinking_end_tag;
  const struct llama_rs_common_grammar_trigger *grammar_triggers;
  size_t n_grammar_triggers;
  char **preserved_tokens;
  size_t n_preserved_tokens;
  char **additional_stops;
  size_t n_additional_stops;
  char *parser;
  const struct llama_rs_common_chat_msg_span *message_spans;
  size_t n_message_spans;
} llama_rs_chat_params;

typedef struct llama_rs_common_grammar_trigger {
  enum llama_rs_common_grammar_trigger_type type;
  char *value;
  llama_token token;
} llama_rs_common_grammar_trigger;

typedef struct llama_rs_common_chat_msg_span {
  char *role;
  size_t pos;
  size_t len;
} llama_rs_common_chat_msg_span;
