#pragma once

#include "llama.cpp/include/llama.h"
#include <stdbool.h>
#include <stddef.h>

struct llama_rs_common_chat_params;
struct llama_rs_chat_parser;
struct llama_rs_common_chat_msg_diffs;

// ============================================================================
// Enum forwards
// ============================================================================

// Continuation method provided via `continue_final_message`
enum llama_rs_common_chat_continuation {
  LLAMA_RS_COMMON_CHAT_CONTINUATION_NONE,
  LLAMA_RS_COMMON_CHAT_CONTINUATION_AUTO,
  LLAMA_RS_COMMON_CHAT_CONTINUATION_REASONING,
  LLAMA_RS_COMMON_CHAT_CONTINUATION_CONTENT,
};

// reasoning API response format (not to be confused as chat template's
// reasoning format) only used by server
enum llama_rs_common_reasoning_format {
  /// Skip reasoning extraction.
  LLAMA_RS_COMMON_REASONING_FORMAT_NONE,
  /// Same as deepseek, using `message.reasoning_content`
  LLAMA_RS_COMMON_REASONING_FORMAT_AUTO,
  /// Extract thinking tag contents and return as `message.reasoning_content`,
  /// or leave inline in <think> tags in stream mode
  LLAMA_RS_COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY,
  /// Extract thinking tag contents and return as `message.reasoning_content`,
  /// including in streaming deltas.
  LLAMA_RS_COMMON_REASONING_FORMAT_DEEPSEEK,
};

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

  enum llama_rs_common_reasoning_format reasoning_format;
  enum llama_rs_common_chat_continuation continue_final_message;

  /// Stringified JSON object for Jinja kwargs
  const char *extra_context;
  /// Stringified JSON schema for constrained output
  const char *json_schema;
  const char *grammar;

  bool parallel_tool_calls;
  bool add_bos;
  bool add_eos;

} llama_rs_chat_template_generation_params;

/// tool declaration.
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
// Chat Parser Supporting Structs
// ============================================================================

typedef struct llama_rs_chat_msg_diff_view {
  const char *reasoning_content;
  const char *content;

  size_t tool_call_index; // SIZE_MAX if no tool call delta is present
  const char *tool_call_name;
  const char *tool_call_arguments;
  const char *tool_call_id;
} llama_rs_chat_msg_diff_view;

// ============================================================================
// Chat Params
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

typedef struct llama_rs_common_chat_params_view {
  enum llama_rs_common_chat_format format;
  const char *prompt;
  const char *grammar;
  bool grammar_lazy;
  const char *generation_prompt;
  bool supports_thinking;
  /// e.g., "<think>"
  const char *thinking_start_tag;
  /// e.g., "</think>"
  const char *thinking_end_tag;
  const struct llama_rs_common_grammar_trigger *grammar_triggers;
  size_t n_grammar_triggers;
  const char **preserved_tokens;
  size_t n_preserved_tokens;
  const char **additional_stops;
  size_t n_additional_stops;
  const char *parser;
  const struct llama_rs_common_chat_msg_span *message_spans;
  size_t n_message_spans;
} llama_rs_chat_params_view;

typedef struct llama_rs_common_grammar_trigger {
  enum llama_rs_common_grammar_trigger_type type;
  const char *value;
  llama_token token;
} llama_rs_common_grammar_trigger;

typedef struct llama_rs_common_chat_msg_span {
  const char *role;
  size_t pos;
  size_t len;
} llama_rs_common_chat_msg_span;
