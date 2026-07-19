#pragma once

#include "llama.cpp/include/llama.h"

#include <stdbool.h>
#include <stddef.h>

struct common_chat_template;
struct llama_model;
struct llama_sampler;
struct llama_rs_mtp_speculative;
struct llama_vocab;

#include "wrapper_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

llama_rs_status llama_rs_json_schema_to_grammar(const char *schema_json,
                                                bool force_gbnf,
                                                char **out_grammar);

struct llama_sampler *
llama_rs_sampler_init_grammar(const struct llama_vocab *vocab,
                              const char *grammar_str,
                              const char *grammar_root);

struct llama_sampler *llama_rs_sampler_init_grammar_lazy(
    const struct llama_vocab *vocab, const char *grammar_str,
    const char *grammar_root, const char **trigger_words,
    size_t num_trigger_words, const llama_token *trigger_tokens,
    size_t num_trigger_tokens);

struct llama_sampler *llama_rs_sampler_init_grammar_lazy_patterns(
    const struct llama_vocab *vocab, const char *grammar_str,
    const char *grammar_root, const char **trigger_patterns,
    size_t num_trigger_patterns, const llama_token *trigger_tokens,
    size_t num_trigger_tokens);

llama_rs_status llama_rs_sampler_accept(struct llama_sampler *sampler,
                                        llama_token token);

// Fit model/context params to device memory (wraps llama.cpp's
// common_fit_params). Returns common_params_fit_status as an int: 0 = success,
// 1 = failure, 2 = error.
int llama_rs_fit_params(
    const char *path_model, struct llama_model_params *mparams,
    struct llama_context_params *cparams, float *tensor_split,
    struct llama_model_tensor_buft_override *tensor_buft_overrides,
    size_t *margins, uint32_t n_ctx_min, enum ggml_log_level log_level);

void llama_rs_memory_breakdown_print(const struct llama_context *ctx);

struct llama_rs_mtp_speculative *
llama_rs_mtp_speculative_init(struct llama_context *ctx_tgt,
                              struct llama_context *ctx_dft, int32_t n_max,
                              int32_t n_min, float p_min);

void llama_rs_mtp_speculative_free(struct llama_rs_mtp_speculative *spec);

llama_rs_status
llama_rs_mtp_speculative_begin(struct llama_rs_mtp_speculative *spec,
                               const llama_token *prompt_tokens,
                               size_t prompt_tokens_count);

llama_rs_status
llama_rs_mtp_speculative_process(struct llama_rs_mtp_speculative *spec,
                                 const struct llama_batch *batch);

llama_rs_status llama_rs_mtp_speculative_draft(
    struct llama_rs_mtp_speculative *spec, llama_pos n_past,
    llama_token id_last, const llama_token *prompt_tokens,
    size_t prompt_tokens_count, llama_token *out_tokens,
    size_t out_tokens_capacity, size_t *out_tokens_count);

llama_rs_status
llama_rs_mtp_speculative_accept(struct llama_rs_mtp_speculative *spec,
                                uint16_t n_accepted);

void llama_rs_string_free(char *ptr);

struct common_chat_templates_inputs;
struct common_chat_params;
struct common_chat_parser_params;
struct common_chat_msg;
struct common_chat_msg_diffs; // Opaque wrapper for
                              // std::vector<common_chat_msg_diff>

enum llama_rs_common_chat_format {
  LLAMA_RS_COMMON_CHAT_FORMAT_CONTENT_ONLY,
  LLAMA_RS_COMMON_CHAT_FORMAT_PEG_NATIVE,
  LLAMA_RS_COMMON_CHAT_FORMAT_PEG_SIMPLE,
  LLAMA_RS_COMMON_CHAT_FORMAT_PEG_GEMMA4,
  LLAMA_RS_COMMON_CHAT_FORMAT_COUNT,
};

enum llama_rs_common_chat_continuation {
  LLAMA_RS_COMMON_CHAT_CONTINUATION_NONE,
  LLAMA_RS_COMMON_CHAT_CONTINUATION_AUTO,
  LLAMA_RS_COMMON_CHAT_CONTINUATION_REASONING,
  LLAMA_RS_COMMON_CHAT_CONTINUATION_CONTENT,
};

enum llama_rs_common_reasoning_format {
  LLAMA_RS_COMMON_REASONING_FORMAT_NONE,
  LLAMA_RS_COMMON_REASONING_FORMAT_AUTO,
  LLAMA_RS_COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY,
  LLAMA_RS_COMMON_REASONING_FORMAT_DEEPSEEK,
};

struct common_chat_templates_inputs *common_chat_templates_inputs_create(
    bool add_generation_prompt, bool enable_thinking, int32_t reasoning_format,
    int32_t continue_final_message, bool parallel_tool_calls, bool add_bos,
    bool add_eos, const char *json_schema, const char *grammar,
    const char *extra_context);
void common_chat_templates_inputs_free(
    struct common_chat_templates_inputs *inputs);

llama_rs_status common_chat_templates_inputs_add_message(
    struct common_chat_templates_inputs *inputs, const char *role,
    const char *content, const char *reasoning_content, const char *tool_name,
    const char *tool_call_id);
llama_rs_status common_chat_templates_inputs_add_tool_call_to_last_message(
    struct common_chat_templates_inputs *inputs, const char *name,
    const char *arguments, const char *id);
llama_rs_status common_chat_templates_inputs_add_tool(
    struct common_chat_templates_inputs *inputs, const char *name,
    const char *description, const char *parameters);

struct common_chat_params *
common_chat_apply_template(const struct llama_model *model,
                           const char *chat_template,
                           const struct common_chat_templates_inputs *inputs);
void common_chat_params_free(struct common_chat_params *params);

struct common_chat_params_view {
  int32_t format;
  const char *prompt;
  const char *grammar;
  bool grammar_lazy;
  const char *generation_prompt;
  bool supports_thinking;
  const char *thinking_start_tag;
  const char *thinking_end_tag;
  const char *parser;
};
struct common_chat_params_view
common_chat_params_get_view(const struct common_chat_params *params);

enum llama_rs_common_grammar_trigger_type {
  LLAMA_RS_COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN = 0,
  LLAMA_RS_COMMON_GRAMMAR_TRIGGER_TYPE_WORD = 1,
  LLAMA_RS_COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN = 2,
  LLAMA_RS_COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL = 3,
};

struct common_grammar_trigger_view {
  int32_t type;
  const char *value;
  int32_t token;
};

enum llama_rs_common_chat_role {
  LLAMA_RS_COMMON_CHAT_ROLE_UNKNOWN,
  LLAMA_RS_COMMON_CHAT_ROLE_SYSTEM,
  LLAMA_RS_COMMON_CHAT_ROLE_ASSISTANT,
  LLAMA_RS_COMMON_CHAT_ROLE_USER,
  LLAMA_RS_COMMON_CHAT_ROLE_TOOL
};

struct common_chat_msg_delimiter_view {
  int32_t role;
  const char *delimiter;
  const int32_t *tokens;
  size_t tokens_count;
};

size_t common_chat_params_get_message_delimiters_count(
    const struct common_chat_params *params);
struct common_chat_msg_delimiter_view common_chat_params_get_message_delimiter(
    const struct common_chat_params *params, size_t index);

size_t common_chat_params_get_grammar_triggers_count(
    const struct common_chat_params *params);
struct common_grammar_trigger_view
common_chat_params_get_grammar_trigger(const struct common_chat_params *params,
                                       size_t index);

size_t common_chat_params_get_preserved_tokens_count(
    const struct common_chat_params *params);
const char *
common_chat_params_get_preserved_token(const struct common_chat_params *params,
                                       size_t index);

struct llama_rs_chat_parser;

struct llama_rs_chat_parser *
llama_rs_chat_parser_init(const struct common_chat_params *params,
                          const struct common_chat_templates_inputs *inputs);

void llama_rs_chat_parser_free(struct llama_rs_chat_parser *parser);

llama_rs_status
llama_rs_chat_parser_feed(struct llama_rs_chat_parser *parser,
                          const char *chunk, bool is_partial,
                          struct common_chat_msg_diffs **out_diffs);

void common_chat_msg_diffs_free(struct common_chat_msg_diffs *diffs);
size_t
common_chat_msg_diffs_get_size(const struct common_chat_msg_diffs *diffs);

struct common_chat_msg_diff_view {
  const char *reasoning_content;
  const char *content;
  size_t tool_call_index;
  const char *tool_call_name;
  const char *tool_call_arguments;
  const char *tool_call_id;
};
struct common_chat_msg_diff_view
common_chat_msg_diffs_get_view(const struct common_chat_msg_diffs *diffs,
                               size_t index);

#ifdef __cplusplus
}
#endif
