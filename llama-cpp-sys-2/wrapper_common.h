#pragma once

#include "llama.cpp/include/llama.h"

#include <stdbool.h>
#include <stddef.h>

struct common_chat_template;
struct llama_model;
struct llama_sampler;
struct llama_rs_mtp_speculative;
struct llama_vocab;

#include "wrapper_chat-parser.h"
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

llama_rs_status llama_rs_chat_apply_template_with_params(
    const struct llama_model *model, const char *chat_template,
    const struct llama_rs_chat_template_generation_params *params,
    struct llama_rs_common_chat_params *out_chat_params);

struct llama_rs_common_chat_params *llama_rs_common_chat_params_init(void);

void llama_rs_common_chat_params_free(
    struct llama_rs_common_chat_params *params);

struct llama_rs_common_chat_params_view *llama_rs_common_chat_params_view_init(
    const struct llama_rs_common_chat_params *params);

void llama_rs_common_chat_params_view_free(
    struct llama_rs_common_chat_params_view *view);

struct llama_rs_chat_parser *llama_rs_chat_parser_init(
    const struct llama_rs_common_chat_params *params,
    const struct llama_rs_chat_template_generation_params *opt);

void llama_rs_chat_parser_free(struct llama_rs_chat_parser *parser);

struct llama_rs_common_chat_msg_diffs *
llama_rs_common_chat_msg_diffs_init(void);

llama_rs_status
llama_rs_chat_parser_feed(struct llama_rs_chat_parser *parser,
                          const char *chunk,
                          struct llama_rs_common_chat_msg_diffs **out_diffs);

void llama_rs_common_chat_msg_diffs_free(
    struct llama_rs_common_chat_msg_diffs *diffs);

size_t
llama_rs_chat_msg_diffs_len(const struct llama_rs_common_chat_msg_diffs *diffs);

struct llama_rs_chat_msg_diff_view *llama_rs_chat_msg_diff_view_init(void);

llama_rs_status llama_rs_chat_msg_diff_get_view(
    const struct llama_rs_common_chat_msg_diffs *diffs, size_t index,
    struct llama_rs_chat_msg_diff_view *out_view);

void llama_rs_chat_msg_diff_view_free(struct llama_rs_chat_msg_diff_view *view);

#ifdef __cplusplus
}
#endif
