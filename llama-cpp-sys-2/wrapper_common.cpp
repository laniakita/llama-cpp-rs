#include "wrapper_common.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <stdint.h>
#include <string>
#include <vector>

#include "chat.h"
#include "llama.cpp/common/chat-auto-parser.h"
#include "llama.cpp/common/common.h"
#include "llama.cpp/common/fit.h"
#include "llama.cpp/common/json-schema-to-grammar.h"
#include "llama.cpp/common/speculative.h"
#include "llama.cpp/include/llama.h"
#include "wrapper_auto-parser.h"
#include "wrapper_utils.h"

#include <nlohmann/json.hpp>

extern "C" llama_rs_status
llama_rs_json_schema_to_grammar(const char *schema_json, bool force_gbnf,
                                char **out_grammar) {
  if (!schema_json || !out_grammar) {
    return LLAMA_RS_STATUS_INVALID_ARGUMENT;
  }

  *out_grammar = nullptr;
  try {
    const auto schema = nlohmann::ordered_json::parse(schema_json);
    const auto grammar = json_schema_to_grammar(schema, force_gbnf);
    *out_grammar = llama_rs_dup_string(grammar);
    return *out_grammar ? LLAMA_RS_STATUS_OK
                        : LLAMA_RS_STATUS_ALLOCATION_FAILED;
  } catch (const std::exception &) {
    return LLAMA_RS_STATUS_EXCEPTION;
  }
}

extern "C" void llama_rs_string_free(char *ptr) {
  if (ptr) {
    std::free(ptr);
  }
}

extern "C" struct llama_sampler *
llama_rs_sampler_init_grammar(const struct llama_vocab *vocab,
                              const char *grammar_str,
                              const char *grammar_root) {
  try {
    return llama_sampler_init_grammar(vocab, grammar_str, grammar_root);
  } catch (...) {
    return nullptr;
  }
}

extern "C" struct llama_sampler *llama_rs_sampler_init_grammar_lazy(
    const struct llama_vocab *vocab, const char *grammar_str,
    const char *grammar_root, const char **trigger_words,
    size_t num_trigger_words, const llama_token *trigger_tokens,
    size_t num_trigger_tokens) {
  try {
    std::vector<std::string> trigger_patterns;
    trigger_patterns.reserve(num_trigger_words);
    for (size_t i = 0; i < num_trigger_words; ++i) {
      const char *word = trigger_words ? trigger_words[i] : nullptr;
      if (word && word[0] != '\0') {
        trigger_patterns.push_back(regex_escape(word));
      }
    }
    std::vector<const char *> trigger_patterns_c;
    trigger_patterns_c.reserve(trigger_patterns.size());
    for (const auto &pattern : trigger_patterns) {
      trigger_patterns_c.push_back(pattern.c_str());
    }
    return llama_sampler_init_grammar_lazy_patterns(
        vocab, grammar_str, grammar_root, trigger_patterns_c.data(),
        trigger_patterns_c.size(), trigger_tokens, num_trigger_tokens);
  } catch (...) {
    return nullptr;
  }
}

extern "C" struct llama_sampler *llama_rs_sampler_init_grammar_lazy_patterns(
    const struct llama_vocab *vocab, const char *grammar_str,
    const char *grammar_root, const char **trigger_patterns,
    size_t num_trigger_patterns, const llama_token *trigger_tokens,
    size_t num_trigger_tokens) {
  try {
    return llama_sampler_init_grammar_lazy_patterns(
        vocab, grammar_str, grammar_root, trigger_patterns,
        num_trigger_patterns, trigger_tokens, num_trigger_tokens);
  } catch (...) {
    return nullptr;
  }
}

extern "C" llama_rs_status
llama_rs_sampler_accept(struct llama_sampler *sampler, llama_token token) {
  if (!sampler) {
    return LLAMA_RS_STATUS_INVALID_ARGUMENT;
  }
  try {
    llama_sampler_accept(sampler, token);
    return LLAMA_RS_STATUS_OK;
  } catch (const std::exception &) {
    return LLAMA_RS_STATUS_EXCEPTION;
  } catch (...) {
    return LLAMA_RS_STATUS_EXCEPTION;
  }
}

// Thin pass-through to llama.cpp's common_fit_params (a C++ symbol in
// libcommon). Returns common_params_fit_status as an int: 0 = success, 1 =
// failure, 2 = error.
extern "C" int llama_rs_fit_params(
    const char *path_model, struct llama_model_params *mparams,
    struct llama_context_params *cparams, float *tensor_split,
    struct llama_model_tensor_buft_override *tensor_buft_overrides,
    size_t *margins, uint32_t n_ctx_min, enum ggml_log_level log_level) {
  return static_cast<int>(common_fit_params(path_model, mparams, cparams,
                                            tensor_split, tensor_buft_overrides,
                                            margins, n_ctx_min, log_level));
}

extern "C" void
llama_rs_memory_breakdown_print(const struct llama_context *ctx) {
  common_memory_breakdown_print(ctx);
}

struct llama_rs_mtp_speculative {
  common_params_speculative params;
  common_speculative *spec = nullptr;
  std::vector<llama_token> prompt;
  std::vector<llama_token> draft;
  size_t last_draft_len = 0;
  bool draft_pending = false;
};

static constexpr llama_seq_id LLAMA_RS_MTP_SEQ_ID = 0;

static bool llama_rs_mtp_batch_compatible(const struct llama_batch &batch) {
  if (batch.n_tokens <= 0 || !batch.token || batch.embd || !batch.pos ||
      !batch.n_seq_id || !batch.seq_id) {
    return false;
  }
  for (int32_t k = 0; k < batch.n_tokens; ++k) {
    if (batch.n_seq_id[k] != 1 || !batch.seq_id[k] ||
        batch.seq_id[k][0] != LLAMA_RS_MTP_SEQ_ID) {
      return false;
    }
  }
  return true;
}

static void llama_rs_assign_tokens(std::vector<llama_token> &dst,
                                   const llama_token *tokens, size_t count) {
  if (count == 0) {
    dst.clear();
    return;
  }
  dst.assign(tokens, tokens + count);
}

extern "C" struct llama_rs_mtp_speculative *
llama_rs_mtp_speculative_init(struct llama_context *ctx_tgt,
                              struct llama_context *ctx_dft, int32_t n_max,
                              int32_t n_min, float p_min) {
  if (!ctx_tgt || !ctx_dft || n_max <= 0 || n_min < 0 || n_min > n_max) {
    return nullptr;
  }

  try {
    auto wrapper = std::make_unique<llama_rs_mtp_speculative>();
    wrapper->params.types = {COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
    wrapper->params.draft.ctx_tgt = ctx_tgt;
    wrapper->params.draft.ctx_dft = ctx_dft;
    wrapper->params.draft.n_max = n_max;
    wrapper->params.draft.n_min = n_min;
    wrapper->params.draft.p_min = p_min;

    wrapper->spec = common_speculative_init(wrapper->params, 1);
    if (!wrapper->spec) {
      return nullptr;
    }

    return wrapper.release();
  } catch (...) {
    return nullptr;
  }
}

extern "C" void
llama_rs_mtp_speculative_free(struct llama_rs_mtp_speculative *spec) {
  if (!spec) {
    return;
  }
  if (spec->spec) {
    common_speculative_free(spec->spec);
    spec->spec = nullptr;
  }
  delete spec;
}

extern "C" llama_rs_status
llama_rs_mtp_speculative_begin(struct llama_rs_mtp_speculative *spec,
                               const llama_token *prompt_tokens,
                               size_t prompt_tokens_count) {
  if (!spec || !spec->spec || (!prompt_tokens && prompt_tokens_count > 0)) {
    return LLAMA_RS_STATUS_INVALID_ARGUMENT;
  }

  try {
    llama_rs_assign_tokens(spec->prompt, prompt_tokens, prompt_tokens_count);
    spec->last_draft_len = 0;
    spec->draft_pending = false;
    common_speculative_begin(spec->spec, LLAMA_RS_MTP_SEQ_ID, spec->prompt);
    return LLAMA_RS_STATUS_OK;
  } catch (...) {
    return LLAMA_RS_STATUS_EXCEPTION;
  }
}

extern "C" llama_rs_status
llama_rs_mtp_speculative_process(struct llama_rs_mtp_speculative *spec,
                                 const struct llama_batch *batch) {
  if (!spec || !spec->spec || !batch) {
    return LLAMA_RS_STATUS_INVALID_ARGUMENT;
  }
  if (!llama_rs_mtp_batch_compatible(*batch)) {
    return LLAMA_RS_STATUS_INVALID_ARGUMENT;
  }

  try {
    return common_speculative_process(spec->spec, *batch)
               ? LLAMA_RS_STATUS_OK
               : LLAMA_RS_STATUS_EXCEPTION;
  } catch (...) {
    return LLAMA_RS_STATUS_EXCEPTION;
  }
}

extern "C" llama_rs_status llama_rs_mtp_speculative_draft(
    struct llama_rs_mtp_speculative *spec, llama_pos n_past,
    llama_token id_last, const llama_token *prompt_tokens,
    size_t prompt_tokens_count, llama_token *out_tokens,
    size_t out_tokens_capacity, size_t *out_tokens_count) {
  if (!spec || !spec->spec || (!prompt_tokens && prompt_tokens_count > 0) ||
      !out_tokens_count || n_past < 0) {
    return LLAMA_RS_STATUS_INVALID_ARGUMENT;
  }

  try {
    if (spec->draft_pending) {
      return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    llama_rs_assign_tokens(spec->prompt, prompt_tokens, prompt_tokens_count);
    spec->draft.clear();
    spec->last_draft_len = 0;

    auto &params =
        common_speculative_get_draft_params(spec->spec, LLAMA_RS_MTP_SEQ_ID);
    params = {
        true,         spec->params.draft.n_max, n_past, id_last, &spec->prompt,
        &spec->draft,
    };

    common_speculative_draft(spec->spec);

    *out_tokens_count = spec->draft.size();
    if (spec->draft.size() > out_tokens_capacity) {
      return LLAMA_RS_STATUS_ALLOCATION_FAILED;
    }
    if (!spec->draft.empty() && !out_tokens) {
      return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    if (!spec->draft.empty()) {
      std::memcpy(out_tokens, spec->draft.data(),
                  spec->draft.size() * sizeof(llama_token));
    }
    spec->last_draft_len = spec->draft.size();
    spec->draft_pending = !spec->draft.empty();
    return LLAMA_RS_STATUS_OK;
  } catch (...) {
    return LLAMA_RS_STATUS_EXCEPTION;
  }
}

extern "C" llama_rs_status
llama_rs_mtp_speculative_accept(struct llama_rs_mtp_speculative *spec,
                                uint16_t n_accepted) {
  if (!spec || !spec->spec) {
    return LLAMA_RS_STATUS_INVALID_ARGUMENT;
  }
  if (!spec->draft_pending || n_accepted > spec->last_draft_len) {
    return LLAMA_RS_STATUS_INVALID_ARGUMENT;
  }

  try {
    common_speculative_accept(spec->spec, LLAMA_RS_MTP_SEQ_ID, n_accepted);
    spec->last_draft_len = 0;
    spec->draft_pending = false;
    return LLAMA_RS_STATUS_OK;
  } catch (...) {
    return LLAMA_RS_STATUS_EXCEPTION;
  }
}

extern "C" struct llama_rs_autoparser *llama_rs_autoparser_init(void) {
  try {
    auto *parser = new autoparser::autoparser();
    return reinterpret_cast<struct llama_rs_autoparser *>(parser);
  } catch (...) {
    return nullptr;
  }
}

extern "C" void llama_rs_autoparser_free(struct llama_rs_autoparser *parser) {
  if (!parser) {
    return;
  }
  delete reinterpret_cast<autoparser::autoparser *>(parser);
}

extern "C" llama_rs_status llama_rs_autoparser_analyze_template(
    struct llama_rs_autoparser *parser, const struct llama_model *model,
    const char *chat_template,
    struct llama_rs_template_analysis *out_analysis) {
  if (!parser || !out_analysis) {
    return LLAMA_RS_STATUS_INVALID_ARGUMENT;
  }
  try {
    common_chat_templates_ptr tmpls = common_chat_templates_init(
        model, chat_template ? chat_template : "", "", "");
    
    std::string source = common_chat_templates_source(tmpls.get());
    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    std::string token_bos;
    llama_token bos_id = llama_vocab_bos(vocab);
    if (bos_id != LLAMA_TOKEN_NULL) {
        token_bos = common_token_to_piece(vocab, bos_id, true);
    }
    std::string token_eos;
    llama_token eos_id = llama_vocab_eos(vocab);
    if (eos_id != LLAMA_TOKEN_NULL) {
        token_eos = common_token_to_piece(vocab, eos_id, true);
    }

    common_chat_template tmpl(source, token_bos, token_eos);

    auto *p = reinterpret_cast<autoparser::autoparser *>(parser);
    p->analyze_template(tmpl);

    out_analysis->reasoning = {
        .mode = static_cast<llama_rs_reasoning_mode>(p->reasoning.mode),
        .start = llama_rs_dup_string(p->reasoning.start),
        .end = llama_rs_dup_string(p->reasoning.end),
    };
    out_analysis->content = {
        .mode = static_cast<llama_rs_content_mode>(p->content.mode),
        .start = llama_rs_dup_string(p->content.start),
        .end = llama_rs_dup_string(p->content.end),
        .requires_nonnull_content = p->content.requires_nonnull_content,
    };
    out_analysis->tools = {
        .format =
            {
                .mode = static_cast<llama_rs_tool_format>(p->tools.format.mode),
                .section_start =
                    llama_rs_dup_string(p->tools.format.section_start),
                .section_end = llama_rs_dup_string(p->tools.format.section_end),
                .per_call_start =
                    llama_rs_dup_string(p->tools.format.per_call_start),
                .per_call_end =
                    llama_rs_dup_string(p->tools.format.per_call_end),
                .fun_name_is_key = p->tools.format.fun_name_is_key,
                .tools_array_wrapped = p->tools.format.tools_array_wrapped,
                .function_field =
                    llama_rs_dup_string(p->tools.format.function_field),
                .name_field = llama_rs_dup_string(p->tools.format.name_field),
                .args_field = llama_rs_dup_string(p->tools.format.args_field),
                .id_field = llama_rs_dup_string(p->tools.format.id_field),
                .gen_id_field =
                    llama_rs_dup_string(p->tools.format.gen_id_field),
                .parameter_order =
                    llama_rs_dup_string_vector(p->tools.format.parameter_order),
            },
        .function = {.name_prefix =
                         llama_rs_dup_string(p->tools.function.name_prefix),
                     .name_suffix =
                         llama_rs_dup_string(p->tools.function.name_suffix),
                     .close = llama_rs_dup_string(p->tools.function.close)},
        .arguments =
            {.start = llama_rs_dup_string(p->tools.arguments.start),
             .end = llama_rs_dup_string(p->tools.arguments.end),
             .name_prefix = llama_rs_dup_string(p->tools.arguments.name_prefix),
             .name_suffix = llama_rs_dup_string(p->tools.arguments.name_suffix),
             .value_prefix =
                 llama_rs_dup_string(p->tools.arguments.value_prefix),
             .value_suffix =
                 llama_rs_dup_string(p->tools.arguments.value_suffix),
             .separator = llama_rs_dup_string(p->tools.arguments.separator)},
        .call_id = {
            .pos = static_cast<llama_rs_call_id_position>(p->tools.call_id.pos),
            .prefix = llama_rs_dup_string(p->tools.call_id.prefix),
            .suffix = llama_rs_dup_string(p->tools.call_id.suffix)}};

    return LLAMA_RS_STATUS_OK;
  } catch (...) {
    return LLAMA_RS_STATUS_EXCEPTION;
  }
}

extern "C" void
llama_rs_template_analysis_free(struct llama_rs_template_analysis *analysis) {
  if (!analysis) {
    return;
  }

  llama_rs_string_free(analysis->reasoning.start);
  llama_rs_string_free(analysis->reasoning.end);

  llama_rs_string_free(analysis->content.start);
  llama_rs_string_free(analysis->content.end);

  llama_rs_string_free(analysis->tools.format.section_start);
  llama_rs_string_free(analysis->tools.format.section_end);
  llama_rs_string_free(analysis->tools.format.per_call_start);
  llama_rs_string_free(analysis->tools.format.per_call_end);
  llama_rs_string_free(analysis->tools.format.function_field);
  llama_rs_string_free(analysis->tools.format.name_field);
  llama_rs_string_free(analysis->tools.format.args_field);
  llama_rs_string_free(analysis->tools.format.id_field);
  llama_rs_string_free(analysis->tools.format.gen_id_field);
  if (analysis->tools.format.parameter_order) {
    for (size_t i = 0; analysis->tools.format.parameter_order[i] != nullptr;
         i++) {
      llama_rs_string_free(analysis->tools.format.parameter_order[i]);
    }
    std::free(analysis->tools.format.parameter_order);
  }

  llama_rs_string_free(analysis->tools.function.name_prefix);
  llama_rs_string_free(analysis->tools.function.name_suffix);
  llama_rs_string_free(analysis->tools.function.close);

  llama_rs_string_free(analysis->tools.arguments.start);
  llama_rs_string_free(analysis->tools.arguments.end);
  llama_rs_string_free(analysis->tools.arguments.name_prefix);
  llama_rs_string_free(analysis->tools.arguments.name_suffix);
  llama_rs_string_free(analysis->tools.arguments.value_prefix);
  llama_rs_string_free(analysis->tools.arguments.value_suffix);
  llama_rs_string_free(analysis->tools.arguments.separator);

  llama_rs_string_free(analysis->tools.call_id.prefix);
  llama_rs_string_free(analysis->tools.call_id.suffix);
}

extern "C" llama_rs_status llama_rs_chat_apply_template_with_params(
    const struct llama_model *model,
    const char *chat_template,
    const struct llama_rs_chat_template_generation_params *params,
    struct llama_rs_common_chat_params *out_chat_params) {
  if (!params || !out_chat_params) {
    return LLAMA_RS_STATUS_INVALID_ARGUMENT;
  }
  try {
    common_chat_templates_ptr tmpls = common_chat_templates_init(
        model, chat_template ? chat_template : "", "", "");

    common_chat_templates_inputs inputs;

    inputs.enable_thinking = params->enable_thinking;
    inputs.add_bos = params->add_bos;
    inputs.add_eos = params->add_eos;
    inputs.add_generation_prompt = params->add_generation_prompt;
    inputs.parallel_tool_calls = params->parallel_tool_calls;

    if (params->extra_context) {
      auto parsed = nlohmann::json::parse(params->extra_context);
      if (parsed.is_object()) {
        for (const auto &[k, v] : parsed.items()) {
          inputs.chat_template_kwargs[k] = v.dump();
        }
      }
    }

    if (params->grammar) {
      inputs.grammar = params->grammar;
    } else if (params->json_schema) {
      inputs.json_schema = params->json_schema;
    }

    for (size_t i = 0; i < params->n_messages; ++i) {
      common_chat_msg msg;
      if (params->messages[i].role) {
        msg.role = params->messages[i].role;
      }
      if (params->messages[i].content) {
        msg.content = params->messages[i].content;
      }
      if (params->messages[i].reasoning_content) {
        msg.reasoning_content = params->messages[i].reasoning_content;
      }
      if (params->messages[i].tool_name) {
        msg.tool_name = params->messages[i].tool_name;
      }
      if (params->messages[i].tool_call_id) {
        msg.tool_call_id = params->messages[i].tool_call_id;
      }

      if (params->messages[i].n_tool_calls > 0) {
        for (size_t j = 0; j < params->messages[i].n_tool_calls; ++j) {
          common_chat_tool_call tc;
          if (params->messages[i].tool_calls[j].id) {
            tc.id = params->messages[i].tool_calls[j].id;
          }
          if (params->messages[i].tool_calls[j].name) {
            tc.name = params->messages[i].tool_calls[j].name;
          }
          if (params->messages[i].tool_calls[j].arguments) {
            tc.arguments = params->messages[i].tool_calls[j].arguments;
          }
          msg.tool_calls.push_back(tc);
        }
      }
      inputs.messages.push_back(msg);
    }

    if (params->n_tools > 0) {
      for (size_t i = 0; i < params->n_tools; ++i) {
        common_chat_tool tl;
        if (params->tools[i].name) {
          tl.name = params->tools[i].name;
        }
        if (params->tools[i].description) {
          tl.description = params->tools[i].description;
        }
        if (params->tools[i].parameters) {
          tl.parameters = params->tools[i].parameters;
        }
        inputs.tools.push_back(tl);
      }
    }

    common_chat_params res = common_chat_templates_apply(tmpls.get(), inputs);

    llama_rs_common_grammar_trigger *triggers_arr = nullptr;
    if (!res.grammar_triggers.empty()) {
      triggers_arr = static_cast<llama_rs_common_grammar_trigger *>(
          std::malloc(res.grammar_triggers.size() *
                      sizeof(llama_rs_common_grammar_trigger)));
      for (size_t i = 0; i < res.grammar_triggers.size(); ++i) {
        triggers_arr[i].type =
            llama_rs_common_grammar_trigger_type(res.grammar_triggers[i].type);
        triggers_arr[i].value =
            llama_rs_dup_string(res.grammar_triggers[i].value);
        triggers_arr[i].token = res.grammar_triggers[i].token;
      }
    }

    llama_rs_common_chat_msg_span *spans_arr = nullptr;
    if (!res.message_spans.empty()) {
      spans_arr = static_cast<llama_rs_common_chat_msg_span *>(std::malloc(
          res.message_spans.size() * sizeof(llama_rs_common_chat_msg_span)));
      for (size_t i = 0; i < res.message_spans.size(); ++i) {
        spans_arr[i].role = llama_rs_dup_string(res.message_spans[i].role);
        spans_arr[i].pos = res.message_spans[i].pos;
        spans_arr[i].len = res.message_spans[i].len;
      }
    }

    *out_chat_params = {
        .format = llama_rs_common_chat_format(res.format),
        .prompt = llama_rs_dup_string(res.prompt),
        .grammar = llama_rs_dup_string(res.grammar),
        .grammar_lazy = res.grammar_lazy,
        .generation_prompt = llama_rs_dup_string(res.generation_prompt),
        .supports_thinking = res.supports_thinking,
        .thinking_start_tag = llama_rs_dup_string(res.thinking_start_tag),
        .thinking_end_tag = llama_rs_dup_string(res.thinking_end_tag),
        .grammar_triggers = triggers_arr,
        .n_grammar_triggers = res.grammar_triggers.size(),
        .preserved_tokens = llama_rs_dup_string_vector(res.preserved_tokens),
        .n_preserved_tokens = res.preserved_tokens.size(),
        .additional_stops = llama_rs_dup_string_vector(res.additional_stops),
        .n_additional_stops = res.additional_stops.size(),
        .parser = llama_rs_dup_string(res.parser),
        .message_spans = spans_arr,
        .n_message_spans = res.message_spans.size(),
    };
    return LLAMA_RS_STATUS_OK;
  } catch (...) {
    return LLAMA_RS_STATUS_EXCEPTION;
  }
}

extern "C" struct llama_rs_common_chat_params *
llama_rs_common_chat_params_init() {
  return new llama_rs_common_chat_params();
}

extern "C" void
llama_rs_common_chat_params_free(struct llama_rs_common_chat_params *params) {
  if (!params) {
    return;
  }
  llama_rs_string_free(params->prompt);
  llama_rs_string_free(params->grammar);
  llama_rs_string_free(params->generation_prompt);
  llama_rs_string_free(params->thinking_start_tag);
  llama_rs_string_free(params->thinking_end_tag);
  if (params->grammar_triggers) {
    for (size_t i = 0; i < params->n_grammar_triggers; ++i) {
      llama_rs_string_free(params->grammar_triggers[i].value);
    }
    std::free(const_cast<struct llama_rs_common_grammar_trigger *>(
        params->grammar_triggers));
  }
  if (params->preserved_tokens) {
    for (size_t i = 0; i < params->n_preserved_tokens; ++i) {
      llama_rs_string_free(params->preserved_tokens[i]);
    }
    std::free(params->preserved_tokens);
  }
  if (params->additional_stops) {
    for (size_t i = 0; i < params->n_additional_stops; ++i) {
      llama_rs_string_free(params->additional_stops[i]);
    }
    std::free(params->additional_stops);
  }
  llama_rs_string_free(params->parser);
  if (params->message_spans) {
    for (size_t i = 0; i < params->n_message_spans; ++i) {
      llama_rs_string_free(const_cast<char *>(params->message_spans[i].role));
    }
    std::free(const_cast<struct llama_rs_common_chat_msg_span *>(
        params->message_spans));
  }
  delete params;
}


