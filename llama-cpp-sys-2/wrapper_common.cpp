#include "wrapper_common.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <stdint.h>
#include <string>
#include <vector>

#include "chat.h"
#include "llama.cpp/common/common.h"
#include "llama.cpp/common/fit.h"
#include "llama.cpp/common/json-schema-to-grammar.h"
#include "llama.cpp/common/speculative.h"
#include "llama.cpp/include/llama.h"
#include "wrapper_chat-parser.h"
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

extern "C" llama_rs_status llama_rs_chat_apply_template_with_params(
    const struct llama_model *model, const char *chat_template,
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
    inputs.reasoning_format = common_reasoning_format(params->reasoning_format);
    inputs.continue_final_message =
        common_chat_continuation(params->continue_final_message);

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
    *reinterpret_cast<common_chat_params *>(out_chat_params) = std::move(res);

    return LLAMA_RS_STATUS_OK;
  } catch (...) {
    return LLAMA_RS_STATUS_EXCEPTION;
  }
}

extern "C" struct llama_rs_common_chat_params *
llama_rs_common_chat_params_init() {
  return reinterpret_cast<llama_rs_common_chat_params *>(
      new common_chat_params());
}

extern "C" void
llama_rs_common_chat_params_free(struct llama_rs_common_chat_params *params) {
  if (!params) {
    return;
  }

  delete reinterpret_cast<common_chat_params *>(params);
}

extern "C" struct llama_rs_common_chat_params_view *
llama_rs_common_chat_params_view_init(
    const struct llama_rs_common_chat_params *params) {
  if (!params) {
    return nullptr;
  }

  const common_chat_params *chat_params =
      reinterpret_cast<const common_chat_params *>(params);
  auto *view = new llama_rs_common_chat_params_view();

  llama_rs_common_grammar_trigger *triggers_arr = nullptr;
  if (!chat_params->grammar_triggers.empty()) {
    triggers_arr = static_cast<llama_rs_common_grammar_trigger *>(
        std::malloc(chat_params->grammar_triggers.size() *
                    sizeof(llama_rs_common_grammar_trigger)));
    if (triggers_arr) {
      for (size_t i = 0; i < chat_params->grammar_triggers.size(); ++i) {
        triggers_arr[i].type = llama_rs_common_grammar_trigger_type(
            chat_params->grammar_triggers[i].type);
        triggers_arr[i].value =
            llama_rs_dup_string(chat_params->grammar_triggers[i].value);
        triggers_arr[i].token = chat_params->grammar_triggers[i].token;
      }
    }
  }

  llama_rs_common_chat_msg_delimiter *delimiters_arr = nullptr;
  if (!chat_params->message_delimiters.delimiters.empty()) {
    delimiters_arr = static_cast<llama_rs_common_chat_msg_delimiter *>(
        std::malloc(chat_params->message_delimiters.delimiters.size() *
                    sizeof(llama_rs_common_chat_msg_delimiter)));
    if (delimiters_arr) {
      for (size_t i = 0; i < chat_params->message_delimiters.delimiters.size();
           ++i) {
        delimiters_arr[i].role = llama_rs_common_chat_role(
            chat_params->message_delimiters.delimiters[i].role);
        delimiters_arr[i].delimiter = llama_rs_dup_string(
            chat_params->message_delimiters.delimiters[i].delimiter);
        if (!chat_params->message_delimiters.delimiters[i].tokens.empty()) {
          llama_token *tokens_arr = static_cast<llama_token *>(std::malloc(
              chat_params->message_delimiters.delimiters[i].tokens.size() *
              sizeof(llama_token)));
          if (tokens_arr) {
            std::memcpy(
                tokens_arr,
                chat_params->message_delimiters.delimiters[i].tokens.data(),
                chat_params->message_delimiters.delimiters[i].tokens.size() *
                    sizeof(llama_token));
          }
          delimiters_arr[i].tokens = tokens_arr;
          delimiters_arr[i].n_tokens =
              chat_params->message_delimiters.delimiters[i].tokens.size();
        } else {
          delimiters_arr[i].tokens = nullptr;
          delimiters_arr[i].n_tokens = 0;
        }
      }
    }
  }

  view->format = llama_rs_common_chat_format(chat_params->format);
  view->prompt = llama_rs_dup_string(chat_params->prompt);
  view->grammar = llama_rs_dup_string(chat_params->grammar);
  view->generation_prompt = llama_rs_dup_string(chat_params->generation_prompt);
  view->supports_thinking = chat_params->supports_thinking;
  view->thinking_start_tag =
      llama_rs_dup_string(chat_params->thinking_start_tag);
  view->thinking_end_tag = llama_rs_dup_string(chat_params->thinking_end_tag);
  view->grammar_triggers = triggers_arr;
  view->n_grammar_triggers =
      triggers_arr ? chat_params->grammar_triggers.size() : 0;
  view->preserved_tokens =
      llama_rs_dup_string_vector(chat_params->preserved_tokens);
  view->n_preserved_tokens =
      view->preserved_tokens ? chat_params->preserved_tokens.size() : 0;
  view->additional_stops =
      llama_rs_dup_string_vector(chat_params->additional_stops);
  view->n_additional_stops =
      view->additional_stops ? chat_params->additional_stops.size() : 0;
  view->parser = llama_rs_dup_string(chat_params->parser);
  view->message_delimiters = delimiters_arr;
  view->n_message_delimiters =
      delimiters_arr ? chat_params->message_delimiters.delimiters.size() : 0;

  return view;
}

extern "C" void llama_rs_common_chat_params_view_free(
    struct llama_rs_common_chat_params_view *view) {
  if (!view) {
    return;
  }

  llama_rs_string_free(const_cast<char *>(view->prompt));
  llama_rs_string_free(const_cast<char *>(view->grammar));
  llama_rs_string_free(const_cast<char *>(view->generation_prompt));
  llama_rs_string_free(const_cast<char *>(view->thinking_start_tag));
  llama_rs_string_free(const_cast<char *>(view->thinking_end_tag));
  if (view->grammar_triggers) {
    for (size_t i = 0; i < view->n_grammar_triggers; ++i) {
      llama_rs_string_free(const_cast<char *>(view->grammar_triggers[i].value));
    }
    std::free(const_cast<struct llama_rs_common_grammar_trigger *>(
        view->grammar_triggers));
  }
  if (view->preserved_tokens) {
    for (size_t i = 0; i < view->n_preserved_tokens; ++i) {
      llama_rs_string_free(const_cast<char *>(view->preserved_tokens[i]));
    }
    std::free(view->preserved_tokens);
  }
  if (view->additional_stops) {
    for (size_t i = 0; i < view->n_additional_stops; ++i) {
      llama_rs_string_free(const_cast<char *>(view->additional_stops[i]));
    }
    std::free(view->additional_stops);
  }
  llama_rs_string_free(const_cast<char *>(view->parser));
  if (view->message_delimiters) {
    for (size_t i = 0; i < view->n_message_delimiters; ++i) {
      llama_rs_string_free(
          const_cast<char *>(view->message_delimiters[i].delimiter));
      if (view->message_delimiters[i].tokens) {
        std::free(
            const_cast<llama_token *>(view->message_delimiters[i].tokens));
      }
    }
    std::free(const_cast<struct llama_rs_common_chat_msg_delimiter *>(
        view->message_delimiters));
  }

  delete reinterpret_cast<struct llama_rs_common_chat_params_view *>(view);
}

struct llama_rs_chat_parser {
  common_chat_parser_params params;
  common_chat_msg msg_state;
  std::string generated_text;
};

extern "C" struct llama_rs_chat_parser *llama_rs_chat_parser_init(
    const struct llama_rs_common_chat_params *params,
    const struct llama_rs_chat_template_generation_params *opt) {
  if (!params) {
    return nullptr;
  }

  const common_chat_params *chat_params =
      reinterpret_cast<const common_chat_params *>(params);
  auto *parser = new llama_rs_chat_parser();

  // Apply the input chat params to the parser, as seen here:
  // https://github.com/ggml-org/llama.cpp/blob/e8f19cc0ad70a243c8012bf17b4be601abfc8ea2/tools/server/server-common.cpp#L1035
  parser->params.format = chat_params->format;
  parser->params.reasoning_format =
      common_reasoning_format(opt->reasoning_format);
  parser->params.reasoning_in_content =
      (common_reasoning_format(opt->reasoning_format) ==
       COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY);
  parser->params.generation_prompt = chat_params->generation_prompt;

  if (!chat_params->parser.empty()) {
    try {
      parser->params.parser.load(chat_params->parser);
    } catch (...) {
      delete parser;
      return nullptr;
    }
  }

  parser->params.is_continuation =
      (common_chat_continuation(opt->continue_final_message) !=
       COMMON_CHAT_CONTINUATION_NONE);
  parser->params.echo = false;
  parser->params.debug = false;
  parser->params.parse_tool_calls = true;

  if (parser->params.is_continuation && !parser->params.echo) {
    parser->msg_state = common_chat_parse("", true, parser->params);
  }

  return parser;
}

extern "C" void llama_rs_chat_parser_free(struct llama_rs_chat_parser *parser) {
  delete reinterpret_cast<struct llama_rs_chat_parser *>(parser);
}

extern "C" llama_rs_common_chat_msg_diffs *
llama_rs_common_chat_msg_diffs_init(void) {
  return reinterpret_cast<llama_rs_common_chat_msg_diffs *>(
      new std::vector<struct common_chat_msg_diff>());
}

extern "C" llama_rs_status
llama_rs_chat_parser_feed(struct llama_rs_chat_parser *parser,
                          const char *chunk,
                          llama_rs_common_chat_msg_diffs **out_diffs) {
  if (!parser) {
    return LLAMA_RS_STATUS_INVALID_ARGUMENT;
  }
  try {
    parser->generated_text += chunk;
    common_chat_msg new_state =
        common_chat_parse(parser->generated_text, true, parser->params);
    auto *diffs = new std::vector<struct common_chat_msg_diff>(
        common_chat_msg_diff::compute_diffs(parser->msg_state, new_state));
    parser->msg_state = std::move(new_state);
    *out_diffs =
        reinterpret_cast<struct llama_rs_common_chat_msg_diffs *>(diffs);
    return LLAMA_RS_STATUS_OK;
  } catch (...) {
    return LLAMA_RS_STATUS_EXCEPTION;
  }
}

extern "C" size_t llama_rs_chat_msg_diffs_len(
    const struct llama_rs_common_chat_msg_diffs *diffs) {
  if (!diffs) {
    return 0;
  }
  return reinterpret_cast<const std::vector<struct common_chat_msg_diff> *>(
             diffs)
      ->size();
}

extern "C" llama_rs_status llama_rs_chat_msg_diff_get_view(
    const struct llama_rs_common_chat_msg_diffs *diffs, size_t index,
    struct llama_rs_chat_msg_diff_view *out_view) {
  if (!diffs || !out_view) {
    return LLAMA_RS_STATUS_INVALID_ARGUMENT;
  }
  try {
    auto *msg_diffs =
        reinterpret_cast<const std::vector<struct common_chat_msg_diff> *>(
            diffs);

    if (index >= msg_diffs->size()) {
      return LLAMA_RS_STATUS_EXCEPTION;
    }
    const auto &diff = (*msg_diffs)[index];
    out_view->reasoning_content =
        llama_rs_dup_string(diff.reasoning_content_delta);
    out_view->content = llama_rs_dup_string(diff.content_delta);
    out_view->tool_call_index = diff.tool_call_index;
    out_view->tool_call_name = llama_rs_dup_string(diff.tool_call_delta.name);
    out_view->tool_call_arguments =
        llama_rs_dup_string(diff.tool_call_delta.arguments);
    out_view->tool_call_id = llama_rs_dup_string(diff.tool_call_delta.id);

    return LLAMA_RS_STATUS_OK;
  } catch (...) {
    return LLAMA_RS_STATUS_EXCEPTION;
  }
}

void llama_rs_common_chat_msg_diffs_free(
    struct llama_rs_common_chat_msg_diffs *diffs) {
  delete reinterpret_cast<std::vector<struct common_chat_msg_diff> *>(diffs);
}

extern "C" struct llama_rs_chat_msg_diff_view *
llama_rs_chat_msg_diff_view_init(void) {
  return reinterpret_cast<struct llama_rs_chat_msg_diff_view *>(
      new llama_rs_chat_msg_diff_view());
}

extern "C" void
llama_rs_chat_msg_diff_view_free(struct llama_rs_chat_msg_diff_view *view) {
  if (!view) {
    return;
  }
  llama_rs_string_free(const_cast<char *>(view->reasoning_content));
  llama_rs_string_free(const_cast<char *>(view->content));
  llama_rs_string_free(const_cast<char *>(view->tool_call_name));
  llama_rs_string_free(const_cast<char *>(view->tool_call_arguments));
  llama_rs_string_free(const_cast<char *>(view->tool_call_id));

  delete reinterpret_cast<struct llama_rs_chat_msg_diff_view *>(view);
}