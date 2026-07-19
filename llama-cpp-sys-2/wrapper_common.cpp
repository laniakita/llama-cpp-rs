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

extern "C" struct common_chat_templates_inputs *
common_chat_templates_inputs_create(
    bool add_generation_prompt, bool enable_thinking, int32_t reasoning_format,
    int32_t continue_final_message, bool parallel_tool_calls, bool add_bos,
    bool add_eos, const char *json_schema, const char *grammar,
    const char *extra_context) {
  try {
    auto inputs = new common_chat_templates_inputs();
    inputs->add_generation_prompt = add_generation_prompt;
    inputs->enable_thinking = enable_thinking;
    inputs->reasoning_format =
        static_cast<common_reasoning_format>(reasoning_format);
    inputs->continue_final_message =
        static_cast<common_chat_continuation>(continue_final_message);
    inputs->parallel_tool_calls = parallel_tool_calls;
    inputs->add_bos = add_bos;
    inputs->add_eos = add_eos;

    if (json_schema)
      inputs->json_schema = json_schema;
    if (grammar)
      inputs->grammar = grammar;

    // Parse extra context JSON string back to dict
    if (extra_context) {
      auto parsed = nlohmann::json::parse(extra_context);
      if (parsed.is_object()) {
        for (const auto &[k, v] : parsed.items()) {
          inputs->chat_template_kwargs[k] = v.dump();
        }
      }
    }
    return inputs;
  } catch (...) {
    return nullptr;
  }
}

extern "C" void
common_chat_templates_inputs_free(struct common_chat_templates_inputs *inputs) {
  delete inputs;
}

extern "C" llama_rs_status common_chat_templates_inputs_add_message(
    struct common_chat_templates_inputs *inputs, const char *role,
    const char *content, const char *reasoning_content, const char *tool_name,
    const char *tool_call_id) {
  if (!inputs)
    return LLAMA_RS_STATUS_INVALID_ARGUMENT;
  try {
    common_chat_msg msg;
    if (role)
      msg.role = role;
    if (content)
      msg.content = content;
    if (reasoning_content)
      msg.reasoning_content = reasoning_content;
    if (tool_name)
      msg.tool_name = tool_name;
    if (tool_call_id)
      msg.tool_call_id = tool_call_id;
    inputs->messages.push_back(std::move(msg));
    return LLAMA_RS_STATUS_OK;
  } catch (...) {
    return LLAMA_RS_STATUS_EXCEPTION;
  }
}

// Appends tool calls safely to the back of the vector, avoiding FFI pointer
// invalidation
extern "C" llama_rs_status
common_chat_templates_inputs_add_tool_call_to_last_message(
    struct common_chat_templates_inputs *inputs, const char *name,
    const char *arguments, const char *id) {
  if (!inputs || inputs->messages.empty())
    return LLAMA_RS_STATUS_INVALID_ARGUMENT;
  try {
    common_chat_tool_call call;
    if (name)
      call.name = name;
    if (arguments)
      call.arguments = arguments;
    if (id)
      call.id = id;
    inputs->messages.back().tool_calls.push_back(std::move(call));
    return LLAMA_RS_STATUS_OK;
  } catch (...) {
    return LLAMA_RS_STATUS_EXCEPTION;
  }
}

extern "C" llama_rs_status common_chat_templates_inputs_add_tool(
    struct common_chat_templates_inputs *inputs, const char *name,
    const char *description, const char *parameters) {
  if (!inputs)
    return LLAMA_RS_STATUS_INVALID_ARGUMENT;
  try {
    common_chat_tool tool;
    if (name)
      tool.name = name;
    if (description)
      tool.description = description;
    if (parameters)
      tool.parameters = parameters;
    inputs->tools.push_back(std::move(tool));
    return LLAMA_RS_STATUS_OK;
  } catch (...) {
    return LLAMA_RS_STATUS_EXCEPTION;
  }
}

extern "C" struct common_chat_params *
common_chat_apply_template(const struct llama_model *model,
                           const char *chat_template,
                           const struct common_chat_templates_inputs *inputs) {
  if (!inputs)
    return nullptr;
  try {
    auto tmpls = common_chat_templates_init(
        model, chat_template ? chat_template : "", "", "");
    auto res = common_chat_templates_apply(tmpls.get(), *inputs);
    return new common_chat_params(std::move(res));
  } catch (...) {
    return nullptr;
  }
}

extern "C" void common_chat_params_free(struct common_chat_params *params) {
  delete params;
}

extern "C" struct common_chat_params_view
common_chat_params_get_view(const struct common_chat_params *params) {
  struct common_chat_params_view view = {0};
  if (!params)
    return view;

  view.format = static_cast<int32_t>(params->format);
  view.prompt = params->prompt.c_str();
  view.grammar = params->grammar.c_str();
  view.grammar_lazy = params->grammar_lazy;
  view.generation_prompt = params->generation_prompt.c_str();
  view.supports_thinking = params->supports_thinking;
  view.thinking_start_tag = params->thinking_start_tag.c_str();
  view.thinking_end_tag = params->thinking_end_tag.c_str();
  view.parser = params->parser.c_str();

  return view; // Returned by value across FFI
}

extern "C" size_t common_chat_params_get_grammar_triggers_count(
    const struct common_chat_params *params) {
  return params ? params->grammar_triggers.size() : 0;
}

extern "C" struct common_grammar_trigger_view
common_chat_params_get_grammar_trigger(const struct common_chat_params *params,
                                       size_t index) {
  struct common_grammar_trigger_view view = {0};
  if (!params || index >= params->grammar_triggers.size())
    return view;

  const auto &trigger = params->grammar_triggers[index];
  view.type = static_cast<int32_t>(trigger.type);
  view.value = trigger.value.c_str();
  view.token = trigger.token;

  return view;
}

extern "C" size_t common_chat_params_get_message_delimiters_count(
    const struct common_chat_params *params) {
  return params ? params->message_delimiters.delimiters.size() : 0;
}

extern "C" struct common_chat_msg_delimiter_view
common_chat_params_get_message_delimiter(
    const struct common_chat_params *params, size_t index) {
  struct common_chat_msg_delimiter_view view = {0};
  if (!params || index >= params->message_delimiters.delimiters.size())
    return view;

  const auto &delim = params->message_delimiters.delimiters[index];
  view.role = static_cast<int32_t>(delim.role);
  view.delimiter = delim.delimiter.c_str();
  view.tokens = delim.tokens.empty() ? nullptr : delim.tokens.data();
  view.tokens_count = delim.tokens.size();

  return view;
}

extern "C" size_t common_chat_params_get_preserved_tokens_count(
    const struct common_chat_params *params) {
  return params ? params->preserved_tokens.size() : 0;
}

extern "C" const char *
common_chat_params_get_preserved_token(const struct common_chat_params *params,
                                       size_t index) {
  if (!params || index >= params->preserved_tokens.size())
    return nullptr;
  return params->preserved_tokens[index].c_str();
}

struct llama_rs_chat_parser {
  common_chat_parser_params params;
  common_chat_msg msg_state;
  std::string generated_text;
};

extern "C" struct llama_rs_chat_parser *
llama_rs_chat_parser_init(const struct common_chat_params *params,
                          const struct common_chat_templates_inputs *inputs) {
  if (!params || !inputs) {
    return nullptr;
  }

  auto *parser = new llama_rs_chat_parser();

  parser->params.format = params->format;
  parser->params.reasoning_format = inputs->reasoning_format;
  parser->params.reasoning_in_content =
      (inputs->reasoning_format == COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY);
  parser->params.generation_prompt = params->generation_prompt;

  if (!params->parser.empty()) {
    try {
      parser->params.parser.load(params->parser);
    } catch (...) {
      delete parser;
      return nullptr;
    }
  }

  parser->params.is_continuation =
      (inputs->continue_final_message != COMMON_CHAT_CONTINUATION_NONE);
  parser->params.echo = false;
  parser->params.debug = false;
  parser->params.parse_tool_calls = true;

  if (parser->params.is_continuation && !parser->params.echo) {
    parser->msg_state = common_chat_parse("", true, parser->params);
  }

  return parser;
}

extern "C" void llama_rs_chat_parser_free(struct llama_rs_chat_parser *parser) {
  delete parser;
}

extern "C" llama_rs_status
llama_rs_chat_parser_feed(struct llama_rs_chat_parser *parser,
                          const char *chunk, bool is_partial,
                          struct common_chat_msg_diffs **out_diffs) {
  if (!parser || !out_diffs) {
    return LLAMA_RS_STATUS_INVALID_ARGUMENT;
  }
  try {
    parser->generated_text += chunk;
    common_chat_msg new_state =
        common_chat_parse(parser->generated_text, is_partial, parser->params);
    auto *diffs = new std::vector<struct common_chat_msg_diff>(
        common_chat_msg_diff::compute_diffs(parser->msg_state, new_state));
    parser->msg_state = std::move(new_state);
    *out_diffs = reinterpret_cast<struct common_chat_msg_diffs *>(diffs);
    return LLAMA_RS_STATUS_OK;
  } catch (...) {
    return LLAMA_RS_STATUS_EXCEPTION;
  }
}

extern "C" void
common_chat_msg_diffs_free(struct common_chat_msg_diffs *diffs) {
  delete reinterpret_cast<std::vector<struct common_chat_msg_diff> *>(diffs);
}

extern "C" size_t
common_chat_msg_diffs_get_size(const struct common_chat_msg_diffs *diffs) {
  if (!diffs)
    return 0;
  return reinterpret_cast<const std::vector<struct common_chat_msg_diff> *>(
             diffs)
      ->size();
}

extern "C" struct common_chat_msg_diff_view
common_chat_msg_diffs_get_view(const struct common_chat_msg_diffs *diffs,
                               size_t index) {
  struct common_chat_msg_diff_view view = {0};
  if (!diffs)
    return view;

  auto *msg_diffs =
      reinterpret_cast<const std::vector<struct common_chat_msg_diff> *>(diffs);
  if (index >= msg_diffs->size())
    return view;

  const auto &diff = (*msg_diffs)[index];
  view.reasoning_content = diff.reasoning_content_delta.c_str();
  view.content = diff.content_delta.c_str();
  view.tool_call_index = diff.tool_call_index;
  view.tool_call_name = diff.tool_call_delta.name.c_str();
  view.tool_call_arguments = diff.tool_call_delta.arguments.c_str();
  view.tool_call_id = diff.tool_call_delta.id.c_str();

  return view;
}