//! Based on the mtmd cli example from llama.cpp.

mod tools;

use std::borrow::Cow;
use std::ffi::CString;
use std::io::{self, Write};
use std::num::NonZeroU32;
use std::path::Path;

use clap::Parser;
use encoding_rs::{Decoder, UTF_8};

use llama_cpp_2::chat_parser::{
    ChatDiff, ChatParser, LlamaChatMessageDelimiter, LlamaChatParamsView, LlamaGenerationParams,
    LlamaGrammarTriggerType,
};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::mtmd::{
    MtmdBitmap, MtmdBitmapError, MtmdContext, MtmdContextParams, MtmdInputText,
};

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::{LlamaChatMessage, LlamaChatTool, LlamaChatToolCall, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::speculative::{MtpSpeculative, MtpSpeculativeParams};
use llama_cpp_2::token::data::LlamaTokenData;
use llama_cpp_2::token::data_array::LlamaTokenDataArray;
use llama_cpp_2::token::LlamaToken;

/// Command line parameters for the ChatParser CLI application
#[derive(clap::Parser, Debug)]
#[command(name = "chat-parser-cli")]
#[command(about = "Experimental CLI for multimodal llama.cpp")]
pub struct ChatParserCliParams {
    /// Path to the model file
    #[arg(short = 'm', long = "model", value_name = "PATH")]
    pub model_path: String,
    /// Path to the multimodal projection file
    #[arg(long = "mmproj", value_name = "PATH")]
    pub mmproj_path: String,

    /// Path to the MTP draft model file
    #[arg(long = "mtp", value_name = "PATH")]
    pub mtp_path: Option<String>,

    /// Path to image file(s)
    #[arg(long = "image", value_name = "PATH")]
    pub images: Vec<String>,
    /// Path to audio file(s)
    #[arg(long = "audio", value_name = "PATH")]
    pub audio: Vec<String>,
    /// Text prompt to use as input to the model. May include media markers - else they will be added automatically.
    #[arg(short = 'p', long = "prompt", value_name = "TEXT")]
    pub prompt: String,
    /// Number of tokens to predict (-1 for unlimited)
    #[arg(
        short = 'n',
        long = "n-predict",
        value_name = "N",
        default_value = "-1"
    )]
    pub n_predict: i32,
    /// Number of threads
    #[arg(short = 't', long = "threads", value_name = "N", default_value = "4")]
    pub n_threads: i32,
    /// Number of tokens to process in a batch during eval chunks
    #[arg(long = "batch-size", value_name = "b", default_value = "512")]
    pub batch_size: i32,
    /// Maximum number of tokens in context
    #[arg(long = "n-tokens", value_name = "N", default_value = "4096")]
    pub n_tokens: NonZeroU32,
    /// Chat template to use, default template if not provided
    #[arg(long = "chat-template", value_name = "TEMPLATE")]
    pub chat_template: Option<String>,

    /// Enable thinking
    #[arg(
        long = "enable-thinking",
        value_name = "ENABLE_THINKING",
        default_value = "true"
    )]
    pub enable_thinking: bool,

    /// Disable GPU acceleration
    #[arg(long = "no-gpu")]
    pub no_gpu: bool,
    /// Disable GPU offload for multimodal projection
    #[arg(long = "no-mmproj-offload")]
    pub no_mmproj_offload: bool,
    /// Media marker. If not provided, the default marker will be used.
    #[arg(long = "marker", value_name = "TEXT")]
    pub media_marker: Option<String>,
    /// Minimum number of tokens used to represent an image (-1 for model default).
    #[arg(long = "image-min-tokens", value_name = "N", default_value = "-1")]
    pub image_min_tokens: i32,
    /// Maximum number of tokens used to represent an image (-1 for model default).
    #[arg(long = "image-max-tokens", value_name = "N", default_value = "-1")]
    pub image_max_tokens: i32,
}

/// State of the CLI application.
#[allow(missing_debug_implementations)]
pub struct ChatParserCliContext<'a> {
    /// The MTMD context for multimodal processing.
    pub mtmd_ctx: Option<MtmdContext>,

    /// The list of loaded bitmaps (images/audio).
    pub bitmaps: Vec<MtmdBitmap>,

    /// The batch used for processing tokens.
    pub batch: Option<LlamaBatch<'a>>,
}

impl<'a> ChatParserCliContext<'a> {
    /// Creates a new ChatParser CLI context
    ///
    /// # Errors
    pub fn new(
        params: &ChatParserCliParams,
        model: &'a LlamaModel,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Initialize MTMD context
        let mtmd_params = MtmdContextParams {
            use_gpu: !params.no_gpu && !params.no_mmproj_offload,
            print_timings: true,
            n_threads: params.n_threads,
            media_marker: CString::new(
                params
                    .media_marker
                    .as_ref()
                    .unwrap_or(&llama_cpp_2::mtmd::mtmd_default_marker().to_string())
                    .clone(),
            )?,
            image_min_tokens: params.image_min_tokens,
            image_max_tokens: params.image_max_tokens,
        };

        let mtmd_ctx = MtmdContext::init_from_file(&params.mmproj_path, model, &mtmd_params)?;

        let batch = LlamaBatch::new(params.n_tokens.get() as usize, 1);

        Ok(Self {
            mtmd_ctx: Some(mtmd_ctx),
            batch: Some(batch),
            bitmaps: Vec::new(),
        })
    }

    /// Loads media (image or audio) from the specified file path
    /// # Errors
    pub fn load_media(&mut self, path: &str) -> Result<(), MtmdBitmapError> {
        if let Some(mtmd_ctx) = self.mtmd_ctx.as_ref() {
            let bitmap = MtmdBitmap::from_file(mtmd_ctx, path, false)?;
            self.bitmaps.push(bitmap);
        }
        Ok(())
    }

    /// Runs a single turn using the chat model.
    pub fn run_single_turn(
        &mut self,
        model: &'a LlamaModel,
        context: LlamaContext<'a>,
        mtp_ctx: Option<LlamaContext<'a>>,
        params: &ChatParserCliParams,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let t_initial = std::time::Instant::now();
        let start_timestamp = chrono::Utc::now().to_rfc3339();

        let mut prompt = params.prompt.clone();

        let default_marker = llama_cpp_2::mtmd::mtmd_default_marker().to_string();
        let media_marker = params.media_marker.as_ref().unwrap_or(&default_marker);
        let total_media_files = params.images.len() + params.audio.len();
        let current_markers = prompt.matches(media_marker).count();

        if current_markers < total_media_files {
            for _ in 0..(total_media_files - current_markers) {
                if !prompt.is_empty() && !prompt.ends_with(' ') && !prompt.ends_with('\n') {
                    prompt.push('\n');
                }
                prompt.push_str(media_marker);
            }
        } else if current_markers > total_media_files {
            eprintln!(
                "Warning: Prompt contains {} media markers, but only {} media files were provided.",
                current_markers, total_media_files
            );
        }

        for image_path in &params.images {
            println!("Loading image: {image_path}");
            self.load_media(image_path)?;
        }
        for audio_path in &params.audio {
            self.load_media(audio_path)?;
        }

        let mut next_turn_delta = String::new();

        let mut history = vec![
            LlamaChatMessage::new("system".into(), "You are a helpful assistant.".into())?,
            LlamaChatMessage::new("user".into(), prompt)?,
        ];

        let bitmap_refs: &[&MtmdBitmap] = &self.bitmaps.iter().map(|b| b).collect::<Vec<_>>();

        let tools = vec![
            crate::tools::weather::get_geocode_tool_definition()?,
            crate::tools::weather::get_weather_tool_definition()?,
        ];

        let active_context = match mtp_ctx {
            Some(m_ctx) => ActiveContext::Mtp(MtpSpeculative::new(
                context,
                m_ctx,
                MtpSpeculativeParams {
                    n_max: 2,
                    n_min: 0,
                    p_min: 0.85,
                    n_seq: 1,
                },
            )?),
            None => ActiveContext::Standard(context),
        };

        let batch = self.batch.take().ok_or("No batch available")?;
        let mtmd_ctx = self.mtmd_ctx.take().ok_or("No mtmd_ctx available")?;

        let mut session = ChatSession::new(params.n_predict, batch, active_context, mtmd_ctx);

        let mut responses: Vec<Response> = Vec::new();

        let model_arch = model
            .meta_val_str("general.architecture")
            .unwrap_or_else(|_| "Unknown".to_string());
        let model_name = model
            .meta_val_str("general.name")
            .unwrap_or_else(|_| "Unknown".to_string());

        let (using_mtp, mtp_model_arch, mtp_model_name) = match &mut session.active_context {
            ActiveContext::Mtp(ref mut mtp) => {
                let name = mtp
                    .draft_context_mut()
                    .model
                    .meta_val_str("general.name")
                    .unwrap_or_else(|_| "Unknown".to_string());
                let arch = mtp
                    .draft_context_mut()
                    .model
                    .meta_val_str("general.architecture")
                    .unwrap_or_else(|_| "Unknown".to_string());
                (true, arch, name)
            }
            ActiveContext::Standard(_) => (false, "N/A".to_string(), "N/A".to_string()),
        };

        loop {
            println!("\n--- [TURN {}] ---", history.len() / 2);
            let turn_start_time = std::time::Instant::now();

            let (chat_params_view, sampler, parser, prompt_tokens, gen_params) = session
                .apply_turn(
                    model,
                    &history,
                    &next_turn_delta,
                    &bitmap_refs,
                    &tools,
                    true,
                    params.batch_size,
                )?;

            let mut turn =
                ChatTurn::new(sampler, parser, model, &chat_params_view.message_delimiters);

            let n_past_before_generation = session.n_past;

            let (assistant_msg, response) = match &session.active_context {
                ActiveContext::Mtp(_) => {
                    turn.generate_with_mtp(&mut session, &prompt_tokens, turn_start_time)?
                }
                ActiveContext::Standard(_) => {
                    turn.generate_classic(&mut session, turn_start_time)?
                }
            };
            responses.push(response);

            let mut turn_delta = String::new();
            turn_delta.push_str(&model.chat_format_single(
                None,
                &gen_params,
                &assistant_msg,
                false,
                true,
            )?);

            let tool_calls = assistant_msg.tool_calls().to_vec();
            history.push(assistant_msg);

            if tool_calls.is_empty() {
                println!("\n[FINISHED] Assistant did not make any more tool calls.");
                break;
            }

            // Roll back the context so we don't duplicate the assistant message
            // when it's re-evaluated via turn_delta in the next iteration.
            match &mut session.active_context {
                ActiveContext::Mtp(mtp) => {
                    mtp.target_context_mut().kv_cache_seq_rm(
                        0,
                        Some(n_past_before_generation as u32),
                        None,
                    )?;
                    mtp.draft_context_mut().kv_cache_seq_rm(
                        0,
                        Some(n_past_before_generation as u32),
                        None,
                    )?;
                }
                ActiveContext::Standard(ctx) => {
                    ctx.kv_cache_seq_rm(0, Some(n_past_before_generation as u32), None)?;
                }
            }
            session.n_past = n_past_before_generation;

            for tool_call in tool_calls {
                let result = if tool_call.name() == "geocode_city" {
                    crate::tools::weather::execute_geocode(&tool_call.arguments())
                } else if tool_call.name() == "get_current_weather" {
                    crate::tools::weather::execute_weather(&tool_call.arguments())
                } else {
                    format!("Unknown tool: {}", tool_call.name())
                };

                let tool_msg = LlamaChatMessage::new("tool".into(), result.to_string())?
                    .with_tool_call_id(tool_call.id().to_string())?
                    .with_tool_name(tool_call.name().to_string())?;
                let new_params = &gen_params.clone().with_messages(&history);

                turn_delta.push_str(&model.chat_format_single(
                    None,
                    &new_params,
                    &tool_msg,
                    true,
                    true,
                )?);
                history.push(tool_msg);
            }

            next_turn_delta = turn_delta;
        }
        let elapsed = t_initial.elapsed().as_secs_f64();
        let end_timestamp = responses
            .last()
            .map_or_else(|| chrono::Utc::now(), |l| l.finished_at)
            .to_rfc3339();

        let total_tool_calls = history.iter().filter(|m| m.role() == "tool").count();
        let tokens_per_second_avg = responses.iter().map(|r| r.tokens_per_second).sum::<f32>()
            as f64
            / responses.len() as f64;
        let total_tokens = responses.iter().map(|r| r.tokens_generated).sum::<usize>();
        let tokens_per_second_actual = total_tokens as f64 / elapsed;

        let ttft_sum: f64 = responses.iter().map(|r| r.time_to_first_token).sum();
        let ttft_avg = if !responses.is_empty() {
            ttft_sum / responses.len() as f64
        } else {
            0.0
        };

        println!(
            r"--- Final Chat History State ---
RESPONSES: {responses:#?}
CHAT HISTORY:{history:#?}
----------- METRICS ------------
MODEL: {model_arch}:{model_name}
MULTIMODAL: true
MTP_ENABLED: {using_mtp}
MTP_MODEL: {mtp_model_arch}:{mtp_model_name}
START_TIME: {start_timestamp}
END_TIME: {end_timestamp}
TOTAL_RUN_TIME: {elapsed:.2} seconds
TOTAL_TOKENS: {total_tokens}
TOTAL_TOOL_CALLS: {total_tool_calls}
AVERAGE_TOKENS_PER_SECOND: {tokens_per_second_avg}
ACTUAL_TOKENS_PER_SECOND: {tokens_per_second_actual}
SUM_TIME_TO_FIRST_TOKEN: {ttft_sum:.4} seconds
AVERAGE_TIME_TO_FIRST_TOKEN: {ttft_avg:.4} seconds
--------------------------------"
        );

        Ok(())
    }
}

enum ActiveContext<'a> {
    Standard(LlamaContext<'a>),
    Mtp(MtpSpeculative<'a>),
}

struct CommonSampler {
    pub chain: LlamaSampler,
    pub grmr: Option<LlamaSampler>,
}

impl CommonSampler {
    fn new(
        model: &LlamaModel,
        params: &LlamaChatParamsView,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let grammar_str = params.grammar.to_string_lossy().into_owned();

        let mut trigger_patterns: Vec<String> = Vec::new();
        let mut trigger_tokens: Vec<LlamaToken> = Vec::new();

        params
            .grammar_triggers
            .iter()
            .for_each(|trigger| match trigger.trigger_type {
                LlamaGrammarTriggerType::Token => trigger_tokens.push(trigger.token),
                LlamaGrammarTriggerType::Word => {
                    trigger_patterns.push(regex::escape(&trigger.value.to_string_lossy()))
                }
                LlamaGrammarTriggerType::Pattern => {
                    trigger_patterns.push(trigger.value.to_string_lossy().into());
                }
                LlamaGrammarTriggerType::PatternFull => {
                    let pattern = trigger.value.to_string_lossy().to_string();
                    if !pattern.is_empty() {
                        let anchored_parts = &[
                            (if !pattern.starts_with("^") { "^" } else { "" }),
                            &pattern,
                            (if !pattern.ends_with("$") { "$" } else { "" }),
                        ];
                        let anchored = anchored_parts.join("");
                        trigger_patterns.push(anchored);
                    }
                }
            });

        let grmr: Option<LlamaSampler> = if !grammar_str.is_empty() {
            if params.grammar_lazy {
                Some(LlamaSampler::grammar_lazy_patterns(
                    model,
                    &grammar_str,
                    "root",
                    &trigger_patterns,
                    &trigger_tokens,
                )?)
            } else {
                unimplemented!("Non-lazy grammar is not implemented for this example");
            }
        } else {
            None
        };

        // Compute prefill tokens from the generation prompt
        let mut prefill_tokens: Vec<LlamaToken> = Vec::new();
        if !params.generation_prompt.is_empty() {
            let generation_prompt = params.generation_prompt.to_string_lossy();
            let mut decoder = UTF_8.new_decoder();
            if let Ok(tokens) =
                model.str_to_token(&generation_prompt, llama_cpp_2::model::AddBos::Never)
            {
                for (i, &token) in tokens.iter().enumerate() {
                    let piece = model.token_to_piece(token, &mut decoder, true, None)?;
                    if i == 0 {
                        // Check if the tokenizer inappropriately added a leading space to the first special token
                        if piece.starts_with(" ") && !generation_prompt.starts_with(" ") {
                            // Some tokenizers will add a space before the first special token,
                            continue;
                        }
                    }
                    //println!("eval_message: prefill token: {} = {}", token, piece);
                    prefill_tokens.push(token);
                }
            }
        }

        let chain = LlamaSampler::chain_simple([LlamaSampler::greedy()]);

        Ok(Self { grmr, chain })
    }

    fn sample<'a>(
        &mut self,
        ctx: &LlamaContext<'a>,
        idx: i32,
    ) -> Result<LlamaToken, Box<dyn std::error::Error>> {
        let mut cur_p = LlamaTokenDataArray::from_iter(ctx.candidates_ith(idx), false);
        self.chain.apply(&mut cur_p);
        let id = cur_p.selected_token().ok_or("Failed to select token")?;

        // Grammar Rejection Sampling
        if let Some(grmr) = &mut self.grmr {
            let mut single_array =
                LlamaTokenDataArray::new(vec![LlamaTokenData::new(id, 1.0, 0.0)], false);

            grmr.apply(&mut single_array);
            if single_array.data[0].logit() != f32::NEG_INFINITY {
                return Ok(id);
            }

            // If the first token is rejected, try again with the rest of the tokens
            let mut fallback_p = LlamaTokenDataArray::from_iter(ctx.candidates_ith(idx), false);

            grmr.apply(&mut fallback_p);
            self.chain.apply(&mut fallback_p);
            return Ok(fallback_p
                .selected_token()
                .ok_or("Failed to get selected token")?);
        }

        Ok(id)
    }

    fn accept(&mut self, token: LlamaToken) {
        if let Some(grmr) = &mut self.grmr {
            grmr.accept(token);
        }
        self.chain.accept(token);
    }
}

struct ChatSession<'a> {
    pub n_predict: i32,
    pub n_past: i32,
    pub chunk_offset: usize,
    pub batch: LlamaBatch<'a>,
    pub active_context: ActiveContext<'a>,
    pub mtmd_context: MtmdContext,
}

impl<'a> ChatSession<'a> {
    pub fn new(
        n_predict: i32,
        batch: LlamaBatch<'a>,
        active_context: ActiveContext<'a>,
        mtmd_context: MtmdContext,
    ) -> Self {
        Self {
            n_predict,
            n_past: 0,
            batch,
            active_context,
            chunk_offset: 0,
            mtmd_context,
        }
    }

    pub fn apply_turn(
        &mut self,
        model: &LlamaModel,
        history: &[LlamaChatMessage],
        delta: &str,
        bitmaps: &[&MtmdBitmap],
        tools: &[LlamaChatTool],
        add_bos: bool,
        batch_size: i32,
    ) -> Result<
        (
            LlamaChatParamsView,
            CommonSampler,
            ChatParser,
            Vec<LlamaToken>,
            LlamaGenerationParams,
        ),
        Box<dyn std::error::Error>,
    > {
        let gen_params = LlamaGenerationParams::default()
            .with_add_generation_prompt(true)
            .with_enable_thinking(true)
            .with_messages(history)
            .with_tools(tools)
            .with_add_bos(add_bos);

        println!("Generation params: {:#?}", gen_params);

        // Format the message using chat template (simplified)
        let chat_params = model
            .apply_chat_template_with_params(None, &gen_params)
            .map_err(|e| format!("Failed to apply chat template: {e}"))?;
        let chat_params_view = chat_params.view();

        println!("Chat params: {:#?}", &chat_params_view);

        let common_sampler = CommonSampler::new(model, &chat_params_view)?;
        let chat_parser = ChatParser::new(&chat_params, &gen_params)?;

        if bitmaps.is_empty() {
            println!("No bitmaps provided, only tokenizing text");
        } else {
            println!("Tokenizing with {} bitmaps", bitmaps.len());
        }

        self.chunk_offset = 0;

        let input_text = MtmdInputText {
            text: if !delta.is_empty() {
                delta.to_string()
            } else {
                chat_params_view.prompt.to_string_lossy().to_string()
            },
            add_special: add_bos,
            parse_special: true,
        };

        let mut prompt_tokens = Vec::new();
        let chunks = self.mtmd_context.tokenize(input_text, bitmaps)?;
        for i in self.chunk_offset..chunks.len() {
            if let Some(chunk) = chunks.get(i) {
                let active_ctx = match &mut self.active_context {
                    ActiveContext::Standard(ctx) => ctx,
                    ActiveContext::Mtp(ctx) => ctx.target_context_mut(),
                };

                self.n_past = chunk.eval_chunk_single(
                    &self.mtmd_context,
                    active_ctx,
                    self.n_past,
                    0,
                    batch_size,
                    true,
                )?;
                active_ctx.clear_and_mark_logit(-1);
                if let Some(tokens) = chunk.text_tokens() {
                    prompt_tokens.extend_from_slice(tokens);
                }
            }
        }
        self.chunk_offset += chunks.len();

        Ok((
            chat_params_view,
            common_sampler,
            chat_parser,
            prompt_tokens,
            gen_params,
        ))
    }
}

struct ChatTurn<'a> {
    sampler: CommonSampler,
    parser: ChatParser,
    model: &'a LlamaModel,
    delimiters: Vec<LlamaChatMessageDelimiter>,
}

impl<'a> ChatTurn<'a> {
    /// Creates a new ChatTurn
    pub fn new(
        sampler: CommonSampler,
        parser: ChatParser,
        model: &'a LlamaModel,
        delimiters: &[LlamaChatMessageDelimiter],
    ) -> Self {
        Self {
            sampler,
            parser,
            model,
            delimiters: delimiters.to_vec(),
        }
    }

    /// Generates a response by sampling tokens from the model
    fn generate_classic(
        &mut self,
        session: &mut ChatSession<'a>,
        turn_start_time: std::time::Instant,
    ) -> Result<(LlamaChatMessage, Response), Box<dyn std::error::Error>> {
        let t_initial = std::time::Instant::now();

        let session_ctx = match &mut session.active_context {
            ActiveContext::Standard(ctx) => ctx,
            _ => return Err("Not in standard context".into()),
        };

        let mut generated_tokens = Vec::new();
        let max_predict = if session.n_predict < 0 {
            i32::MAX
        } else {
            session.n_predict
        };
        let mut decoder = UTF_8.new_decoder();
        let mut response = Response::default();

        let mut sample_idx = -1;
        for _i in 0..max_predict {
            let token = self.sampler.sample(&session_ctx, sample_idx)?;
            self.sampler.accept(token);

            if generated_tokens.is_empty() {
                response.time_to_first_token = turn_start_time.elapsed().as_secs_f64();
            }

            if self.should_stop(&generated_tokens) || token == self.model.token_eos() {
                if let Ok(final_diffs) = self.parser.finish() {
                    Self::handle_diffs(&final_diffs, &mut response);
                }
                break;
            }

            Self::handle_token(
                token,
                &mut decoder,
                &mut self.parser,
                &mut response,
                &mut generated_tokens,
                &self.model,
            )?;

            // Prepare next batch
            session.batch.clear();
            session.batch.add(token, session.n_past, &[0], true)?;
            session.n_past += 1;

            // Decode
            session_ctx.decode(&mut session.batch)?;
            sample_idx = 0;
        }
        let t_final = t_initial.elapsed();
        response.finished_at = chrono::Utc::now();
        response.tokens_per_second = response.tokens_generated as f32 / t_final.as_secs_f32();

        let mut assistant_msg =
            LlamaChatMessage::new("assistant".into(), response.content.clone())?;
        if !response.reasoning.is_empty() {
            assistant_msg = assistant_msg.with_reasoning_content(response.reasoning.clone())?;
        }
        if !response.tool_calls.is_empty() {
            assistant_msg = assistant_msg.with_tool_calls(
                &response
                    .tool_calls
                    .iter()
                    .map(|tc| {
                        LlamaChatToolCall::new(tc.name.clone(), tc.arguments.clone(), tc.id.clone())
                            .expect("could get tool call")
                    })
                    .collect::<Vec<LlamaChatToolCall>>(),
            );
        }

        Ok((assistant_msg, response))
    }

    /// Generates a response by sampling tokens from the model with a Multi-Token_Prediction companion model.
    fn generate_with_mtp(
        &mut self,
        session: &mut ChatSession<'a>,
        prompt_tokens: &[LlamaToken],
        turn_start_time: std::time::Instant,
    ) -> Result<(LlamaChatMessage, Response), Box<dyn std::error::Error>> {
        let t_initial = std::time::Instant::now();

        let max_predict = if session.n_predict < 0 {
            i32::MAX
        } else {
            session.n_predict
        };
        let mut decoder = UTF_8.new_decoder();

        let mtp = match &mut session.active_context {
            ActiveContext::Mtp(ctx) => ctx,
            _ => return Err("Not in mtp context".into()),
        };

        let mut id_last = self.sampler.sample(mtp.target_context(), -1)?;
        let ttft = turn_start_time.elapsed().as_secs_f64();
        self.sampler.accept(id_last);

        mtp.begin(prompt_tokens, 0)
            .map_err(|err| format!("Failed to start speculative decoding: {err:#?}"))?;
        println!("mtp loaded");

        let mut response = Response::default();
        response.time_to_first_token = ttft;

        let mut generated_tokens = Vec::new();

        for _ in 0..max_predict {
            if self.should_stop(&generated_tokens) || id_last == self.model.token_eos() {
                if let Ok(final_diffs) = self.parser.finish() {
                    Self::handle_diffs(&final_diffs, &mut response);
                }
                break;
            }

            Self::handle_token(
                id_last,
                &mut decoder,
                &mut self.parser,
                &mut response,
                &mut generated_tokens,
                &self.model,
            )?;

            let draft_tokens = mtp.draft(session.n_past, id_last, &[], 0)?;

            session.batch.clear();

            let mut batch_n_past = session.n_past;
            session.batch.add(id_last, batch_n_past, &[0], true)?;
            batch_n_past += 1;

            for tk in &draft_tokens {
                session.batch.add(*tk, batch_n_past, &[0], true)?;
                batch_n_past += 1;
            }

            mtp.target_context_mut().decode(&mut session.batch)?;

            let mut n_accepted = 0;
            let mut new_id_last = None;

            for i in 0..draft_tokens.len() {
                let target_sampled_token = self.sampler.sample(mtp.target_context(), i as i32)?;
                if target_sampled_token == draft_tokens[i] {
                    n_accepted += 1;
                    self.sampler.accept(target_sampled_token);

                    Self::handle_token(
                        target_sampled_token,
                        &mut decoder,
                        &mut self.parser,
                        &mut response,
                        &mut generated_tokens,
                        &self.model,
                    )?;
                } else {
                    new_id_last = Some(target_sampled_token);
                    self.sampler.accept(target_sampled_token);
                    break;
                }
            }

            if let Some(new_id) = new_id_last {
                id_last = new_id;
            } else {
                let last_sampled_token = self
                    .sampler
                    .sample(mtp.target_context(), draft_tokens.len() as i32)?;
                new_id_last.replace(last_sampled_token);
                self.sampler.accept(last_sampled_token);
                id_last = last_sampled_token;
            }

            if !draft_tokens.is_empty() {
                mtp.accept(n_accepted, 0)?;
                let tokens_to_keep_until = session.n_past + 1 + n_accepted as i32;
                if (n_accepted as usize) < draft_tokens.len() {
                    mtp.target_context_mut().kv_cache_seq_rm(
                        0,
                        Some(tokens_to_keep_until as u32),
                        None,
                    )?;
                    mtp.draft_context_mut().kv_cache_seq_rm(
                        0,
                        Some(tokens_to_keep_until as u32),
                        None,
                    )?;
                }

                session.n_past = tokens_to_keep_until;
            } else {
                session.n_past += 1;
            }
        }
        let t_final = t_initial.elapsed();
        response.finished_at = chrono::Utc::now();
        response.tokens_per_second = response.tokens_generated as f32 / t_final.as_secs_f32();

        println!("{response:#?}");

        let mut assistant_msg =
            LlamaChatMessage::new("assistant".into(), response.content.clone())?;
        if !response.reasoning.is_empty() {
            assistant_msg = assistant_msg.with_reasoning_content(response.reasoning.clone())?;
        }
        if !response.tool_calls.is_empty() {
            assistant_msg = assistant_msg.with_tool_calls(
                &response
                    .tool_calls
                    .iter()
                    .map(|tc| {
                        LlamaChatToolCall::new(tc.name.clone(), tc.arguments.clone(), tc.id.clone())
                            .expect("could get tool call")
                    })
                    .collect::<Vec<LlamaChatToolCall>>(),
            );
        }

        Ok((assistant_msg, response))
    }

    /// Helper function to process a single token and update a response.
    fn handle_token(
        token: LlamaToken,
        dcdr: &mut Decoder,
        parser: &mut ChatParser,
        res: &mut Response,
        generated_tokens: &mut Vec<LlamaToken>,
        model: &LlamaModel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        res.tokens_generated += 1;
        generated_tokens.push(token);
        let piece = model.token_to_piece(token, dcdr, true, None)?;
        if let Ok(diffs) = parser.feed(&piece) {
            Self::handle_diffs(&diffs, res);
        }
        Ok(())
    }

    /// Processes a slice of `ChatDiff` and updates the `response` accordingly.
    /// This method handles updating the response's reasoning, content, and tool calls based on the diffs.
    fn handle_diffs(diffs: &[ChatDiff], response: &mut Response) {
        StreamChunk::from_diffs(&diffs).iter().for_each(|chunk| {
            if let Some(rsn) = &chunk.reasoning {
                print!("{rsn}");
                response.reasoning.push_str(rsn);
            }
            if let Some(content) = &chunk.content {
                print!("{content}");
                response.content.push_str(content);
            }
            if let Some(tcs) = &chunk.tool_call {
                // Print each chunk of TCS as it streams in.
                if !tcs.id().is_empty() {
                    print!("{}", tcs.id());
                }
                if !tcs.name().is_empty() {
                    print!("{}", tcs.name())
                }
                if !tcs.arguments().is_empty() {
                    print!("{}", tcs.arguments());
                }

                if let Some(idx) = chunk.tool_call_index {
                    if idx >= response.tool_calls.len() {
                        response.tool_calls.push(ToolCall {
                            name: tcs.name().to_string(),
                            arguments: tcs.arguments().to_string(),
                            id: tcs.id().to_string(),
                        });
                    } else {
                        let tc = &mut response.tool_calls[idx];
                        tc.name.push_str(&tcs.name());
                        tc.arguments.push_str(&tcs.arguments());
                        if !tcs.id().is_empty() {
                            tc.id.push_str(&tcs.id());
                        }
                    }
                }
            }
        });
    }

    fn should_stop(&self, tokens: &[LlamaToken]) -> bool {
        for delim in &self.delimiters {
            // We don't stop if the model is generating its own role's delimiter
            if delim.role == llama_cpp_2::chat_parser::LlamaChatRole::Assistant {
                continue;
            }

            //let tokens = &delim.tokens;
            if delim.tokens.is_empty() {
                continue;
            }

            if tokens.len() >= delim.tokens.len() {
                let tail = &tokens[tokens.len() - delim.tokens.len()..];
                if tail == delim.tokens {
                    println!(
                        "\n[DEBUG] Detected role delimiter {:?}. Stopping generation.",
                        delim.role
                    );
                    return true;
                }
            }
        }
        false
    }
}

#[derive(Debug, Clone)]
struct ToolCall {
    pub name: String,
    pub arguments: String,
    pub id: String,
}

#[allow(unused)]
#[derive(Debug, Clone)]
struct Response {
    pub role: String,
    pub reasoning: String,
    pub content: String,
    pub tool_calls: Vec<ToolCall>,

    pub started_at: chrono::DateTime<chrono::Utc>,
    pub finished_at: chrono::DateTime<chrono::Utc>,
    pub tokens_generated: usize,
    pub tokens_per_second: f32,
    pub time_to_first_token: f64,
}

impl Default for Response {
    fn default() -> Self {
        Self {
            role: "assistant".into(),
            reasoning: String::default(),
            content: String::default(),
            tool_calls: Vec::default(),
            started_at: chrono::Utc::now(),
            finished_at: chrono::Utc::now(),
            tokens_generated: 0,
            tokens_per_second: 0.0,
            time_to_first_token: 0.0,
        }
    }
}

#[derive(Debug)]
struct StreamChunk<'a> {
    content: Option<Cow<'a, str>>,
    reasoning: Option<Cow<'a, str>>,
    tool_call_index: Option<usize>,
    tool_call: Option<LlamaChatToolCall>,
}

impl<'a> StreamChunk<'a> {
    /// Creates a vector of `StreamChunk` from a slice of `ChatDiff`
    pub fn from_diffs(diffs: &'a [ChatDiff]) -> Vec<StreamChunk<'a>> {
        diffs
            .iter()
            .map(|diff| {
                let mut chunk = StreamChunk {
                    content: None,
                    reasoning: None,
                    tool_call_index: diff.tool_call_index(),
                    tool_call: None,
                };
                if let Some(reasoning) = diff.reasoning() {
                    chunk.reasoning.replace(reasoning);
                }
                if let Some(content) = diff.content() {
                    chunk.content.replace(content);
                }
                if let Some(tools) = diff.tool_call() {
                    chunk.tool_call.replace(tools);
                }
                chunk
            })
            .collect::<Vec<_>>()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let params = ChatParserCliParams::parse();

    // Validate required parameters
    if !Path::new(&params.model_path).exists() {
        eprintln!("Error: Model file not found: {}", params.model_path);
        return Err("Model file not found".into());
    }

    if !Path::new(&params.mmproj_path).exists() {
        eprintln!(
            "Error: Multimodal projection file not found: {}",
            params.mmproj_path
        );
        return Err("Multimodal projection file not found".into());
    }

    println!("Loading model: {}", params.model_path);

    // Initialize backend
    let backend = LlamaBackend::init()?;

    // Setup model parameters
    let mut model_params = LlamaModelParams::default();
    if !params.no_gpu {
        model_params = model_params.with_n_gpu_layers(1_000_000); // Use all layers on GPU
    }

    // Load model
    let model = LlamaModel::load_from_file(&backend, &params.model_path, &model_params)?;

    let context_params = LlamaContextParams::default()
        .with_n_threads(params.n_threads)
        .with_n_batch(params.batch_size.try_into()?)
        .with_n_ctx(Some(params.n_tokens));
    let context = model.new_context(&backend, context_params.clone())?;

    let mut draft_model: Option<LlamaModel> = None;
    let mut draft_context: Option<LlamaContext> = None;

    if let Some(mtp_path) = &params.mtp_path {
        if Path::new(&mtp_path).exists() {
            println!("Loading MTP context: {mtp_path}");
            draft_model.replace(LlamaModel::load_from_file(
                &backend,
                mtp_path,
                &model_params,
            )?);
            if let Some(dm) = &draft_model {
                draft_context.replace(
                    dm.new_context_with_ctx_other(&backend, context_params.clone(), &context)
                        .expect("could create draft context"),
                );
            }
        }
    }

    println!("Model loaded successfully");
    println!("Loading mtmd projection: {}", params.mmproj_path);

    // Create the MTMD context
    let mut ctx = ChatParserCliContext::new(&params, &model)?;

    ctx.run_single_turn(&model, context, draft_context, &params)?;

    println!("\n");

    Ok(())
}
