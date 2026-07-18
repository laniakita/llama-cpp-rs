//! Based on the mtmd cli example from llama.cpp.

use std::borrow::Cow;
use std::ffi::CString;
use std::io::{self, Write};
use std::num::NonZeroU32;
use std::path::Path;

use clap::Parser;
use encoding_rs::{Decoder, UTF_8};

use llama_cpp_2::chat_parser::{ChatDiff, ChatParser, ChatParserInitError, LlamaGenerationParams};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::mtmd::{
    MtmdBitmap, MtmdBitmapError, MtmdContext, MtmdContextParams, MtmdInputText,
};

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::{LlamaChatMessage, LlamaChatTemplate, LlamaChatToolCall, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::speculative::{MtpSpeculative, MtpSpeculativeParams};
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
    #[arg(long = "batch-size", value_name = "b", default_value = "1")]
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
    pub mtmd_ctx: MtmdContext,

    /// Tokenized prompt.
    pub prompt_tokens: Vec<LlamaToken>,

    /// The batch used for processing tokens.
    pub batch: LlamaBatch<'a>,
    /// The list of loaded bitmaps (images/audio).
    pub bitmaps: Vec<MtmdBitmap>,
    /// The number of past tokens processed.
    pub n_past: i32,
    /// The chat template used for formatting messages.
    pub chat_template: LlamaChatTemplate,
    /// The current chat generation params.
    pub generation_params: Option<LlamaGenerationParams>,
    /// The chat parser.
    pub parser: Option<ChatParser>,
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

        let chat_template = model
            .chat_template(params.chat_template.as_deref())
            .map_err(|e| format!("Failed to get chat template: {e}"))?;

        let batch = LlamaBatch::new(params.n_tokens.get() as usize, 1);

        Ok(Self {
            mtmd_ctx,
            batch,
            bitmaps: Vec::new(),
            prompt_tokens: Vec::new(),
            n_past: 0,
            chat_template,
            generation_params: None,
            parser: None,
        })
    }

    /// Loads media (image or audio) from the specified file path
    /// # Errors
    pub fn load_media(&mut self, path: &str) -> Result<(), MtmdBitmapError> {
        let bitmap = MtmdBitmap::from_file(&self.mtmd_ctx, path, false)?;
        self.bitmaps.push(bitmap);
        Ok(())
    }

    /// Evaluates a chat message, tokenizing and processing it through the model
    /// # Errors
    pub fn eval_message(
        &mut self,
        model: &LlamaModel,
        context: &mut LlamaContext,
        msg: LlamaChatMessage,
        add_bos: bool,
        batch_size: i32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let gen_params = LlamaGenerationParams::builder()
            .with_add_generation_prompt(true)
            .with_enable_thinking(true)
            .with_messages(&[msg])
            .with_add_bos(add_bos)
            .build();

        println!("Generation params: {:#?}", gen_params);

        // Format the message using chat template (simplified)
        let chat_params = model
            .apply_chat_template_full(Some(&self.chat_template), &gen_params)
            .map_err(|e| format!("Failed to apply chat template: {e}"))?;

        println!("Chat params: {:#?}", chat_params.view());

        let parser = ChatParser::new(&chat_params, &gen_params)?;
        self.parser.replace(parser);
        self.generation_params.replace(gen_params);

        let input_text = MtmdInputText {
            text: chat_params.prompt().to_string(),
            add_special: add_bos,
            parse_special: true,
        };

        let bitmap_refs: Vec<&MtmdBitmap> = self.bitmaps.iter().collect();

        if bitmap_refs.is_empty() {
            println!("No bitmaps provided, only tokenizing text");
        } else {
            println!("Tokenizing with {} bitmaps", bitmap_refs.len());
        }

        // Tokenize the input
        self.prompt_tokens.clear();
        let chunks = self.mtmd_ctx.tokenize(input_text, &bitmap_refs)?;
        for i in 0..chunks.len() {
            if let Some(chunk) = chunks.get(i) {
                if let Some(tokens) = chunk.text_tokens() {
                    self.prompt_tokens.extend_from_slice(tokens);
                }
            }
        }

        // Clear bitmaps after tokenization
        self.bitmaps.clear();

        self.n_past = chunks.eval_chunks(&self.mtmd_ctx, context, 0, 0, batch_size, true)?;
        Ok(())
    }

    /// Generates a response by sampling tokens from the model
    /// # Errors
    pub fn generate_response(
        &mut self,
        model: &LlamaModel,
        mut context: LlamaContext,
        mtp_ctx: Option<LlamaContext>,
        sampler: &mut LlamaSampler,
        n_predict: i32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        match mtp_ctx {
            Some(mtp_context) => {
                self.generate_with_mtp(model, context, mtp_context, sampler, n_predict)
            }
            None => self.generate(model, &mut context, sampler, n_predict),
        }
    }

    /// Generates a response by sampling tokens from the model
    /// # Errors
    pub fn generate(
        &mut self,
        model: &LlamaModel,
        context: &mut LlamaContext,
        sampler: &mut LlamaSampler,
        n_predict: i32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let t_initial = std::time::Instant::now();

        let mut generated_tokens = Vec::new();
        let max_predict = if n_predict < 0 { i32::MAX } else { n_predict };
        let mut decoder = UTF_8.new_decoder();

        let mut parser = self
            .parser
            .take()
            .ok_or_else(|| ChatParserInitError::NullStateReturn)
            .map_err(|err| format!("Failed to create parser: {err:#?}"))?;

        let mut response = Response {
            reasoning: "".into(),
            content: "".into(),
            tool_calls: Vec::new(),
            started_at: std::time::SystemTime::now(),
            finished_at: std::time::SystemTime::now(),
            tokens_generated: 0,
            tokens_per_second: 0.0,
        };

        let handle_token = |token: LlamaToken,
                            dcdr: &mut Decoder,
                            parser: &mut ChatParser,
                            res: &mut Response|
         -> Result<(), Box<dyn std::error::Error>> {
            res.tokens_generated += 1;
            let piece = model.token_to_piece(token, dcdr, true, None)?;
            if let Ok(diffs) = parser.feed_piece(&piece) {
                StreamChunk::from_diffs(&diffs).iter().for_each(|chunk| {
                    if let Some(rsn) = &chunk.reasoning {
                        print!("{rsn}");
                        res.reasoning.push_str(rsn);
                    }
                    if let Some(content) = &chunk.content {
                        print!("{content}");
                        res.content.push_str(content);
                    }
                    if let Some(tcs) = &chunk.tool_call {
                        print!("{tcs:#?}");
                        res.tool_calls.push(tcs.clone());
                    }
                    let _ = io::stdout().flush();
                });
            }
            Ok(())
        };
        for _i in 0..max_predict {
            let token = sampler.sample(&context, -1);
            generated_tokens.push(token);
            sampler.accept(token);

            if model.is_eog_token(token) {
                println!();
                break;
            }

            handle_token(token, &mut decoder, &mut parser, &mut response)?;

            // Prepare next batch
            self.batch.clear();
            self.batch.add(token, self.n_past, &[0], true)?;
            self.n_past += 1;

            // Decode
            context.decode(&mut self.batch)?;
        }
        let t_final = t_initial.elapsed();
        response.finished_at = std::time::SystemTime::now();
        response.tokens_per_second = response.tokens_generated as f32 / t_final.as_secs_f32();
        println!("{response:#?}");

        println!("{response:#?}");

        Ok(())
    }

    /// Generates a response by sampling tokens from the model
    /// # Errors
    pub fn generate_with_mtp(
        &mut self,
        model: &LlamaModel,
        context: LlamaContext,
        mtp_ctx: LlamaContext,
        sampler: &mut LlamaSampler,
        n_predict: i32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let t_initial = std::time::Instant::now();

        let max_predict = if n_predict < 0 { i32::MAX } else { n_predict };
        let mut decoder = UTF_8.new_decoder();
        let mut parser = self
            .parser
            .take()
            .ok_or_else(|| ChatParserInitError::NullStateReturn)
            .map_err(|err| format!("Failed to create parser: {err:#?}"))?;

        let mut id_last = sampler.sample(&context, -1);
        sampler.accept(id_last);

        let mut mtp = MtpSpeculative::new(
            context,
            mtp_ctx,
            MtpSpeculativeParams {
                n_max: 2,
                n_min: 0,
                p_min: 0.85,
            },
        )
        .map_err(|err| format!("Failed to create speculative decoder: {err:#?}"))?;
        mtp.begin(&self.prompt_tokens)
            .map_err(|err| format!("Failed to start speculative decoding: {err:#?}"))?;
        println!("mtp loaded");

        let mut response = Response {
            reasoning: "".into(),
            content: "".into(),
            tool_calls: Vec::new(),
            started_at: std::time::SystemTime::now(),
            finished_at: std::time::SystemTime::now(),
            tokens_generated: 0,
            tokens_per_second: 0.0,
        };

        let handle_token = |token: LlamaToken,
                            dcdr: &mut Decoder,
                            parser: &mut ChatParser,
                            res: &mut Response|
         -> Result<(), Box<dyn std::error::Error>> {
            res.tokens_generated += 1;
            let piece = model.token_to_piece(token, dcdr, true, None)?;
            if let Ok(diffs) = parser.feed_piece(&piece) {
                StreamChunk::from_diffs(&diffs).iter().for_each(|chunk| {
                    if let Some(rsn) = &chunk.reasoning {
                        print!("{rsn}");
                        res.reasoning.push_str(rsn);
                    }
                    if let Some(content) = &chunk.content {
                        print!("{content}");
                        res.content.push_str(content);
                    }
                    if let Some(tcs) = &chunk.tool_call {
                        print!("{tcs:#?}");
                        res.tool_calls.push(tcs.clone());
                    }
                    let _ = io::stdout().flush();
                });
            }
            Ok(())
        };

        for _ in 0..max_predict {
            if model.is_eog_token(id_last) {
                println!();
                break;
            }

            handle_token(id_last, &mut decoder, &mut parser, &mut response)?;

            let draft_tokens = mtp.draft(self.n_past, id_last, &[])?;

            self.batch.clear();

            let mut batch_n_past = self.n_past;
            self.batch.add(id_last, batch_n_past, &[0], true)?;
            batch_n_past += 1;

            for tk in &draft_tokens {
                self.batch.add(*tk, batch_n_past, &[0], true)?;
                batch_n_past += 1;
            }

            mtp.target_context_mut().decode(&mut self.batch)?;

            if !draft_tokens.is_empty() {
                mtp.process(&self.batch)
                    .map_err(|err| format!("MTP failed to process batch: {err:#?}"))?;
            }

            let mut n_accepted = 0;
            let mut new_id_last = None;

            for i in 0..draft_tokens.len() {
                let target_sampled_token = sampler.sample(mtp.target_context(), i as i32);
                if target_sampled_token == draft_tokens[i] {
                    n_accepted += 1;
                    sampler.accept(target_sampled_token);

                    handle_token(
                        target_sampled_token,
                        &mut decoder,
                        &mut parser,
                        &mut response,
                    )?;
                } else {
                    new_id_last = Some(target_sampled_token);
                    sampler.accept(target_sampled_token);
                    break;
                }
            }

            if let Some(new_id) = new_id_last {
                id_last = new_id;
            } else {
                let last_sampled_token =
                    sampler.sample(mtp.target_context(), draft_tokens.len() as i32);
                new_id_last.replace(last_sampled_token);
                sampler.accept(last_sampled_token);
                id_last = last_sampled_token;
            }

            if !draft_tokens.is_empty() {
                mtp.accept(n_accepted)?;
                let tokens_to_keep_until = self.n_past + 1 + n_accepted as i32;
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

                self.n_past = tokens_to_keep_until;
            } else {
                self.n_past += 1;
            }
        }
        let t_final = t_initial.elapsed();
        response.finished_at = std::time::SystemTime::now();
        response.tokens_per_second = response.tokens_generated as f32 / t_final.as_secs_f32();
        println!("{response:#?}");

        Ok(())
    }
}

#[allow(unused)]
#[derive(Debug, Clone)]
struct Response {
    pub reasoning: String,
    pub content: String,
    pub tool_calls: Vec<LlamaChatToolCall>,

    pub started_at: std::time::SystemTime,
    pub finished_at: std::time::SystemTime,
    pub tokens_generated: usize,
    pub tokens_per_second: f32,
}

#[derive(Debug)]
struct StreamChunk<'a> {
    content: Option<Cow<'a, str>>,
    reasoning: Option<Cow<'a, str>>,
    tool_call: Option<LlamaChatToolCall>,
}

impl<'a> StreamChunk<'a> {
    /// Creates a vector of `StreamChunk` from a slice of `ChatDiff`
    pub fn from_diffs(diffs: &[ChatDiff<'a>]) -> Vec<StreamChunk<'a>> {
        diffs
            .iter()
            .map(|diff| {
                let mut chunk = StreamChunk {
                    content: None,
                    reasoning: None,
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

fn run_single_turn(
    ctx: &mut ChatParserCliContext,
    model: &LlamaModel,
    mut context: LlamaContext,
    mtp_ctx: Option<LlamaContext>,
    sampler: &mut LlamaSampler,
    params: &ChatParserCliParams,
) -> Result<(), Box<dyn std::error::Error>> {
    // Add media marker if not present
    let mut prompt = params.prompt.clone();

    let default_marker = llama_cpp_2::mtmd::mtmd_default_marker().to_string();
    let media_marker = params.media_marker.as_ref().unwrap_or(&default_marker);
    if (!params.images.is_empty() || !params.audio.is_empty()) && !prompt.contains(media_marker) {
        prompt.push_str(media_marker);
    }

    // Load media files
    for image_path in &params.images {
        println!("Loading image: {image_path}");
        ctx.load_media(image_path)?;
    }
    for audio_path in &params.audio {
        ctx.load_media(audio_path)?;
    }

    // Create user message
    let msg = LlamaChatMessage::new("user".into(), prompt)?;

    println!("Evaluating message: {msg:?}");

    // Evaluate the message (prefill)
    ctx.eval_message(model, &mut context, msg, true, params.batch_size)?;
    // Generate response (decode)
    ctx.generate_response(model, context, mtp_ctx, sampler, params.n_predict)?;

    Ok(())
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
        .with_n_batch(params.batch_size.max(256).try_into()?)
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

    // Create sampler
    let mut sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);

    println!("Model loaded successfully");
    println!("Loading mtmd projection: {}", params.mmproj_path);

    // Create the MTMD context
    let mut ctx = ChatParserCliContext::new(&params, &model)?;

    run_single_turn(
        &mut ctx,
        &model,
        context,
        draft_context,
        &mut sampler,
        &params,
    )?;

    println!("\n");

    Ok(())
}
