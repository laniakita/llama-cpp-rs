//! Safe wrapper around llama.cpp's `common_reasoning_budget` sampler.

use std::borrow::Borrow;
use std::ops::Deref;
use std::ptr;
use std::slice;

use crate::sampling::LlamaSampler;
use crate::token::data_array::LlamaTokenDataArray;
use crate::token::LlamaToken;
use crate::vocab::LlamaVocab;

/// States of the reasoning budget state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ReasoningBudgetState {
    /// Waiting for a start sequence.
    #[default]
    Idle,
    /// Counting down remaining tokens; watching for natural end sequence.
    Counting,
    /// Forcing budget message + end sequence token-by-token.
    Forcing,
    /// Budget exhausted; waiting for UTF-8 byte completion.
    WaitingUtf8,
    /// Reasoning complete; passthrough forever.
    Done,
}

impl From<llama_cpp_sys_2::llama_rs_reasoning_budget_state> for ReasoningBudgetState {
    fn from(s: llama_cpp_sys_2::llama_rs_reasoning_budget_state) -> Self {
        match s {
            llama_cpp_sys_2::LLAMA_RS_REASONING_BUDGET_COUNTING => Self::Counting,
            llama_cpp_sys_2::LLAMA_RS_REASONING_BUDGET_FORCING => Self::Forcing,
            llama_cpp_sys_2::LLAMA_RS_REASONING_BUDGET_WAITING_UTF8 => Self::WaitingUtf8,
            llama_cpp_sys_2::LLAMA_RS_REASONING_BUDGET_DONE => Self::Done,
            _ => Self::Idle,
        }
    }
}

impl From<ReasoningBudgetState> for llama_cpp_sys_2::llama_rs_reasoning_budget_state {
    fn from(s: ReasoningBudgetState) -> Self {
        match s {
            ReasoningBudgetState::Idle => llama_cpp_sys_2::LLAMA_RS_REASONING_BUDGET_IDLE,
            ReasoningBudgetState::Counting => llama_cpp_sys_2::LLAMA_RS_REASONING_BUDGET_COUNTING,
            ReasoningBudgetState::Forcing => llama_cpp_sys_2::LLAMA_RS_REASONING_BUDGET_FORCING,
            ReasoningBudgetState::WaitingUtf8 => {
                llama_cpp_sys_2::LLAMA_RS_REASONING_BUDGET_WAITING_UTF8
            }
            ReasoningBudgetState::Done => llama_cpp_sys_2::LLAMA_RS_REASONING_BUDGET_DONE,
        }
    }
}

/// Errors that can occur when creating or configuring a reasoning budget sampler.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum ReasoningBudgetError {
    /// Failed to initialize reasoning budget sampler.
    #[error("Failed to initialize reasoning budget sampler")]
    NullSampler,
    /// Start sequences must not be empty.
    #[error("Start sequences must not be empty")]
    EmptyStartSequences,
    /// End sequences must not be empty.
    #[error("End sequences must not be empty")]
    EmptyEndSequences,
    /// Forced tokens must not be empty.
    #[error("Forced tokens must not be empty")]
    EmptyForcedTokens,
}

/// Safe wrapper around llama.cpp's `common_reasoning_budget` sampler.
///
/// This sampler monitors the generated token stream for reasoning delimiters (e.g. `<think>` and
/// `</think>`). When a start sequence is detected, it counts generated tokens against a budget.
/// Once the budget is exhausted, it forces the generation of a termination sequence (such as a
/// timeout message followed by the closing tag) by setting all other candidate logits to `-INFINITY`.
///
/// Furthermore, this sampler provides state queries to gate lazy grammar samplers during thinking,
/// and facilitates replaying the matched end tag into dependent samplers.
#[derive(Debug)]
pub struct ReasoningBudget {
    sampler: LlamaSampler,
}

impl Clone for ReasoningBudget {
    fn clone(&self) -> Self {
        let ptr = unsafe { llama_cpp_sys_2::llama_sampler_clone(self.sampler.sampler) };
        assert!(!ptr.is_null(), "llama_sampler_clone returned null");
        Self {
            sampler: LlamaSampler { sampler: ptr },
        }
    }
}

impl Deref for ReasoningBudget {
    type Target = LlamaSampler;
    fn deref(&self) -> &Self::Target {
        &self.sampler
    }
}

impl ReasoningBudget {
    /// Creates a new reasoning budget sampler from token sequences.
    ///
    /// # Parameters
    /// - `vocab`: Optional reference to [`LlamaVocab`]. If provided, the sampler validates UTF-8
    ///   completeness before forcing tokens when the budget expires.
    /// - `start_seqs`: Token sequences, any of which activates budget counting.
    /// - `end_seqs`: Token sequences, any of which deactivates counting naturally.
    /// - `forced_tokens`: Token sequence forced token-by-token when budget expires.
    /// - `budget`: Maximum allowed tokens in the reasoning block. Passing a negative number (e.g. `-1`)
    ///   maps to unlimited budget (`i32::MAX`), matching `llama.cpp`'s `sampling.cpp:317`.
    /// - `initial_state`: Initial state of the sampler (defaults to [`ReasoningBudgetState::Idle`]).
    ///
    /// # Errors
    /// Returns [`ReasoningBudgetError`] if sequences are empty or FFI allocation fails.
    pub fn new(
        vocab: Option<&LlamaVocab>,
        start_seqs: &[&[LlamaToken]],
        end_seqs: &[&[LlamaToken]],
        forced_tokens: &[LlamaToken],
        budget: i32,
        initial_state: Option<ReasoningBudgetState>,
    ) -> Result<Self, ReasoningBudgetError> {
        if start_seqs.is_empty() || start_seqs.iter().all(|s| s.is_empty()) {
            return Err(ReasoningBudgetError::EmptyStartSequences);
        }
        if end_seqs.is_empty() || end_seqs.iter().all(|s| s.is_empty()) {
            return Err(ReasoningBudgetError::EmptyEndSequences);
        }
        if forced_tokens.is_empty() {
            return Err(ReasoningBudgetError::EmptyForcedTokens);
        }

        // Map negative budgets (e.g. -1 for unlimited) to i32::MAX, matching sampling.cpp:317
        let effective_budget = if budget < 0 { i32::MAX } else { budget };

        let start_ptrs: Vec<*const llama_cpp_sys_2::llama_token> =
            start_seqs.iter().map(|s| s.as_ptr().cast()).collect();
        let start_lens: Vec<usize> = start_seqs.iter().map(|s| s.len()).collect();

        let end_ptrs: Vec<*const llama_cpp_sys_2::llama_token> =
            end_seqs.iter().map(|s| s.as_ptr().cast()).collect();
        let end_lens: Vec<usize> = end_seqs.iter().map(|s| s.len()).collect();

        let vocab_ptr = vocab.map_or(ptr::null(), crate::vocab::LlamaVocab::as_ptr);
        let initial_state_ffi = initial_state.unwrap_or_default().into();

        let sampler_ptr = unsafe {
            llama_cpp_sys_2::llama_rs_reasoning_budget_init(
                vocab_ptr,
                start_ptrs.as_ptr(),
                start_lens.as_ptr(),
                start_seqs.len(),
                end_ptrs.as_ptr(),
                end_lens.as_ptr(),
                end_seqs.len(),
                forced_tokens.as_ptr().cast(),
                forced_tokens.len(),
                effective_budget,
                initial_state_ffi,
            )
        };

        if sampler_ptr.is_null() {
            Err(ReasoningBudgetError::NullSampler)
        } else {
            Ok(Self {
                sampler: LlamaSampler {
                    sampler: sampler_ptr,
                },
            })
        }
    }

    /// Returns the current state of the reasoning budget sampler.
    #[must_use]
    pub fn state(&self) -> ReasoningBudgetState {
        let raw =
            unsafe { llama_cpp_sys_2::llama_rs_reasoning_budget_get_state(self.sampler.sampler) };
        raw.into()
    }

    /// Returns the matched end sequence that transitioned the sampler to `Done`, or `None` if
    /// none was recorded or the sampler has since been re-armed.
    #[must_use]
    pub fn end_match(&self) -> Option<&[LlamaToken]> {
        let mut len: usize = 0;
        let ptr = unsafe {
            llama_cpp_sys_2::llama_rs_reasoning_budget_get_end_match(
                self.sampler.sampler,
                std::ptr::addr_of_mut!(len),
            )
        };
        if ptr.is_null() || len == 0 {
            None
        } else {
            Some(unsafe { slice::from_raw_parts(ptr.cast::<LlamaToken>(), len) })
        }
    }

    /// Manually transition the reasoning budget sampler into the `Forcing` state.
    ///
    /// Returns `true` if the transition occurred (i.e. if the sampler was currently in `Counting` state).
    pub fn force(&mut self) -> bool {
        unsafe { llama_cpp_sys_2::llama_rs_reasoning_budget_force(self.sampler.sampler) }
    }

    /// Accepts a token into the reasoning budget sampler.
    pub fn accept(&mut self, token: LlamaToken) {
        self.sampler.accept(token);
    }

    /// Accepts multiple tokens into the reasoning budget sampler (e.g. for feeding prefill tokens).
    pub fn accept_many(&mut self, tokens: impl IntoIterator<Item = impl Borrow<LlamaToken>>) {
        self.sampler.accept_many(tokens);
    }

    /// Applies the reasoning budget sampler to a candidate token array.
    ///
    /// When forcing, all logits except the forced token are set to `-INFINITY`.
    pub fn apply(&self, data_array: &mut LlamaTokenDataArray) {
        self.sampler.apply(data_array);
    }

    /// Resets the internal state of the reasoning budget sampler to `Idle`.
    pub fn reset(&mut self) {
        self.sampler.reset();
    }

    /// Returns a reference to the underlying [`LlamaSampler`].
    #[must_use]
    pub fn as_sampler(&self) -> &LlamaSampler {
        &self.sampler
    }

    /// Consumes the wrapper and returns the underlying [`LlamaSampler`].
    #[must_use]
    pub fn into_sampler(self) -> LlamaSampler {
        self.sampler
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token::data::LlamaTokenData;

    #[test]
    fn test_validation_errors() {
        let valid_tok = [LlamaToken(1)];
        let valid_slice = [&valid_tok[..]];

        assert_eq!(
            ReasoningBudget::new(None, &[], &valid_slice, &valid_tok, 10, None).unwrap_err(),
            ReasoningBudgetError::EmptyStartSequences
        );

        assert_eq!(
            ReasoningBudget::new(None, &valid_slice, &[], &valid_tok, 10, None).unwrap_err(),
            ReasoningBudgetError::EmptyEndSequences
        );

        assert_eq!(
            ReasoningBudget::new(None, &valid_slice, &valid_slice, &[], 10, None).unwrap_err(),
            ReasoningBudgetError::EmptyForcedTokens
        );
    }

    #[test]
    fn test_negative_budget_mapping() {
        let start = [LlamaToken(10), LlamaToken(11)];
        let end = [LlamaToken(20), LlamaToken(21)];
        let forced = [LlamaToken(99)];

        // -1 should map to i32::MAX and remain in Counting, not forcing immediately
        let mut budget = ReasoningBudget::new(None, &[&start], &[&end], &forced, -1, None).unwrap();

        assert_eq!(budget.state(), ReasoningBudgetState::Idle);
        budget.accept(LlamaToken(10));
        budget.accept(LlamaToken(11));
        assert_eq!(budget.state(), ReasoningBudgetState::Counting);

        // Feed multiple tokens; it should remain in Counting
        for _ in 0..10 {
            budget.accept(LlamaToken(1));
            assert_eq!(budget.state(), ReasoningBudgetState::Counting);
        }
    }

    #[test]
    fn test_full_lifecycle_and_forcing() {
        let start = [LlamaToken(10), LlamaToken(11)];
        let end = [LlamaToken(20), LlamaToken(21)];
        let forced = [LlamaToken(99), LlamaToken(20), LlamaToken(21)];

        let mut budget = ReasoningBudget::new(None, &[&start], &[&end], &forced, 2, None).unwrap();

        assert_eq!(budget.state(), ReasoningBudgetState::Idle);

        // Feed start tokens
        budget.accept(LlamaToken(10));
        assert_eq!(budget.state(), ReasoningBudgetState::Idle);
        budget.accept(LlamaToken(11));
        assert_eq!(budget.state(), ReasoningBudgetState::Counting);

        // Token 1 of 2
        budget.accept(LlamaToken(1));
        assert_eq!(budget.state(), ReasoningBudgetState::Counting);

        // Token 2 of 2 (budget exhausted -> transitions to Forcing)
        budget.accept(LlamaToken(2));
        assert_eq!(budget.state(), ReasoningBudgetState::Forcing);

        // Check logit clamping during forcing
        let mut data_array = LlamaTokenDataArray::new(
            vec![
                LlamaTokenData::new(LlamaToken(1), 5.0, 0.0),
                LlamaTokenData::new(LlamaToken(99), 1.0, 0.0),
            ],
            false,
        );
        budget.apply(&mut data_array);
        assert_eq!(data_array.data[0].logit(), f32::NEG_INFINITY);
        assert_eq!(data_array.data[1].logit(), 1.0);

        // Accept forced tokens: [99, 20, 21]
        budget.accept(LlamaToken(99));
        assert_eq!(budget.state(), ReasoningBudgetState::Forcing);

        budget.accept(LlamaToken(20));
        assert_eq!(budget.state(), ReasoningBudgetState::Forcing);

        budget.accept(LlamaToken(21));
        assert_eq!(budget.state(), ReasoningBudgetState::Done);

        // End match should match the end sequence
        let matched = budget.end_match().expect("expected end match");
        assert_eq!(matched, &[LlamaToken(20), LlamaToken(21)]);
    }

    #[test]
    fn test_natural_end_sequence() {
        let start = [LlamaToken(10)];
        let end = [LlamaToken(20)];
        let forced = [LlamaToken(99)];

        let mut budget = ReasoningBudget::new(None, &[&start], &[&end], &forced, 10, None).unwrap();

        budget.accept(LlamaToken(10));
        assert_eq!(budget.state(), ReasoningBudgetState::Counting);

        budget.accept(LlamaToken(5));
        assert_eq!(budget.state(), ReasoningBudgetState::Counting);

        // Natural end tag accepted before budget runs out
        budget.accept(LlamaToken(20));
        assert_eq!(budget.state(), ReasoningBudgetState::Done);
        assert_eq!(budget.end_match(), Some(&[LlamaToken(20)][..]));
    }

    #[test]
    fn test_manual_force() {
        let start = [LlamaToken(10)];
        let end = [LlamaToken(20)];
        let forced = [LlamaToken(99)];

        let mut budget =
            ReasoningBudget::new(None, &[&start], &[&end], &forced, 100, None).unwrap();

        // Cannot force while Idle
        assert!(!budget.force());

        budget.accept(LlamaToken(10));
        assert_eq!(budget.state(), ReasoningBudgetState::Counting);

        // Force transition to Forcing
        assert!(budget.force());
        assert_eq!(budget.state(), ReasoningBudgetState::Forcing);

        // Cannot force again once already Forcing
        assert!(!budget.force());
    }

    #[test]
    fn test_clone_independence() {
        let start = [LlamaToken(10)];
        let end = [LlamaToken(20)];
        let forced = [LlamaToken(99)];

        let mut budget = ReasoningBudget::new(None, &[&start], &[&end], &forced, 5, None).unwrap();

        budget.accept(LlamaToken(10));
        assert_eq!(budget.state(), ReasoningBudgetState::Counting);

        let mut cloned = budget.clone();
        assert_eq!(cloned.state(), ReasoningBudgetState::Counting);

        // Exhaust budget on original
        for _ in 0..5 {
            budget.accept(LlamaToken(1));
        }
        assert_eq!(budget.state(), ReasoningBudgetState::Forcing);

        // Cloned remains in Counting
        assert_eq!(cloned.state(), ReasoningBudgetState::Counting);
        cloned.accept(LlamaToken(1));
        assert_eq!(cloned.state(), ReasoningBudgetState::Counting);
    }

    #[test]
    fn test_end_match_retrieval() {
        let start = [LlamaToken(10)];
        let end = [LlamaToken(20), LlamaToken(21)];
        let forced = [LlamaToken(99)];

        let mut budget = ReasoningBudget::new(None, &[&start], &[&end], &forced, 10, None).unwrap();

        budget.accept(LlamaToken(10));
        budget.accept(LlamaToken(20));
        budget.accept(LlamaToken(21));
        assert_eq!(budget.state(), ReasoningBudgetState::Done);
        assert_eq!(
            budget.end_match(),
            Some(&[LlamaToken(20), LlamaToken(21)][..])
        );
    }
}
