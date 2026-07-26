//! utilities for working with the kv cache

use crate::context::LlamaContext;
use std::ffi::c_int;
use std::num::{NonZeroU8, TryFromIntError};

/// Errors that can occur when attempting to prepare values for the kv cache
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
#[allow(clippy::module_name_repetitions)]
pub enum KvCacheConversionError {
    /// Sequence id conversion to i32 failed
    #[error("Provided sequence id is too large for a i32")]
    SeqIdTooLarge(#[source] TryFromIntError),
    /// Position 0 conversion to i32 failed
    #[error("Provided start position is too large for a i32")]
    P0TooLarge(#[source] TryFromIntError),
    /// Position 1 conversion to i32 failed
    #[error("Provided end position is too large for a i32")]
    P1TooLarge(#[source] TryFromIntError),
    /// Partial sequence couldn't be removed
    #[error("Couldn't remove partial sequence")]
    PartialSequenceRemovalFailed(i32, i32),
    /// The specified sequence was not found when attempting to checkpoint.
    #[error("Sequence not found or has no state")]
    SequenceNotFound,
}

/// An opaque wrapper around a serialized sequence state from the KV cache.
#[derive(Debug, Clone)]
pub struct SeqDataExt {
    pub(crate) data: Vec<u8>,
}

impl SeqDataExt {
    /// Returns the raw binary data of the sequence state.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }
}

impl LlamaContext<'_> {
    /// Copy the cache from one sequence to another.
    ///
    /// # Parameters
    ///
    /// * `src` - The sequence id to copy the cache from.
    /// * `dest` - The sequence id to copy the cache to.
    /// * `size` - The size of the cache to copy.
    pub fn copy_cache(&mut self, src: i32, dest: i32, size: i32) {
        let mem = unsafe { llama_cpp_sys_2::llama_get_memory(self.context.as_ptr()) };
        unsafe { llama_cpp_sys_2::llama_memory_seq_cp(mem, src, dest, 0, size) }
    }

    /// Copy the cache from one sequence to another.
    ///
    /// # Returns
    /// A `Result` indicating whether the operation was successful.
    ///
    /// # Parameters
    /// * `src` - The sequence id to copy the cache from.
    /// * `dest` - The sequence id to copy the cache to.
    /// * `p0` - The start position of the cache to clear. If `None`, the entire cache is copied up to `p1`.
    /// * `p1` - The end position of the cache to clear. If `None`, the entire cache is copied starting from `p0`.
    ///
    /// # Errors
    /// If either position exceeds [`i32::MAX`].
    pub fn copy_kv_cache_seq(
        &mut self,
        src: i32,
        dest: i32,
        p0: Option<u32>,
        p1: Option<u32>,
    ) -> Result<(), KvCacheConversionError> {
        let p0 = p0
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::P0TooLarge)?;
        let p1 = p1
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::P1TooLarge)?;
        let mem = unsafe { llama_cpp_sys_2::llama_get_memory(self.context.as_ptr()) };
        unsafe { llama_cpp_sys_2::llama_memory_seq_cp(mem, src, dest, p0, p1) };
        Ok(())
    }

    /// Clear the kv cache for the given sequence within the specified range `[p0, p1)`
    /// Returns `false` only when partial sequence removals fail. Full sequence removals always succeed.
    ///
    /// # Returns
    /// A `Result` indicating whether the operation was successful. If the sequence id or
    /// either position exceeds the maximum i32 value, no removal is attempted and an `Err` is returned.
    ///
    /// # Parameters
    /// * `src` - The sequence id to clear the cache for. If `None`, matches all sequences
    /// * `p0` - The start position of the cache to clear. If `None`, the entire cache is cleared up to `p1`.
    /// * `p1` - The end position of the cache to clear. If `None`, the entire cache is cleared from `p0`.
    ///
    /// # Errors
    /// If the sequence id or either position exceeds [`i32::MAX`].
    pub fn clear_kv_cache_seq(
        &mut self,
        src: Option<u32>,
        p0: Option<u32>,
        p1: Option<u32>,
    ) -> Result<bool, KvCacheConversionError> {
        let src = src
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::SeqIdTooLarge)?;
        let p0 = p0
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::P0TooLarge)?;
        let p1 = p1
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::P1TooLarge)?;
        let mem = unsafe { llama_cpp_sys_2::llama_get_memory(self.context.as_ptr()) };
        Ok(unsafe { llama_cpp_sys_2::llama_memory_seq_rm(mem, src, p0, p1) })
    }

    /// Clear the KV cache
    pub fn clear_kv_cache(&mut self) {
        let mem = unsafe { llama_cpp_sys_2::llama_get_memory(self.context.as_ptr()) };
        // clear both metadata and data buffers to match previous semantics
        unsafe { llama_cpp_sys_2::llama_memory_clear(mem, true) }
    }

    /// Removes all tokens that do not belong to the specified sequence
    ///
    /// # Parameters
    ///
    /// * `seq_id` - The sequence id to keep
    pub fn llama_kv_cache_seq_keep(&mut self, seq_id: i32) {
        let mem = unsafe { llama_cpp_sys_2::llama_get_memory(self.context.as_ptr()) };
        unsafe { llama_cpp_sys_2::llama_memory_seq_keep(mem, seq_id) }
    }

    #[allow(clippy::doc_markdown)]
    /// Copy all tokens that belong to the specified sequence to another sequence
    /// If the KV cache is RoPEd, the KV data is updated accordingly:
    ///   - lazily on next [`LlamaContext::decode`]
    ///   - explicitly with [`Self::kv_cache_update`]
    ///
    /// # Returns
    /// A `Result` indicating whether the operation was successful.
    ///
    /// # Parameters
    ///
    /// * `seq_id_src` - The sequence id to copy from. If negative, matches any sequence.
    /// * `seq_id_dst` - The sequence id to copy to
    /// * `p0` - The start position of the cache to copy from. If `None`, the entire cache is updated up to `p1`.
    /// * `p1` - The end position of the cache to copy from. If `None`, the entire cache is updated starting from `p0`.
    ///
    /// # Errors
    /// If either position exceeds [`i32::MAX`].
    pub fn kv_cache_seq_cp(
        &mut self,
        seq_id_src: i32,
        seq_id_dst: i32,
        p0: Option<u32>,
        p1: Option<u32>,
    ) -> Result<(), KvCacheConversionError> {
        let p0 = p0
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::P0TooLarge)?;
        let p1 = p1
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::P1TooLarge)?;
        let mem = unsafe { llama_cpp_sys_2::llama_get_memory(self.context.as_ptr()) };
        unsafe { llama_cpp_sys_2::llama_memory_seq_cp(mem, seq_id_src, seq_id_dst, p0, p1) };
        Ok(())
    }

    #[allow(clippy::doc_markdown)]
    /// Adds relative position "delta" to all tokens that belong to the specified sequence and have positions in `[p0, p1)`
    /// If the KV cache is RoPEd, the KV data is updated accordingly:
    ///   - lazily on next [`LlamaContext::decode`]
    ///   - explicitly with [`Self::kv_cache_update`]
    ///
    /// # Returns
    /// A `Result` indicating whether the operation was successful.
    ///
    /// # Parameters
    ///
    /// * `seq_id` - The sequence id to update
    /// * `p0` - The start position of the cache to update. If `None`, the entire cache is updated up to `p1`.
    /// * `p1` - The end position of the cache to update. If `None`, the entire cache is updated starting from `p0`.
    /// * `delta` - The relative position to add to the tokens
    ///
    /// # Errors
    /// If either position exceeds [`i32::MAX`].
    pub fn kv_cache_seq_add(
        &mut self,
        seq_id: i32,
        p0: Option<u32>,
        p1: Option<u32>,
        delta: i32,
    ) -> Result<(), KvCacheConversionError> {
        let p0 = p0
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::P0TooLarge)?;
        let p1 = p1
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::P1TooLarge)?;
        let mem = unsafe { llama_cpp_sys_2::llama_get_memory(self.context.as_ptr()) };
        unsafe { llama_cpp_sys_2::llama_memory_seq_add(mem, seq_id, p0, p1, delta) };
        Ok(())
    }

    #[allow(clippy::doc_markdown)]
    /// Removes all tokens that belong to the specified sequence and have positions in [p0, p1)
    /// If the KV cache is RoPEd, the KV data is updated accordingly:
    ///   - lazily on next [`LlamaContext::decode`]
    ///   - explicitly with [`Self::kv_cache_update`]
    ///
    /// # Returns
    /// A `Result` indicating whether the operation was successful.
    ///
    /// # Parameters
    ///
    /// * `seq_id` - The sequence id to update
    /// * `p0` - The start position of the cache to update. If `None`, the entire cache is updated up to `p1`.
    /// * `p1` - The end position of the cache to update. If `None`, the entire cache is updated starting from `p0`.
    ///
    /// # Errors
    /// If either position exceeds [`i32::MAX`].
    pub fn kv_cache_seq_rm(
        &mut self,
        seq_id: i32,
        p0: Option<u32>,
        p1: Option<u32>,
    ) -> Result<(), KvCacheConversionError> {
        let p0 = p0
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::P0TooLarge)?;
        let p1 = p1
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::P1TooLarge)?;
        let mem = unsafe { llama_cpp_sys_2::llama_get_memory(self.context.as_ptr()) };
        if !unsafe { llama_cpp_sys_2::llama_memory_seq_rm(mem, seq_id, p0, p1) } {
            return Err(KvCacheConversionError::PartialSequenceRemovalFailed(p0, p1));
        }
        Ok(())
    }

    /// Integer division of the positions by factor of `d > 1`
    /// If the KV cache is `RoPEd`, the KV data is updated accordingly:
    ///   - lazily on next [`LlamaContext::decode`]
    ///   - explicitly with [`Self::kv_cache_update`]
    ///
    /// # Returns
    /// A `Result` indicating whether the operation was successful.
    ///
    /// # Parameters
    ///
    /// * `seq_id` - The sequence id to update
    /// * `p0` - The start position of the cache to update. If `None`, the entire cache is updated up to `p1`.
    /// * `p1` - The end position of the cache to update. If `None`, the entire cache is updated starting from `p0`.
    /// * `d` - The factor to divide the positions by
    ///
    /// # Errors
    /// If either position exceeds [`i32::MAX`].
    pub fn kv_cache_seq_div(
        &mut self,
        seq_id: i32,
        p0: Option<u32>,
        p1: Option<u32>,
        d: NonZeroU8,
    ) -> Result<(), KvCacheConversionError> {
        let p0 = p0
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::P0TooLarge)?;
        let p1 = p1
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::P1TooLarge)?;
        let d = c_int::from(d.get());
        let mem = unsafe { llama_cpp_sys_2::llama_get_memory(self.context.as_ptr()) };
        unsafe { llama_cpp_sys_2::llama_memory_seq_div(mem, seq_id, p0, p1, d) }
        Ok(())
    }

    /// Returns the smallest position present in the KV cache for the specified sequence
    ///
    /// # Parameters
    ///
    /// * `seq_id` - The sequence id to get the min position for
    #[must_use]
    pub fn kv_cache_seq_pos_min(&self, seq_id: i32) -> i32 {
        let mem = unsafe { llama_cpp_sys_2::llama_get_memory(self.context.as_ptr()) };
        unsafe { llama_cpp_sys_2::llama_memory_seq_pos_min(mem, seq_id) }
    }

    /// Returns the largest position present in the KV cache for the specified sequence
    ///
    /// # Parameters
    ///
    /// * `seq_id` - The sequence id to get the max position for
    #[must_use]
    pub fn kv_cache_seq_pos_max(&self, seq_id: i32) -> i32 {
        let mem = unsafe { llama_cpp_sys_2::llama_get_memory(self.context.as_ptr()) };
        unsafe { llama_cpp_sys_2::llama_memory_seq_pos_max(mem, seq_id) }
    }

    /// Checkpoint a sequence's KV cache state.
    ///
    /// # Parameters
    /// * `seq_id` - The sequence id to checkpoint
    /// * `flags` - Flags to pass to `llama_state_seq_get_size_ext` and `llama_state_seq_get_data_ext` (default: 0)
    ///
    /// # Errors
    ///
    /// Returns an error if the sequence does not exist or has zero size.
    pub fn get_seq_state_ext(
        &self,
        seq_id: i32,
        flags: u32,
    ) -> Result<SeqDataExt, KvCacheConversionError> {
        let size = unsafe {
            llama_cpp_sys_2::llama_state_seq_get_size_ext(
                self.context.as_ptr(),
                seq_id,
                flags,
            )
        };
        if size == 0 {
            return Err(KvCacheConversionError::SequenceNotFound);
        }

        let mut data = vec![0u8; size];
        unsafe {
            llama_cpp_sys_2::llama_state_seq_get_data_ext(
                self.context.as_ptr(),
                data.as_mut_ptr(),
                size,
                seq_id,
                flags,
            );
        }
        Ok(SeqDataExt { data })
    }

    /// Restore a sequence's KV cache state from a checkpoint.
    ///
    /// # Parameters
    /// * `seq_id` - The sequence id to restore
    /// * `data` - The checkpoint data to restore
    /// * `flags` - Flags to pass to `llama_state_seq_set_data_ext` (default: 0)
    pub fn set_seq_state_ext(&mut self, seq_id: i32, data: &SeqDataExt, flags: u32) {
        unsafe {
            llama_cpp_sys_2::llama_state_seq_set_data_ext(
                self.context.as_ptr(),
                data.data.as_ptr(),
                data.data.len(),
                seq_id,
                flags,
            );
        }
    }
}

