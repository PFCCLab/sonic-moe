# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

# this impl is adapted from QuACK's topk https://github.com/Dao-AILab/quack/blob/main/quack/topk.py
import math
from typing import Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import quack.utils as utils
from cutlass import const_expr
from quack.sort.bitonic_sort import bitonic_topk
from triton import next_power_of_2


class _BaseTopK:
    def __init__(
        self,
        input_dtype: Type[cutlass.Numeric],
        output_dtype: Type[cutlass.Numeric],
        N: int,
        k: int,
        require_softmax_fusion: bool = True,
    ):
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype
        self.N = N
        self.input_vecsize = 128 // input_dtype.width
        self.output_vecsize = 128 // output_dtype.width
        self.k = k
        self.next_power_of_2_N = next_power_of_2(N)
        self.next_power_of_2_K = next_power_of_2(k)
        assert k <= 128 and k <= N
        assert N <= 4096 and N % 8 == 0
        assert input_dtype.width <= output_dtype.width, "input bitwidth must <= output bitwidth"

        self.require_softmax_fusion = require_softmax_fusion

    def _calculate_threads_per_row(self):
        # we want num_elems_per_thread >= self.k
        # and each thread can handle at most 64 elements
        N = self.next_power_of_2_N
        num_threads_per_row = max(min(N // self.k, 32, N // 64), 1)
        return num_threads_per_row

    def _get_tv_layout(self, vecsize):
        N = self.next_power_of_2_N
        num_threads = 128 if N <= 16384 else 256
        threads_per_row = self._calculate_threads_per_row()
        cols_per_block = num_threads // threads_per_row
        num_blocks_N = cute.ceil_div(min(N, 16384) // vecsize, threads_per_row)
        tiler_mn = (cols_per_block, vecsize * num_blocks_N * threads_per_row)
        tv_layout = cute.make_layout(
            ((threads_per_row, cols_per_block), (vecsize, num_blocks_N)),
            stride=(
                (vecsize * cols_per_block, 1),
                (cols_per_block, cols_per_block * vecsize * threads_per_row),
            ),
        )
        return tiler_mn, tv_layout

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mValues: cute.Tensor,
        mIndices: cute.Tensor,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.input_dtype
        assert mValues.element_type == self.output_dtype
        assert mIndices.element_type == cutlass.Int32
        input_tiler_mn, input_tv_layout = self._get_tv_layout(self.input_vecsize)
        output_tiler_mn, output_tv_layout = self._get_tv_layout(self.output_vecsize)

        num_threads = cute.size(input_tv_layout, mode=[0])
        self.kernel(mX, mValues, mIndices, input_tv_layout, input_tiler_mn, output_tv_layout, output_tiler_mn).launch(
            grid=[cute.ceil_div(mX.shape[0], input_tiler_mn[0]), 1, 1],
            block=[num_threads, 1, 1],
            stream=stream,
        )

    def _load_and_process_input(
        self,
        mX: cute.Tensor,
        input_tv_layout: cute.Layout,
        input_tiler_mn: cute.Shape,
    ):
        """Load input data from global memory and convert to f32."""
        import cute

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        shape = mX.shape
        idX = cute.make_identity_tensor(shape)

        # --- Data Loading ---
        mX = utils.domain_offset_i64((bidx * input_tiler_mn[0], 0), mX)
        gX = cute.local_tile(mX, input_tiler_mn, (0, 0))
        cX = cute.local_tile(idX, input_tiler_mn, (bidx, 0))

        copy_atom_load_X = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gX.element_type, num_bits_per_copy=128)
        thr_copy_X = cute.make_tiled_copy(copy_atom_load_X, input_tv_layout, input_tiler_mn).get_slice(tidx)
        tXgX = thr_copy_X.partition_S(gX)
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]

        tXrX = cute.make_fragment_like(tXgX)

        is_even_N = const_expr(shape[1] == input_tiler_mn[1])
        tXpX = (
            utils.predicate_k(thr_copy_X.partition_S(cX), limit=shape[1])
            if const_expr((not is_even_N) or (self.N != self.next_power_of_2_N))
            else None
        )
        if tXcX[0][0] < shape[0]:
            cute.copy(copy_atom_load_X, tXgX, tXrX, pred=tXpX)

        tXrX_f32 = cute.make_fragment(tXrX.shape, cutlass.Float32)
        tXrX_f32.store(tXrX.load().to(cutlass.Float32))

        return tXrX_f32, tXcX, is_even_N, tXpX, shape

    def _apply_activation(self, tXrX_f32: cute.Fragment, threads_per_row: int, input_tv_layout: cute.Layout):
        """Apply activation function. Override in subclasses."""
        raise NotImplementedError

    def _encode_indices(
        self,
        tXrX_f32: cute.Fragment,
        tXcX: cute.Tensor,
        input_tv_layout: cute.Layout,
    ):
        """Encode indices into the bottom bits of values."""
        log_N = int(math.log2(self.next_power_of_2_N))
        idx_mask = const_expr((1 << log_N) - 1)
        input_vecsize = cutlass.const_expr(input_tv_layout.shape[1][0])
        tXrX_u32 = cute.recast_tensor(tXrX_f32, cutlass.Uint32)

        for i in cutlass.range(cute.size(tXrX_u32), unroll_full=True):
            col_idx = cutlass.Uint32(tXcX[i // input_vecsize][1] + i % input_vecsize)
            encoded_idx = ~col_idx
            encoded_idx = encoded_idx & idx_mask
            tXrX_u32[i] = (tXrX_u32[i] & ~idx_mask) | encoded_idx

        return tXrX_u32, idx_mask

    def _extract_and_clean_indices(
        self, topk_vals: cute.Fragment, topk_vals_u32: cute.Fragment, idx_mask: int, k: int
    ):
        """Extract indices and clean values after top-k."""
        topk_indices = cute.make_fragment(k, cutlass.Int32)
        for i in cutlass.range_constexpr(k):
            encoded_idx = topk_vals_u32[i] & idx_mask
            topk_vals_u32[i] = topk_vals_u32[i] & ~idx_mask
            col_idx = ~encoded_idx if topk_vals[i] >= 0 else encoded_idx
            topk_indices[i] = cutlass.Int32(col_idx & idx_mask)
        return topk_indices

    def _store_results(
        self,
        topk_vals: cute.Fragment,
        topk_vals_out: cute.Fragment,
        topk_indices: cute.Fragment,
        mValues: cute.Tensor,
        mIndices: cute.Tensor,
        tXcX: cute.Tensor,
        shape: tuple,
        output_tv_layout: cute.Layout,
    ):
        """Store top-k results to global memory."""
        row = tXcX[0][0]
        output_vecsize = cutlass.const_expr(output_tv_layout.shape[1][0])
        if row < shape[0] and tXcX[0][1] == 0:
            elems_per_store = const_expr(math.gcd(output_vecsize, self.k))
            mValues_store = cute.tiled_divide(mValues[row, None], (elems_per_store,))
            mIndices_store = cute.tiled_divide(mIndices[row, None], (elems_per_store,))
            topk_vals_out_store = cute.tiled_divide(topk_vals_out, (elems_per_store,))
            topk_indices_store = cute.tiled_divide(topk_indices, (elems_per_store,))
            for i in cutlass.range_constexpr(cute.size(topk_vals_out_store.shape, [1])):
                cute.autovec_copy(topk_vals_out_store[None, i], mValues_store[None, i])
                cute.autovec_copy(topk_indices_store[None, i], mIndices_store[None, i])


class TopK_Softmax(_BaseTopK):
    # @cute.kernel
    # def kernel(
    #     self,
    #     mX: cute.Tensor,
    #     mValues: cute.Tensor,
    #     mIndices: cute.Tensor,
    #     input_tv_layout: cute.Layout,
    #     input_tiler_mn: cute.Shape,
    #     output_tv_layout: cute.Layout,
    #     output_tiler_mn: cute.Shape,
    # ):
    #     tXrX_f32, tXcX, is_even_N, tXpX, shape = self._load_and_process_input(mX, input_tv_layout, input_tiler_mn)

    #     # Encode the indices into the bottom bits of values.
    #     log_N = int(math.log2(self.next_power_of_2_N))
    #     idx_mask = const_expr((1 << log_N) - 1)
    #     input_vecsize = cutlass.const_expr(input_tv_layout.shape[1][0])
    #     tXrX_u32 = cute.recast_tensor(tXrX_f32, cutlass.Uint32)
    #     # Encode indices into the last log_N bits of tXrX_u32
    #     for i in cutlass.range(cute.size(tXrX_u32), unroll_full=True):
    #         # tXcX only keeps track of the indices for every @vecsize elements
    #         col_idx = cutlass.Uint32(tXcX[i // input_vecsize][1] + i % input_vecsize)
    #         # If positive, invert the bits of the index, so that if there's a tie,
    #         # indices coming from a earlier column will win.
    #         encoded_idx = ~col_idx if tXrX_f32[i] >= 0 else col_idx
    #         # Mask to keep only the last log_N bits of the encoded index
    #         encoded_idx = encoded_idx & idx_mask
    #         # Clear the last log_N bits and set them to our encoded index
    #         tXrX_u32[i] = (tXrX_u32[i] & ~idx_mask) | encoded_idx

    #     # Fill OOB values with -inf for top-k
    #     if const_expr((not is_even_N) or (self.N != self.next_power_of_2_N)):
    #         utils.fill_oob(tXrX_f32, tXpX, -tXrX_f32.element_type.inf)

    #     threads_per_row = input_tv_layout.shape[0][0]
    #     topk_vals = bitonic_topk(tXrX_f32, self.next_power_of_2_K, warp_width=threads_per_row)

    #     # Extract indices and clean values
    #     topk_vals_u32 = cute.recast_tensor(topk_vals, cutlass.Uint32)
    #     topk_indices = cute.make_fragment(self.k, cutlass.Int32)
    #     for i in cutlass.range_constexpr(self.k):
    #         # Extract the encoded index from the last log_N bits
    #         encoded_idx = topk_vals_u32[i] & idx_mask
    #         # Check if original value was positive by looking at the cleaned value
    #         topk_vals_u32[i] = topk_vals_u32[i] & ~idx_mask  # Clear last log_N bits
    #         # If positive, we need to invert the bits back to get original index
    #         col_idx = ~encoded_idx if topk_vals[i] >= 0 else encoded_idx
    #         topk_indices[i] = cutlass.Int32(col_idx & idx_mask)

    #     if const_expr(self.require_softmax_fusion):
    #         topk_vals_max = -cutlass.Float32.inf
    #         for i in cutlass.range_constexpr(self.k):
    #             topk_vals_max = cute.arch.fmax(topk_vals[i], topk_vals_max)

    #         topk_exp_sum = cutlass.Int32(0.0)
    #         for i in cutlass.range_constexpr(self.k):
    #             topk_vals[i] = cute.math.exp(topk_vals[i] - topk_vals_max)
    #             topk_exp_sum = topk_exp_sum + topk_vals[i]

    #         for i in cutlass.range_constexpr(self.k):
    #             topk_vals[i] = topk_vals[i] / topk_exp_sum

    #     # Convert cleaned values to output type
    #     topk_vals_out = cute.make_fragment_like(topk_indices, mValues.element_type)
    #     for i in cutlass.range_constexpr(self.k):
    #         topk_vals_out[i] = topk_vals[i].to(mValues.element_type)

    #     row = tXcX[0][0]
    #     # Only the 1st thread in this row writes the top-k values and indices
    #     output_vecsize = cutlass.const_expr(output_tv_layout.shape[1][0])
    #     if row < shape[0] and tXcX[0][1] == 0:
    #         # Vectorized write
    #         elems_per_store = const_expr(math.gcd(output_vecsize, self.k))
    #         mValues_store = cute.tiled_divide(mValues[row, None], (elems_per_store,))
    #         mIndices_store = cute.tiled_divide(mIndices[row, None], (elems_per_store,))
    #         topk_vals_out_store = cute.tiled_divide(topk_vals_out, (elems_per_store,))
    #         topk_indices_store = cute.tiled_divide(topk_indices, (elems_per_store,))
    #         for i in cutlass.range_constexpr(cute.size(topk_vals_out_store.shape, [1])):
    #             cute.autovec_copy(topk_vals_out_store[None, i], mValues_store[None, i])
    #             cute.autovec_copy(topk_indices_store[None, i], mIndices_store[None, i])

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mValues: cute.Tensor,
        mIndices: cute.Tensor,
        input_tv_layout: cute.Layout,
        input_tiler_mn: cute.Shape,
        output_tv_layout: cute.Layout,
        output_tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        shape = mX.shape
        idX = cute.make_identity_tensor(shape)
        # slice for CTAs
        # We use domain_offset_i64 to deal with tensors larger than 2^31 elements
        mX = utils.domain_offset_i64((bidx * input_tiler_mn[0], 0), mX)
        gX = cute.local_tile(mX, input_tiler_mn, (0, 0))
        cX = cute.local_tile(idX, input_tiler_mn, (bidx, 0))

        # declare the atoms which will be used later for memory copy
        copy_atom_load_X = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gX.element_type, num_bits_per_copy=128)
        thr_copy_X = cute.make_tiled_copy(copy_atom_load_X, input_tv_layout, input_tiler_mn).get_slice(tidx)
        tXgX = thr_copy_X.partition_S(gX)
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]

        # allocate fragments for gmem->rmem
        tXrX = cute.make_fragment_like(tXgX)

        is_even_N = const_expr(shape[1] == input_tiler_mn[1])
        tXpX = (
            utils.predicate_k(thr_copy_X.partition_S(cX), limit=shape[1])
            if const_expr((not is_even_N) or (self.N != self.next_power_of_2_N))
            else None
        )
        if tXcX[0][0] < shape[0]:
            cute.copy(copy_atom_load_X, tXgX, tXrX, pred=tXpX)
        tXrX_f32 = cute.make_fragment(tXrX.shape, cutlass.Float32)
        tXrX_f32.store(tXrX.load().to(cutlass.Float32))

        # Encode the indices into the bottom bits of values.
        log_N = int(math.log2(self.next_power_of_2_N))
        idx_mask = const_expr((1 << log_N) - 1)
        input_vecsize = cutlass.const_expr(input_tv_layout.shape[1][0])
        tXrX_u32 = cute.recast_tensor(tXrX_f32, cutlass.Uint32)
        # Encode indices into the last log_N bits of tXrX_u32
        for i in cutlass.range(cute.size(tXrX_u32), unroll_full=True):
            # tXcX only keeps track of the indices for every @vecsize elements
            col_idx = cutlass.Uint32(tXcX[i // input_vecsize][1] + i % input_vecsize)
            # If positive, invert the bits of the index, so that if there's a tie,
            # indices coming from a earlier column will win.
            encoded_idx = ~col_idx if tXrX_f32[i] >= 0 else col_idx
            # Mask to keep only the last log_N bits of the encoded index
            encoded_idx = encoded_idx & idx_mask
            # Clear the last log_N bits and set them to our encoded index
            tXrX_u32[i] = (tXrX_u32[i] & ~idx_mask) | encoded_idx

        # Fill OOB values with -inf for top-k
        if const_expr((not is_even_N) or (self.N != self.next_power_of_2_N)):
            utils.fill_oob(tXrX_f32, tXpX, -tXrX_f32.element_type.inf)

        threads_per_row = input_tv_layout.shape[0][0]
        topk_vals = bitonic_topk(tXrX_f32, self.next_power_of_2_K, warp_width=threads_per_row)

        # Extract indices and clean values
        topk_vals_u32 = cute.recast_tensor(topk_vals, cutlass.Uint32)
        topk_indices = cute.make_fragment(self.k, cutlass.Int32)
        for i in cutlass.range_constexpr(self.k):
            # Extract the encoded index from the last log_N bits
            encoded_idx = topk_vals_u32[i] & idx_mask
            # Check if original value was positive by looking at the cleaned value
            topk_vals_u32[i] = topk_vals_u32[i] & ~idx_mask  # Clear last log_N bits
            # If positive, we need to invert the bits back to get original index
            col_idx = ~encoded_idx if topk_vals[i] >= 0 else encoded_idx
            topk_indices[i] = cutlass.Int32(col_idx & idx_mask)

        if const_expr(self.require_softmax_fusion):
            topk_vals_max = -cutlass.Float32.inf
            for i in cutlass.range_constexpr(self.k):
                topk_vals_max = cute.arch.fmax(topk_vals[i], topk_vals_max)

            topk_exp_sum = cutlass.Int32(0.0)
            for i in cutlass.range_constexpr(self.k):
                topk_vals[i] = cute.math.exp(topk_vals[i] - topk_vals_max)
                topk_exp_sum = topk_exp_sum + topk_vals[i]

            for i in cutlass.range_constexpr(self.k):
                topk_vals[i] = topk_vals[i] / topk_exp_sum

        # Convert cleaned values to output type
        topk_vals_out = cute.make_fragment_like(topk_indices, mValues.element_type)
        for i in cutlass.range_constexpr(self.k):
            topk_vals_out[i] = topk_vals[i].to(mValues.element_type)

        row = tXcX[0][0]
        # Only the 1st thread in this row writes the top-k values and indices
        output_vecsize = cutlass.const_expr(output_tv_layout.shape[1][0])
        if row < shape[0] and tXcX[0][1] == 0:
            # Vectorized write
            elems_per_store = const_expr(math.gcd(output_vecsize, self.k))
            mValues_store = cute.tiled_divide(mValues[row, None], (elems_per_store,))
            mIndices_store = cute.tiled_divide(mIndices[row, None], (elems_per_store,))
            topk_vals_out_store = cute.tiled_divide(topk_vals_out, (elems_per_store,))
            topk_indices_store = cute.tiled_divide(topk_indices, (elems_per_store,))
            for i in cutlass.range_constexpr(cute.size(topk_vals_out_store.shape, [1])):
                cute.autovec_copy(topk_vals_out_store[None, i], mValues_store[None, i])
                cute.autovec_copy(topk_indices_store[None, i], mIndices_store[None, i])


class Softmax_TopK(_BaseTopK):
    # @cute.kernel
    # def kernel(
    #     self,
    #     mX: cute.Tensor,
    #     mValues: cute.Tensor,
    #     mIndices: cute.Tensor,
    #     input_tv_layout: cute.Layout,
    #     input_tiler_mn: cute.Shape,
    #     output_tv_layout: cute.Layout,
    #     output_tiler_mn: cute.Shape,
    # ):
    #     tXrX_f32, tXcX, is_even_N, tXpX, shape = self._load_and_process_input(mX, input_tv_layout, input_tiler_mn)

    #     # --- STEP 1: Mask OOB with -inf ---
    #     if const_expr((not is_even_N) or (self.N != self.next_power_of_2_N)):
    #         utils.fill_oob(tXrX_f32, tXpX, -tXrX_f32.element_type.inf)

    #     # --- STEP 2: Softmax Activation (Row-wise Reduction) ---
    #     threads_per_row = input_tv_layout.shape[0][0]

    #     # 2a. Find Max (Reduction) to ensure numerical stability
    #     row_max = -cutlass.Float32.inf
    #     for i in cutlass.range(cute.size(tXrX_f32)):
    #         row_max = cute.arch.fmax(row_max, tXrX_f32[i])

    #     # Warp Reduce Max
    #     mask = 1
    #     while mask < threads_per_row:
    #         other_max = cute.arch.shfl_xor_sync(0xFFFFFFFF, row_max, mask)
    #         row_max = cute.arch.fmax(row_max, other_max)
    #         mask *= 2

    #     # 2b. Compute Exp and Sum
    #     row_sum = cutlass.Float32(0.0)
    #     for i in cutlass.range(cute.size(tXrX_f32)):
    #         tXrX_f32[i] = cute.math.exp(tXrX_f32[i] - row_max)
    #         row_sum += tXrX_f32[i]

    #     # Warp Reduce Sum
    #     mask = 1
    #     while mask < threads_per_row:
    #         other_sum = cute.arch.shfl_xor_sync(0xFFFFFFFF, row_sum, mask)
    #         row_sum += other_sum
    #         mask *= 2

    #     # 2c. Normalize
    #     inv_sum = cutlass.Float32(1.0) / row_sum
    #     for i in cutlass.range(cute.size(tXrX_f32)):
    #         tXrX_f32[i] = tXrX_f32[i] * inv_sum

    #     # --- STEP 3: Encode Indices (on Probabilities) ---
    #     tXrX_u32, idx_mask = self._encode_indices(tXrX_f32, tXcX, input_tv_layout)

    #     # --- STEP 4: TopK ---
    #     topk_vals = bitonic_topk(tXrX_f32, self.next_power_of_2_K, warp_width=threads_per_row)

    #     # --- STEP 5: Extract Indices and Clean Values ---
    #     topk_vals_u32 = cute.recast_tensor(topk_vals, cutlass.Uint32)
    #     topk_indices = self._extract_and_clean_indices(topk_vals, topk_vals_u32, idx_mask, self.k)

    #     # --- Store Results ---
    #     topk_vals_out = cute.make_fragment_like(topk_indices, mValues.element_type)
    #     for i in cutlass.range_constexpr(self.k):
    #         topk_vals_out[i] = topk_vals[i].to(mValues.element_type)

    #     self._store_results(topk_vals, topk_vals_out, topk_indices, mValues, mIndices, tXcX, shape, output_tv_layout)

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mValues: cute.Tensor,
        mIndices: cute.Tensor,
        input_tv_layout: cute.Layout,
        input_tiler_mn: cute.Shape,
        output_tv_layout: cute.Layout,
        output_tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        shape = mX.shape
        idX = cute.make_identity_tensor(shape)

        # --- Data Loading ---
        mX = utils.domain_offset_i64((bidx * input_tiler_mn[0], 0), mX)
        gX = cute.local_tile(mX, input_tiler_mn, (0, 0))
        cX = cute.local_tile(idX, input_tiler_mn, (bidx, 0))

        copy_atom_load_X = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gX.element_type, num_bits_per_copy=128)
        thr_copy_X = cute.make_tiled_copy(copy_atom_load_X, input_tv_layout, input_tiler_mn).get_slice(tidx)
        tXgX = thr_copy_X.partition_S(gX)
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]

        tXrX = cute.make_fragment_like(tXgX)

        is_even_N = const_expr(shape[1] == input_tiler_mn[1])
        tXpX = (
            utils.predicate_k(thr_copy_X.partition_S(cX), limit=shape[1])
            if const_expr((not is_even_N) or (self.N != self.next_power_of_2_N))
            else None
        )
        if tXcX[0][0] < shape[0]:
            cute.copy(copy_atom_load_X, tXgX, tXrX, pred=tXpX)

        tXrX_f32 = cute.make_fragment(tXrX.shape, cutlass.Float32)
        tXrX_f32.store(tXrX.load().to(cutlass.Float32))

        # --- STEP 1: Mask OOB with -inf ---
        # This works for both Softmax and Sigmoid:
        # Softmax: exp(-inf) -> 0
        # Sigmoid: 1 / (1 + exp(-(-inf))) -> 1 / (1 + inf) -> 0
        if const_expr((not is_even_N) or (self.N != self.next_power_of_2_N)):
            utils.fill_oob(tXrX_f32, tXpX, -tXrX_f32.element_type.inf)

        # --- STEP 2: Activation (Sigmoid OR Softmax) ---
        threads_per_row = input_tv_layout.shape[0][0]

        # === Softmax Logic (Row-wise Reduction) ===

        # 2a. Find Max (Reduction) to ensure numerical stability
        row_max = -cutlass.Float32.inf
        for i in cutlass.range(cute.size(tXrX_f32)):
            row_max = cute.arch.fmax(row_max, tXrX_f32[i])

        # Warp Reduce Max
        mask = 1
        while mask < threads_per_row:
            other_max = cute.arch.shfl_xor_sync(0xFFFFFFFF, row_max, mask)
            row_max = cute.arch.fmax(row_max, other_max)
            mask *= 2

        # 2b. Compute Exp and Sum
        row_sum = cutlass.Float32(0.0)
        for i in cutlass.range(cute.size(tXrX_f32)):
            tXrX_f32[i] = cute.math.exp(tXrX_f32[i] - row_max)
            row_sum += tXrX_f32[i]

        # Warp Reduce Sum
        mask = 1
        while mask < threads_per_row:
            other_sum = cute.arch.shfl_xor_sync(0xFFFFFFFF, row_sum, mask)
            row_sum += other_sum
            mask *= 2

        # 2c. Normalize
        inv_sum = cutlass.Float32(1.0) / row_sum
        for i in cutlass.range(cute.size(tXrX_f32)):
            tXrX_f32[i] = tXrX_f32[i] * inv_sum

        # --- STEP 3: Encode Indices (on Probabilities) ---
        log_N = int(math.log2(self.next_power_of_2_N))
        idx_mask = const_expr((1 << log_N) - 1)
        input_vecsize = cutlass.const_expr(input_tv_layout.shape[1][0])
        tXrX_u32 = cute.recast_tensor(tXrX_f32, cutlass.Uint32)

        for i in cutlass.range(cute.size(tXrX_u32), unroll_full=True):
            col_idx = cutlass.Uint32(tXcX[i // input_vecsize][1] + i % input_vecsize)
            # Probabilities are always >= 0 (for both Softmax and Sigmoid)
            encoded_idx = ~col_idx
            encoded_idx = encoded_idx & idx_mask
            tXrX_u32[i] = (tXrX_u32[i] & ~idx_mask) | encoded_idx

        # --- STEP 4: TopK ---
        topk_vals = bitonic_topk(tXrX_f32, self.next_power_of_2_K, warp_width=threads_per_row)

        # --- STEP 5: Extract Indices and Clean Values ---
        topk_vals_u32 = cute.recast_tensor(topk_vals, cutlass.Uint32)
        topk_indices = cute.make_fragment(self.k, cutlass.Int32)
        for i in cutlass.range_constexpr(self.k):
            encoded_idx = topk_vals_u32[i] & idx_mask
            topk_vals_u32[i] = topk_vals_u32[i] & ~idx_mask

            # Decode index
            col_idx = ~encoded_idx if topk_vals[i] >= 0 else encoded_idx
            topk_indices[i] = cutlass.Int32(col_idx & idx_mask)

        # --- Store Results ---
        topk_vals_out = cute.make_fragment_like(topk_indices, mValues.element_type)
        for i in cutlass.range_constexpr(self.k):
            topk_vals_out[i] = topk_vals[i].to(mValues.element_type)

        row = tXcX[0][0]
        output_vecsize = cutlass.const_expr(output_tv_layout.shape[1][0])
        if row < shape[0] and tXcX[0][1] == 0:
            elems_per_store = const_expr(math.gcd(output_vecsize, self.k))
            mValues_store = cute.tiled_divide(mValues[row, None], (elems_per_store,))
            mIndices_store = cute.tiled_divide(mIndices[row, None], (elems_per_store,))
            topk_vals_out_store = cute.tiled_divide(topk_vals_out, (elems_per_store,))
            topk_indices_store = cute.tiled_divide(topk_indices, (elems_per_store,))
            for i in cutlass.range_constexpr(cute.size(topk_vals_out_store.shape, [1])):
                cute.autovec_copy(topk_vals_out_store[None, i], mValues_store[None, i])
                cute.autovec_copy(topk_indices_store[None, i], mIndices_store[None, i])


class Sigmoid_TopK(_BaseTopK):
    # @cute.kernel
    # def kernel(
    #     self,
    #     mX: cute.Tensor,
    #     mValues: cute.Tensor,
    #     mIndices: cute.Tensor,
    #     input_tv_layout: cute.Layout,
    #     input_tiler_mn: cute.Shape,
    #     output_tv_layout: cute.Layout,
    #     output_tiler_mn: cute.Shape,
    # ):
    #     tXrX_f32, tXcX, is_even_N, tXpX, shape = self._load_and_process_input(mX, input_tv_layout, input_tiler_mn)

    #     # --- STEP 1: Mask OOB with -inf ---
    #     if const_expr((not is_even_N) or (self.N != self.next_power_of_2_N)):
    #         utils.fill_oob(tXrX_f32, tXpX, -tXrX_f32.element_type.inf)

    #     # --- STEP 2: Sigmoid Activation (Element-wise) ---
    #     # Formula: 1 / (1 + exp(-x))
    #     for i in cutlass.range(cute.size(tXrX_f32)):
    #         val = tXrX_f32[i]
    #         tXrX_f32[i] = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.math.exp(-val))

    #     # --- STEP 3: Encode Indices (on Probabilities) ---
    #     tXrX_u32, idx_mask = self._encode_indices(tXrX_f32, tXcX, input_tv_layout)

    #     # --- STEP 4: TopK ---
    #     threads_per_row = input_tv_layout.shape[0][0]
    #     topk_vals = bitonic_topk(tXrX_f32, self.next_power_of_2_K, warp_width=threads_per_row)

    #     # --- STEP 5: Extract Indices and Clean Values ---
    #     topk_vals_u32 = cute.recast_tensor(topk_vals, cutlass.Uint32)
    #     topk_indices = self._extract_and_clean_indices(topk_vals, topk_vals_u32, idx_mask, self.k)

    #     # --- Store Results ---
    #     topk_vals_out = cute.make_fragment_like(topk_indices, mValues.element_type)
    #     for i in cutlass.range_constexpr(self.k):
    #         topk_vals_out[i] = topk_vals[i].to(mValues.element_type)

    #     self._store_results(topk_vals, topk_vals_out, topk_indices, mValues, mIndices, tXcX, shape, output_tv_layout)

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mValues: cute.Tensor,
        mIndices: cute.Tensor,
        input_tv_layout: cute.Layout,
        input_tiler_mn: cute.Shape,
        output_tv_layout: cute.Layout,
        output_tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        shape = mX.shape
        idX = cute.make_identity_tensor(shape)

        # --- Data Loading ---
        mX = utils.domain_offset_i64((bidx * input_tiler_mn[0], 0), mX)
        gX = cute.local_tile(mX, input_tiler_mn, (0, 0))
        cX = cute.local_tile(idX, input_tiler_mn, (bidx, 0))

        copy_atom_load_X = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gX.element_type, num_bits_per_copy=128)
        thr_copy_X = cute.make_tiled_copy(copy_atom_load_X, input_tv_layout, input_tiler_mn).get_slice(tidx)
        tXgX = thr_copy_X.partition_S(gX)
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]

        tXrX = cute.make_fragment_like(tXgX)

        is_even_N = const_expr(shape[1] == input_tiler_mn[1])
        tXpX = (
            utils.predicate_k(thr_copy_X.partition_S(cX), limit=shape[1])
            if const_expr((not is_even_N) or (self.N != self.next_power_of_2_N))
            else None
        )
        if tXcX[0][0] < shape[0]:
            cute.copy(copy_atom_load_X, tXgX, tXrX, pred=tXpX)

        tXrX_f32 = cute.make_fragment(tXrX.shape, cutlass.Float32)
        tXrX_f32.store(tXrX.load().to(cutlass.Float32))

        # --- STEP 1: Mask OOB with -inf ---
        # This works for both Softmax and Sigmoid:
        # Softmax: exp(-inf) -> 0
        # Sigmoid: 1 / (1 + exp(-(-inf))) -> 1 / (1 + inf) -> 0
        if const_expr((not is_even_N) or (self.N != self.next_power_of_2_N)):
            utils.fill_oob(tXrX_f32, tXpX, -tXrX_f32.element_type.inf)

        # --- STEP 2: Activation (Sigmoid OR Softmax) ---
        threads_per_row = input_tv_layout.shape[0][0]

        # === Sigmoid Logic (Element-wise) ===
        # Formula: 1 / (1 + exp(-x))
        for i in cutlass.range(cute.size(tXrX_f32)):
            val = tXrX_f32[i]
            # Check for -inf to avoid NaN during arithmetic if generic exp handles it poorly
            # (Though standard exp(-inf) is 0, so exp(inf) is inf. 1/inf is 0. Safe.)
            tXrX_f32[i] = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.math.exp(-val))

        # --- STEP 3: Encode Indices (on Probabilities) ---
        log_N = int(math.log2(self.next_power_of_2_N))
        idx_mask = const_expr((1 << log_N) - 1)
        input_vecsize = cutlass.const_expr(input_tv_layout.shape[1][0])
        tXrX_u32 = cute.recast_tensor(tXrX_f32, cutlass.Uint32)

        for i in cutlass.range(cute.size(tXrX_u32), unroll_full=True):
            col_idx = cutlass.Uint32(tXcX[i // input_vecsize][1] + i % input_vecsize)
            # Probabilities are always >= 0 (for both Softmax and Sigmoid)
            encoded_idx = ~col_idx
            encoded_idx = encoded_idx & idx_mask
            tXrX_u32[i] = (tXrX_u32[i] & ~idx_mask) | encoded_idx

        # --- STEP 4: TopK ---
        topk_vals = bitonic_topk(tXrX_f32, self.next_power_of_2_K, warp_width=threads_per_row)

        # --- STEP 5: Extract Indices and Clean Values ---
        topk_vals_u32 = cute.recast_tensor(topk_vals, cutlass.Uint32)
        topk_indices = cute.make_fragment(self.k, cutlass.Int32)
        for i in cutlass.range_constexpr(self.k):
            encoded_idx = topk_vals_u32[i] & idx_mask
            topk_vals_u32[i] = topk_vals_u32[i] & ~idx_mask

            # Decode index
            col_idx = ~encoded_idx if topk_vals[i] >= 0 else encoded_idx
            topk_indices[i] = cutlass.Int32(col_idx & idx_mask)

        # --- Store Results ---
        topk_vals_out = cute.make_fragment_like(topk_indices, mValues.element_type)
        for i in cutlass.range_constexpr(self.k):
            topk_vals_out[i] = topk_vals[i].to(mValues.element_type)

        row = tXcX[0][0]
        output_vecsize = cutlass.const_expr(output_tv_layout.shape[1][0])
        if row < shape[0] and tXcX[0][1] == 0:
            elems_per_store = const_expr(math.gcd(output_vecsize, self.k))
            mValues_store = cute.tiled_divide(mValues[row, None], (elems_per_store,))
            mIndices_store = cute.tiled_divide(mIndices[row, None], (elems_per_store,))
            topk_vals_out_store = cute.tiled_divide(topk_vals_out, (elems_per_store,))
            topk_indices_store = cute.tiled_divide(topk_indices, (elems_per_store,))
            for i in cutlass.range_constexpr(cute.size(topk_vals_out_store.shape, [1])):
                cute.autovec_copy(topk_vals_out_store[None, i], mValues_store[None, i])
                cute.autovec_copy(topk_indices_store[None, i], mIndices_store[None, i])
