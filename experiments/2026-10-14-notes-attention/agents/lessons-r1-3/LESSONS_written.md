The original kernel was reported at 67.6 ms. Ten benchmark submissions were used; reference times varied from 58.89 to 72.08 ms, so compare kernel times as well as speedups.

PyTorch scaled dot product attention with causal masking took 16.81 ms in float32. Converting all three inputs to half, using the same operation, and returning float32 took 10.52 ms. Both passed correctness.

A custom Metal shader compiled through PyTorch fused causal matrix products and online softmax. Half matrix operands with float32 scores, softmax totals, and output accumulation passed comfortably. A shared memory implementation with thirty two query rows and thirty two keys took 10.53 ms. Sixty four query rows took 10.89 ms; explicit loop unrolling took 10.72 ms. Thirty two queries and sixty four keys, with inputs converted to half beforehand, regressed to 15.29 ms.

Removing shared memory and barriers helped slightly: independent SIMD groups each process eight query rows, load matrices directly from half input buffers, and keep probabilities in registers. Thirty two key tiles and float32 accumulators took 10.38 ms.

Half precision score accumulation failed tolerance: the benchmark reported relative error 0.00202, and a separate check reported 0.00230. No valid time was obtained for that submission.

Keeping float32 scores but accumulating value products in half over thirty two keys took 9.78 ms; benchmark error reached 0.0019. Splitting those products into sixteen key blocks before adding them to float32 output accumulators improved both speed and accuracy: 9.44 ms, 6.97 times faster, benchmark error 0.0012. This is the final implementation. Ten additional random input checks passed, with maximum relative error about 0.0010. The shader preserves stable online softmax and skips future key tiles.
