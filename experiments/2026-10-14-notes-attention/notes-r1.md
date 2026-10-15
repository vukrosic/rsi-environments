Notes from the previous round of agents, one section per agent.

## Agent 1

The original kernel was reported as 67.6 milliseconds. All ten benchmark runs were used. Reference times varied from 57.91 to 72.17 milliseconds, so compare absolute kernel times as well as speedups.

Built-in causal scaled dot product attention took 17.01 milliseconds in float32 and 10.64 milliseconds with half inputs and a float output. Both passed.

The fastest tested implementation was a custom Metal tiled attention shader using SIMD group matrix operations. Each SIMD group handles eight query rows and all 64 output columns. Stage queries, keys, and values as half precision; accumulate scores and outputs in float32. Keep online softmax maxima and denominators in float32, reduce each row using lane shuffles, and skip wholly future key tiles. This avoids materializing the full attention matrix. Compile the shader once at import.

Query/key tile sizes and times: 32 by 64, 8.04 milliseconds; 64 by 64, 7.04 milliseconds; 64 by 32, 7.53 milliseconds; 128 by 64, 10.61 milliseconds. The final kernel restores 64 by 64, which achieved an 8.23-fold benchmark speedup and relative error 0.00061.

Other slower experiments: sixteen query rows per SIMD group with 64 by 32 tiles, 8.55 milliseconds; vectorized four-element staging, 7.43 milliseconds; preconverting keys and values with PyTorch, 8.80 milliseconds; reusing query shared storage for keys, 7.77 milliseconds.

Half precision output accumulation is unsafe: repeated fresh-input checks produced errors up to 0.00364, exceeding the 0.002 tolerance. Retain float32 accumulation.

## Agent 2

The supplied unchanged baseline was 67.6 milliseconds. All ten timing runs passed correctness. The fastest measured kernel took 4.99 milliseconds, with a simultaneous reference time of 59.93 milliseconds and a reported speedup of 12.01 times. Its benchmark relative error was 0.00061, below the allowed 0.002.

Built-in causal scaled dot product attention took 16.93 milliseconds using float32. Casting its three inputs to half precision and its output back to float32 reduced this to 10.57 milliseconds.

A custom Metal tiled attention shader improved further. Use half precision inputs and probabilities, float32 dot product accumulation, float32 scores, and stable online softmax with float32 running maxima and sums. Skip key tiles beyond the causal boundary. Return float32 output. Eight SIMD groups, totaling 256 threads, cooperate per query tile using eight by eight SIMD matrix operations.

With 32 queries and 32 keys per tile, shared partial outputs took 6.95 milliseconds. Increasing the key tile to 64 took 5.80 milliseconds. Keeping partial output matrices in registers instead took 5.26 milliseconds. Increasing queries to 64 produced the best 4.99 milliseconds.

Rearranging matrix products to share more operand loads was slower, taking 7.41 milliseconds. Half precision scores sharing storage with probabilities did not improve observed times: 64 queries with 128 keys took 5.98 milliseconds; 64 with 64 took 6.06 milliseconds during a reported busy GPU run; 128 with 64 took 6.39 milliseconds.

Reference times varied from 58.65 to 77.12 milliseconds because the GPU is shared. Treat small differences cautiously. The final implementation retains the fastest measured 64 query, 64 key, float32 score variant.

## Agent 3

The original kernel was reported at 67.6 ms. Ten benchmark submissions were used; reference times varied from 58.89 to 72.08 ms, so compare kernel times as well as speedups.

PyTorch scaled dot product attention with causal masking took 16.81 ms in float32. Converting all three inputs to half, using the same operation, and returning float32 took 10.52 ms. Both passed correctness.

A custom Metal shader compiled through PyTorch fused causal matrix products and online softmax. Half matrix operands with float32 scores, softmax totals, and output accumulation passed comfortably. A shared memory implementation with thirty two query rows and thirty two keys took 10.53 ms. Sixty four query rows took 10.89 ms; explicit loop unrolling took 10.72 ms. Thirty two queries and sixty four keys, with inputs converted to half beforehand, regressed to 15.29 ms.

Removing shared memory and barriers helped slightly: independent SIMD groups each process eight query rows, load matrices directly from half input buffers, and keep probabilities in registers. Thirty two key tiles and float32 accumulators took 10.38 ms.

Half precision score accumulation failed tolerance: the benchmark reported relative error 0.00202, and a separate check reported 0.00230. No valid time was obtained for that submission.

Keeping float32 scores but accumulating value products in half over thirty two keys took 9.78 ms; benchmark error reached 0.0019. Splitting those products into sixteen key blocks before adding them to float32 output accumulators improved both speed and accuracy: 9.44 ms, 6.97 times faster, benchmark error 0.0012. This is the final implementation. Ten additional random input checks passed, with maximum relative error about 0.0010. The shader preserves stable online softmax and skips future key tiles.

## Agent 4

The timed problem is contiguous float32 causal attention with batch eight, eight heads, sequence length 1024, and head size 64. Half precision inputs, scores, and probabilities stayed within the relative error tolerance of 0.002. Keep softmax reductions and matrix accumulation in float32 and return float32.

The best measured kernel took 8.30 milliseconds, versus a 63.65 millisecond reference, for a 7.67 times speedup. It converts inputs to half precision, uses a custom Metal matrix kernel for scores, applies a custom causal softmax, and uses PyTorch for the probability times value matrix product. Score tiles have 64 query rows and 64 key rows, with 256 threads. Skip whole tiles above the causal diagonal. Eight SIMD groups each handle eight query rows using Metal SIMD group matrix operations. Accumulate scores in float32, store half, then scale by one eighth in softmax.

The successful softmax uses one 256 thread group per row, four scores per thread, SIMD reductions followed by shared memory reductions, and half output. Its barriers must protect shared scratch reuse.

Measured alternatives: built in float32 causal attention took 16.64 milliseconds; half precision built in attention took 11.67; PyTorch half matrix products with the successful custom softmax took 9.26. A softmax with one SIMD group per row and 32 scores per thread worsened that to 10.52.

Fully fused online attention was slower: query/key tile sizes 32/32, 64/32, and 64/64 took 10.45, 10.40, and 11.25 milliseconds. Reusing shared memory and retaining probabilities in registers with 128/64 tiles took 10.31. One trial failed compilation because Metal max is not a constant expression for array sizes; a ternary expression works.

All timed successful variants passed correctness. GPU contention changes reference times, so compare absolute kernel times too.
