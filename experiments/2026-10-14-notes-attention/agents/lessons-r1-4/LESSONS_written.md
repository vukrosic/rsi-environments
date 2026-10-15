The timed problem is contiguous float32 causal attention with batch eight, eight heads, sequence length 1024, and head size 64. Half precision inputs, scores, and probabilities stayed within the relative error tolerance of 0.002. Keep softmax reductions and matrix accumulation in float32 and return float32.

The best measured kernel took 8.30 milliseconds, versus a 63.65 millisecond reference, for a 7.67 times speedup. It converts inputs to half precision, uses a custom Metal matrix kernel for scores, applies a custom causal softmax, and uses PyTorch for the probability times value matrix product. Score tiles have 64 query rows and 64 key rows, with 256 threads. Skip whole tiles above the causal diagonal. Eight SIMD groups each handle eight query rows using Metal SIMD group matrix operations. Accumulate scores in float32, store half, then scale by one eighth in softmax.

The successful softmax uses one 256 thread group per row, four scores per thread, SIMD reductions followed by shared memory reductions, and half output. Its barriers must protect shared scratch reuse.

Measured alternatives: built in float32 causal attention took 16.64 milliseconds; half precision built in attention took 11.67; PyTorch half matrix products with the successful custom softmax took 9.26. A softmax with one SIMD group per row and 32 scores per thread worsened that to 10.52.

Fully fused online attention was slower: query/key tile sizes 32/32, 64/32, and 64/64 took 10.45, 10.40, and 11.25 milliseconds. Reusing shared memory and retaining probabilities in registers with 128/64 tiles took 10.31. One trial failed compilation because Metal max is not a constant expression for array sizes; a ternary expression works.

All timed successful variants passed correctness. GPU contention changes reference times, so compare absolute kernel times too.
