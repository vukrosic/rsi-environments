The supplied unchanged baseline was 67.6 milliseconds. All ten timing runs passed correctness. The fastest measured kernel took 4.99 milliseconds, with a simultaneous reference time of 59.93 milliseconds and a reported speedup of 12.01 times. Its benchmark relative error was 0.00061, below the allowed 0.002.

Built-in causal scaled dot product attention took 16.93 milliseconds using float32. Casting its three inputs to half precision and its output back to float32 reduced this to 10.57 milliseconds.

A custom Metal tiled attention shader improved further. Use half precision inputs and probabilities, float32 dot product accumulation, float32 scores, and stable online softmax with float32 running maxima and sums. Skip key tiles beyond the causal boundary. Return float32 output. Eight SIMD groups, totaling 256 threads, cooperate per query tile using eight by eight SIMD matrix operations.

With 32 queries and 32 keys per tile, shared partial outputs took 6.95 milliseconds. Increasing the key tile to 64 took 5.80 milliseconds. Keeping partial output matrices in registers instead took 5.26 milliseconds. Increasing queries to 64 produced the best 4.99 milliseconds.

Rearranging matrix products to share more operand loads was slower, taking 7.41 milliseconds. Half precision scores sharing storage with probabilities did not improve observed times: 64 queries with 128 keys took 5.98 milliseconds; 64 with 64 took 6.06 milliseconds during a reported busy GPU run; 128 with 64 took 6.39 milliseconds.

Reference times varied from 58.65 to 77.12 milliseconds because the GPU is shared. Treat small differences cautiously. The final implementation retains the fastest measured 64 query, 64 key, float32 score variant.
