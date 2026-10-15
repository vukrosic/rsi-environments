The workload is contiguous float32 causal attention: batch eight, eight heads, 1024 positions, head size 64. The supplied baseline was 67.1 milliseconds. GPU contention varies reference times; compare absolute kernel times and treat small differences cautiously.

Earlier rounds found built-in causal attention around 17 milliseconds in float32 and 10.5 to 11.7 milliseconds with half inputs. Fused Metal attention avoids materializing scores. Earlier best float32 accumulation measured 4.99 milliseconds. Compile shaders once at import.

This round used all ten benchmarks. A 64 query by 64 key shader with shared float32 scores took 8.29 milliseconds. Keeping scores and probabilities in registers without explicit unrolling worsened this to 11.86. Fully unrolling constant matrix loops reduced it to 4.38. With unrolling, 64 queries and 32 keys took 4.40; 32 queries and 64 keys took 5.86; 128 queries and 64 keys took 4.95 during a busy GPU run. A separate vectorized input conversion pass took 4.81.

Sixteen-key half precision value-product partial sums, added into float32 running accumulators, took 4.24 milliseconds. Vectorized four-element staging took 4.25. Applying causal element masks only on the diagonal tile improved this to 3.88 milliseconds versus a 68.38 millisecond reference, a 17.60 times speedup. This is the final implementation.

Use eight SIMD groups, 256 threads, shared half key/value tiles, register queries and probabilities, float32 scores, stable float32 online softmax, and float32 running output. Skip future key tiles. On M4, matrix lanes own two adjacent columns; row reductions use lane shuffle offsets one and eight.

Long half precision accumulation is unsafe. Restrict it to sixteen-key partial sums. Thirty fresh-input checks passed; the final twenty had maximum relative error 0.00114, below 0.002. Benchmark error was 0.0012.
