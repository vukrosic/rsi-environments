Notes from the previous round of agents, one section per agent.

## Agent 1

The workload is contiguous float32 causal attention on MPS: eight batches, eight heads, 1024 positions, 64 channels. Only time with bench.py. GPU sharing changes timings; compare absolute times and speedups and treat small differences cautiously.

Use a fused Metal shader compiled at import. The best configuration uses 64 queries and 64 keys, eight SIMD groups, and 256 threads. Each SIMD group handles eight rows. Explicitly unroll constant matrix operations. Keep queries, float32 scores, half probabilities, and float32 output in registers. Stage inputs through four-element vector loads into separate shared half key and value buffers. Reuse key storage after loading queries. Use stable float32 online softmax, skip future tiles, and mask individual elements only on diagonal tiles. Matrix rows span four lanes; reductions need exclusive-or shuffle offsets one and eight.

This round used ten benchmarks. Kernel times in milliseconds: float32 value accumulation, 3.68; sixteen-key half partial sums, 3.81; 32-key tiles, 3.77; direct query loads, 3.90; 128-query tiles, 4.63 during GPU contention; separate diagonal masking branch, 4.08; query prescaling and fast exponentials, 3.89; 32-key half partial sums, 3.69; final sixteen-key partial sums with prescaling, 3.71; prescaled float32 value accumulation, 3.86.

Final speedup was 18.27 times against 67.74 milliseconds. Benchmark error was 0.0012. Thirty fresh checks passed with maximum error 0.001304; final check also passed. Thirty-two-key half accumulation reached 0.0019, too close to tolerance. Keep dot products and running outputs float32.

Earlier rounds measured built-in attention around 17 milliseconds, or 10.5–11.7 with half inputs. Shared scores and missing unrolling were much slower. Preconversion and larger tiles generally regressed. Earlier optimized fused versions reached 3.88–4.10 milliseconds.

## Agent 2

The workload is contiguous float32 causal attention: eight batches, eight heads, 1024 positions, 64 channels. Supplied baseline: 65.6 milliseconds. Time only through bench.py. Shared GPU load varies; small timing differences are uncertain.

Compile a fused Metal shader once at import. Use 64 queries and 64 keys per tile, eight SIMD groups, 256 threads. Keep queries, float32 scores, half probabilities, and float32 output accumulators in registers. Explicitly generate unrolled matrix operations. Cooperatively stage separate half key and value buffers with four-element vector loads. Maintain stable float32 online softmax maxima and denominators. Skip future tiles; mask individual elements only on the diagonal tile. Matrix lanes own two adjacent columns; row reductions use exclusive-or shuffle offsets one and eight.

This round used ten benchmarks. Times in milliseconds: full float32 accumulation, 3.74; sixteen-key half value sums with output columns processed separately, 3.90; base-two exponentials, 3.77; 32-key tiles, 3.82; 128-query tiles, 4.21; padded shared storage reused for keys and values, 4.02; separate padded storage with 32-key tiles, 3.86; interleaved sixteen-key half value sums, 3.66; sixteen-channel half dot-product sums, 3.97; repeat interleaved half value sums, 4.00.

Final kernel retains full float32 accumulation: 3.74 milliseconds versus 66.70, speedup 17.82 times. Benchmark error was 0.00061. Twenty fresh checks passed, maximum error 0.000517 against 0.002. Half value variants had benchmark error 0.0012; short half dot products reached 0.0016.

Earlier useful findings: built-in attention took approximately 17 milliseconds in float32 and 10.5–11.7 with half inputs. Unrolling register matrix loops and vector staging produced major improvements. Shared score buffers and unrolled larger tiles generally lost. Long half dot-product or output accumulation can exceed tolerance.

## Agent 3

The workload is contiguous float32 causal attention on MPS: eight batches, eight heads, 1024 positions, and 64 channels. GPU contention varies reference times; compare absolute kernel times and treat small differences cautiously. Use only bench.py for timing.

The final kernel measured 3.70 milliseconds against a 67.59 millisecond reference, an 18.25 times speedup. Benchmark relative error was 0.00061 versus the 0.002 tolerance. All ten submissions passed.

Compile a fused Metal shader once at import. Use 64 queries and 64 keys per tile, eight SIMD groups, and 256 threads. Each SIMD group handles eight query rows. Keep queries, float32 scores, half probabilities, and float32 outputs in registers. Initialize query registers directly from float inputs. Cooperatively stage keys and values into separate shared half buffers using four-element vector loads. Explicitly generate constant matrix operations. Maintain stable online softmax maxima and denominators in float32. Skip future tiles and apply element masks only on the diagonal tile. Matrix lanes own two adjacent columns; row reductions use exclusive-or shuffle distances one and eight.

Measured alternatives in milliseconds: sixteen-key half value partials, 3.76; 32-key tiles, 3.76; 128-query tiles, 4.19; 128-key tiles, 4.45; sixteen-channel half score partials, 3.74, with benchmark error 0.0016; base-two exponentials plus staging-loop unrolling, 3.89; additionally grouping causal masks, 3.71; reusing one shared key/value buffer with extra barriers, 4.10; grouped masks with ordinary exponentials, 3.81. The original float32 accumulation version was retained.

Earlier rounds found built-in attention around 17 milliseconds in float32 and 10.5 to 11.7 milliseconds with half inputs. Register matrices without explicit unrolling were much slower. Shared scores and probabilities, input preconversion, and larger tiles generally regressed. Long half score or output accumulation can exceed tolerance. Preserve float32 accumulation for accuracy margin.

## Agent 4

The workload is contiguous float32 causal attention on MPS: eight batches, eight heads, 1024 positions, and 64 channels. GPU sharing changes timings; compare absolute kernel times and treat small differences cautiously. Time only with the provided benchmark.

This round used ten benchmarks. The final kernel measured 3.65 milliseconds against a 68.30 millisecond reference, an 18.71 times speedup. Benchmark relative error was 0.00061. Thirty fresh-input checks passed; maximum error was 0.000595 against the 0.002 tolerance.

Compile fused Metal attention once at import. Use 64 queries and 64 keys per tile, eight SIMD groups, and 256 threads. Stage queries through shared half storage, load query matrices into registers, then reuse that storage for keys. Keep separate shared half key and value buffers, totaling sixteen kilobytes. Stage inputs with four-element vector loads. Keep queries, scores, probabilities, and outputs in registers. Use half matrix operands and probabilities, but float32 dot products, scores, output accumulation, maxima, and denominators. Stable online softmax using base-two exponentials was fastest. Skip future tiles; apply element masks only on diagonal tiles. Row reductions use exclusive-or shuffle distances one and eight.

Explicitly expand fixed matrix loops before compilation. Compiler unroll hints alone measured 6.43 milliseconds; literal expansion measured 3.74. Other measurements: half sixteen-key partial sums with hints, 7.24; direct query loads, 3.87; expanded half partial sums, 3.97; 128-query tiles, 4.12; 32-key tiles, 3.90; separate diagonal-mask branch, 3.94; fast exponential intrinsic, 3.97.

Earlier rounds measured built-in attention around seventeen milliseconds in float32 and 10.5 to 11.7 with half inputs. Fused register implementations previously reached 4.10 and 3.88 milliseconds. Shared score buffers, insufficient unrolling, and separate input conversion often regressed. Half dot-product accumulation and long half output accumulation can exceed tolerance.
