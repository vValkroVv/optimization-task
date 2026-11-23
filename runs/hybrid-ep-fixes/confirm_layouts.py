from functools import partial

import tune_routing as tuner

tuner.TILES = ((8, 8), (8, 4), (4, 4), (4, 2), (2, 1), (1, 1))
tuner.TRIALS = 12
for top_k, layout in ((4, (4, 4)), (8, (4, 4)), (16, (4, 4)), (16, (1, 1))):
    variants = {
        "current": tuner.bench.current.indices_to_map,
        "grouped_8x128_wrapped": partial(tuner.wrapped_tile, rows=layout[0], warps=layout[1]),
    }
    tuner.bench.check_sentinels(variants, 128, top_k)
    print(f"Sentinel parity passed: top_k={top_k}, rows={layout[0]}, warps={layout[1]}", flush=True)
tuner.main()
