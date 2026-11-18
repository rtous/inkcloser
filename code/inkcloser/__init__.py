"""InkCloser: line-art completion for paint-bucket colorization.

Plain-Python port of lineartanimdata3_closer_v12.ipynb. Module <- notebook block:
    data.py        CODE BLOCK 1 (split manifest) and CODE BLOCK 4 (datasets / loaders)
    model.py       CODE BLOCKS 6-9 (positional embedding, attention, dilated stack, U-Net)
    loss.py        CODE BLOCK 10 (LineLoss)
    metrics.py     CODE BLOCK 11 (line_stats / summarize) and the evaluate() of 12a
    viz.py         CODE BLOCKS 12a / 12b (error map, previews, curves)
    paper_eval.py  CODE BLOCK 14 (paper results: baselines, metrics, bootstrap, tables, figures)
"""
