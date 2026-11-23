"""Generate the multiple-panel variant from the proven two-panel replay and FP64 checker."""
import sys
from pathlib import Path


def swap(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def edits(source, old_new):
    for old,new in old_new:
        source = swap(source,old,new)
    return source


def main():
    directory = Path(__file__).resolve().parent
    parent = Path(sys.argv[1])
    replay = (parent/"panel_replay.py").read_text()
    replay = edits(replay,[
        ('def project(hidden, weight, workspace):', '''def ranges(split, vocab):
    return [(start,min(start+split,vocab)) for start in range(0,vocab,split)]


def project(hidden, weight, workspace):'''),
        ('    if FINAL:\n        maximum = tl.load(MAX + row)', '    if START != 0:\n        maximum = tl.load(MAX + row)'),
        ('        tl.store(CHOSEN + row, chosen)', '''        if START != 0:
            chosen = tl.where(hit, chosen, tl.load(CHOSEN + row))
        tl.store(CHOSEN + row, chosen)'''),
        ('rows * max(split, vocab - split)','rows * split'),
        ('for start, panel in ((0, weight[:split]), (split, weight[split:])):', 'for start, stop in ranges(split,vocab):\n            panel = weight[start:stop]'),
        ('vocab, start != 0, 4096','vocab, stop == vocab, 4096'),
        ('for start, stop in ((split, vocab), (0, split)):', 'for start, stop in reversed(ranges(split,vocab)):'),
        ('if start else project(h, panel, workspace)', 'if stop == vocab else project(h, panel, workspace)'),
        ('            if start:\n                torch.mm', '            if stop == vocab:\n                torch.mm'),
        ('widths = args.split, tuning["vocab"] - args.split', 'widths = [stop-start for start,stop in ranges(args.split,tuning["vocab"])]'),
        ('args.chunks * (3 if panel else 1)', 'args.chunks * (2 * len(ranges(args.split, 128256)) - 1 if panel else 1)'),
        ('(2 if panel else 1)\n    assert gather','(len(ranges(args.split, 128256)) if panel else 1)\n    assert gather')
    ])
    (directory/"multi_replay.py").write_text(replay)
    numeric = (parent/"panel_numerics.py").read_text()
    numeric=edits(numeric,[
        ('import panel_replay','import multi_replay as panel_replay'),
        ('rows * max(split, vocab - split)','rows * split'),
        ('for start, stop in ((0, split), (split, vocab)):','for start, stop in panel_replay.ranges(split, vocab):'),
        ('vocab, start != 0, 4096','vocab, stop == vocab, 4096'),
        ('for start, stop in ((split, vocab), (0, split)):', 'for start, stop in reversed(panel_replay.ranges(split, vocab)):'),
        ('if start else panel_replay.project','if stop == vocab else panel_replay.project'),
        ('        if start:\n            torch.mm','        if stop == vocab:\n            torch.mm'),
    ])
    (directory/"multi_numerics.py").write_text(numeric)


if __name__=="__main__":
    main()
