import unittest

STAGES = 8
CP = 8
LOOP = STAGES
TARGET = 16

BARRIER = ("barrier", None)


def _stage_ops(stage, it, defer_commit):
    first, last = stage == 0, stage == STAGES - 1
    ops = []
    if not first:
        ops += [("recv", "reqs"), BARRIER]
    if not last:
        ops.append(("wait", "reqs"))
    if not first:
        ops += [("recv", "boot"), BARRIER]
    ops.append(BARRIER)
    if not last:
        ops.append(("wait", "boot"))
    if not first:
        ops += [("recv", "trans"), BARRIER]
    ops.append(BARRIER)
    if not last:
        ops.append(("wait", "trans"))

    if defer_commit:
        ops += [("wait", "cons"), ("wait", "rel")]
    if last or it >= LOOP:
        ops += [("send", "cons"), ("send", "rel")]
    if it >= LOOP - 1:
        ops += [("recv", "cons"), BARRIER]
    if not defer_commit:
        ops.append(("wait", "cons"))
    if it >= LOOP - 1:
        ops += [("recv", "rel"), BARRIER]
    if not defer_commit:
        ops.append(("wait", "rel"))

    if not last:
        ops += [("send", "reqs"), ("send", "boot"), ("send", "trans")]
    return ops


class _Rank:
    def __init__(self, stage, cp_rank, defer_commit):
        self.stage, self.cp_rank, self.defer = stage, cp_rank, defer_commit
        self.it, self.pc, self.pending = 0, 0, {}
        self.ops = self._ops()

    def _ops(self):
        ops = _stage_ops(self.stage, self.it, self.defer)
        return ops if self.cp_rank == 0 else [o for o in ops if o is BARRIER]

    def advance(self):
        self.pc += 1
        if self.pc == len(self.ops):
            self.it, self.pc = self.it + 1, 0
            self.ops = self._ops()

    @property
    def op(self):
        return self.ops[self.pc]


def run(defer_commit, stale=0):
    ranks = [_Rank(s, c, defer_commit) for s in range(STAGES) for c in range(CP)]
    stages = [[r for r in ranks if r.stage == s] for s in range(STAGES)]
    chan = {(ring, STAGES - 1): [-1] * stale for ring in ("cons", "rel")}
    consumed, seq = set(), [0]

    def step(r):
        kind, ring = op = r.op
        if op is BARRIER:
            if not all(x.op is BARRIER for x in stages[r.stage]):
                return False
            for x in stages[r.stage]:
                x.advance()
            return True
        if kind == "send":
            seq[0] += 1
            chan.setdefault((ring, r.stage), []).append(seq[0])
            r.pending[ring] = seq[0]
            r.advance()
            return True
        if kind == "recv":
            queue = chan.get((ring, (r.stage - 1) % STAGES))
            if not queue:
                return False
            consumed.add(queue.pop(0))
            r.advance()
            return True
        msg = r.pending.get(ring)
        if msg is not None and msg not in consumed:
            return False
        r.advance()
        return True

    while True:
        progressed = False
        for r in ranks:
            progressed |= step(r)
        if min(r.it for r in ranks) >= TARGET:
            return True, ranks
        if not progressed:
            return False, ranks


class TestPPDisaggPyobjRing(unittest.TestCase):
    def test_live_without_skew(self):
        for defer in (False, True):
            self.assertTrue(run(defer, stale=0)[0], defer)

    def test_same_iteration_commit_deadlocks_on_one_stale_message(self):
        live, ranks = run(defer_commit=False, stale=2)
        self.assertFalse(live)
        parked = {r.stage: r.op for r in ranks if r.cp_rank == 0}
        self.assertTrue(all(op[0] == "wait" for op in parked.values()), parked)
        self.assertEqual(parked[STAGES - 1], ("wait", "rel"))

    def test_deferred_commit_absorbs_the_skew(self):
        for stale in (1, 2, 3):
            self.assertTrue(run(defer_commit=True, stale=stale)[0], stale)


if __name__ == "__main__":
    unittest.main()
