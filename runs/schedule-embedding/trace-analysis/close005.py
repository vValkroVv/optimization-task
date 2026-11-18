"""Supersede derived ownership labels while reusing accepted eight-capture proof."""
import json
from capture004 import HERE, RAW, ROOT, digest, ref


def main():
    previous = HERE/'capture-acceptance-final-004.json'
    receipt = json.loads(previous.read_text())
    assert digest(previous) == '6451cc05cb8ff0404f31794d3dd76bbd30b65b78c8cf0ef3903a115162f65bff'
    result = json.loads((HERE/'comparison-004.json').read_text())
    archive = json.loads((HERE/'ownership-correction-005-archive.json').read_text())
    for entry in archive:
        assert digest(ROOT/entry['archive']) == entry['sha256']
    for tag, variant in result['variants'].items():
        captured = next(c for c in receipt['captures'] if c['variant'].lower() == tag)
        for rank in variant['ranks']:
            assert captured['ranks'][rank['rank']]['tail_coverage'] == rank['tail_coverage']
            assert captured['ranks'][rank['rank']]['census'] == rank['census']
            assert captured['ranks'][rank['rank']]['flow_coverage'] == rank['flows']
            for micro in rank['microbatches']:
                checks = micro['semantic_checks']
                assert checks['l7_ag_overlap_owned_mtp_prefix_status'] == 'UNESTABLISHED'
                assert checks['ffn_rs_overlap_owned_mtp_prefix_status'] == 'UNESTABLISHED'
                assert checks['l7_compute_parent_rs_union_overlap_ms'] == 0
    receipt['scope'] = 'Capture integrity/native completion remain accepted. Corrected generic overlap labels do not establish MTP-prefix ownership. L7 compiled-compute overlap remains zero.'
    receipt['comparison'] = ref(HERE/'comparison-004.json')
    receipt['detail'] = result['detail']
    receipt['report'] = ref(HERE/'comparison-004.md')
    receipt['verification'] = ref(HERE/'verification-005.json')
    receipt['supersedes'] = {'receipt': ref(previous), 'archive_manifest': ref(HERE/'ownership-correction-005-archive.json'),
        'reason': 'Correct two generic local-kernel overlap labels; add explicit UNESTABLISHED MTP-prefix ownership status and exact head_to_ffn_gap proxy definition. Reuse all accepted capture counts/flows/tails without new MCP queries.'}
    receipt['methods'] = [ref(HERE/name) for name in ('prepare.py','analyze.py','capture004.py','close005.py','timeline004.py')]
    with (HERE/'capture-acceptance-final-005.json').open('x') as f:
        json.dump(receipt,f,indent=2);f.write('\n')


if __name__ == '__main__':
    main()
