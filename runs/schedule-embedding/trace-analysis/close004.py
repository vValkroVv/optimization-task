"""Create an immutable capture-only acceptance linked to original native proofs."""
import json
from capture004 import HERE, RAW, ROOT, digest, ref

PROOFS = {
    'd0': ('D0','f3420f9d180c18e1e70405f92d454d91bf0910287f33d3346316326797390709'),
    's': ('S','16f21294dcbf10f4f5a4822069d458f3124214e7de8af82cfb3926f60f2f778d'),
    'se': ('SE','c4d8409cf0c374bf3bac15ab2d1cfb126725a41b4b3258d9a31e38e9d000fa3d'),
    'se-fused': ('SE-fused','612880ebb5382b1b23c4fc0d674aaf899ddff3aa303221e7b91c5bc52d0fa988'),
}


def main():
    comparison = json.loads((HERE/'comparison-004.json').read_text())
    identities = json.loads((HERE/'identities-004.json').read_text())
    captures=[]
    for tag,(variant,sha) in PROOFS.items():
        native=RAW.parent/'adjudication-diagnostic-004'/variant/'adjudication.json'
        assert digest(native)==sha
        proof=json.loads(native.read_text())
        assert proof['status']=='DIAGNOSTIC_PARSER_PASS_TAIL_PENDING'
        assert proof['native_capture_status']=='NATIVE_PROFILE_PARSER_PASS'
        ranks=[]
        for rank in (0,1):
            identity=next(r for r in identities if r['tag']==tag+'-004' and r['rank']==rank)
            expected=proof['native_capture_proof']['expected_original_trace_hashes'][f'native/profiling/traces/iteration_21/rank{rank}_trace.json']
            assert expected==identity['original_sha256']
            result=comparison['variants'][tag]['ranks'][rank]
            assert identity['verified_permitted_changes_only'] and result['tail_coverage']['status']=='PASS'
            assert not result['errors']
            ranks.append({'rank':rank,'identity':identity,'tail_coverage':result['tail_coverage'],
                          'census':result['census'],'flow_coverage':result['flows'],
                          'all_original_complete_category_counts_match':True,
                          'mcp_responses':result['sources']})
        captures.append({'variant':variant,'status':'DIAGNOSTIC_CAPTURE_ACCEPTED',
                         'native_proof':ref(native),'ranks':ranks})
    queries=json.loads((HERE/'queries.json').read_text())
    sql=dict(queries);sql['device_pages']=[queries['device']+f' LIMIT 4000 OFFSET {offset}' for offset in (0,4000,8000)]
    extra=json.loads((HERE/'queries-extra-004.json').read_text());sql.update(extra)
    sql['tail_pages']=[extra['tail']+f' LIMIT 4000 OFFSET {offset}' for offset in (0,4000,8000)]
    receipt={'status':'ALL_FOUR_DIAGNOSTIC_CAPTURES_ACCEPTED','overall_capture_acceptance_complete':True,
             'scope':'Capture integrity and native completion only. Schedule and embedding mechanism outcomes are separate; the requested L7 compiled-compute overlap is absent.',
             'native_proofs_remain_unmodified':True,'captures':captures,
             'comparison':ref(HERE/'comparison-004.json'),'detail':comparison['detail'],
             'report':ref(HERE/'comparison-004.md'),
             'verification':ref(HERE/'verification-004.json'),
             'supersedes':{'receipt':ref(HERE/'capture-acceptance-004.json'),
                           'reason':'Required later-backward gap evidence extended derived comparison/method files after the initial receipt. The initial receipt is preserved; its derived references are superseded. Original hashes and native proofs are unchanged.'},
             'sql':sql,'loads':ref(RAW/'loads-004.json'),
             'methods':[ref(HERE/name) for name in ('prepare.py','analyze.py','capture004.py','close004.py','timeline004.py')],
             'timelines':[ref(HERE/f'four-way-004-rank{rank}.html') for rank in (0,1)]}
    with (HERE/'capture-acceptance-final-004.json').open('x') as f:
        json.dump(receipt,f,indent=2);f.write('\n')
    print(receipt['status'])


if __name__=='__main__':main()
