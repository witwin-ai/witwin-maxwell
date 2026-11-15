"""Issue the next feature-group receipt of the consumer metadata bridge (CPU only, no torch import).

A change confined to a feature group (FEATURE_GROUPS in derive_consumer_bridge.py) is admitted by a
receipt instead of a conventional consumer re-receipt. Order on the feature branch:

1. scripts/dev/rfic/evidence_lock.py acquire on a commit C whose witwin/ tree is clean, then
   ``renew`` (writes the renewal with C and the tier-table SHA into the committed evidence);
2. this tool: checks that the evidence files carry one renewal of C, that the tier table under
   the evidence root has that SHA, no unaccepted T2 and the standard's T1 bound, that the group files
   of the working tree equal the git tree of C, and that the core-to-group interface and the op
   namespaces equal the archive's; then writes docs/assessments/consumer-feature-groups/<group>/NNNN.json
   and prints the ``consumer_group_receipt`` value and the note text for the branch's
   [[merge_declarations]] entry in scripts/dev/integration_suites.toml;
3. commit the receipt (``git add -f``: docs/ is ignored by default) with the declaration.

T1 movers are auto-accepted and listed with their max_rel_diff. Accepted T2 entries need
``--approval`` with the user approval reference. Any core, interface, namespace, loader or bridge
change needs a conventional re-receipt (docs/plans/mom_provider_epoch_v3.md section 5) instead.
"""
from pathlib import Path
import argparse,importlib.util,json,os

ROOT=Path(__file__).resolve().parents[4]
PREFIX='<evidence-root>/'


def load_bridge():
    spec=importlib.util.spec_from_file_location('derive_consumer_bridge',Path(__file__).with_name('derive_consumer_bridge.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


bridge=load_bridge()


def baseline(archive):
    """The baseline provenance selected by a consumer archive record (the test's ARCHIVE)."""
    archive=Path(archive);record=json.loads(archive.read_text())['baseline'];proof=archive.parent/Path(record['provenance']).name
    if bridge.digest(proof)!=record['provenance_sha256']:raise SystemExit(f'Baseline provenance SHA differs from the archive record: {proof}')
    return proof,json.loads(proof.read_text())


def resolve(name,evidence_root):
    if not name.startswith(PREFIX):raise SystemExit(f'Tier table must be named under {PREFIX}: {name}')
    return Path(evidence_root)/name[len(PREFIX):]


def build_receipt(source,group,archive,merge_declaration,approval,evidence_root):
    """The receipt for the working tree's group, bound to its evidence lock; refuses nothing and writes nothing."""
    source=Path(source).resolve();proof_path,proof=baseline(archive)
    old_source=bridge.relocate(proof['source_root'],proof.get('relocated_roots',{}))
    admitted=bridge.archive_group(old_source,proof['current_package_before'],group)
    chain=bridge.group_chain(source,group)
    predecessor=dict(predecessor_signature=admitted['signature'],predecessor_receipt=None,predecessor_receipt_sha256=None) if not chain else \
        dict(predecessor_signature=chain[-1][2]['signature'],predecessor_receipt=chain[-1][0],predecessor_receipt_sha256=chain[-1][1])
    evidence={}
    for path in bridge.GROUP_EVIDENCE[group]:
        renewals=json.loads((source/path).read_text()).get('numerical_lock',{}).get('renewals',[])
        if not renewals:raise SystemExit(f'{path} has no numerical_lock renewal: run scripts/dev/rfic/evidence_lock.py renew first')
        evidence[path]=renewals[-1]
    commits={record['acquisition_commit'] for record in evidence.values()};tables={(record['tier_table'],record['tier_table_sha256']) for record in evidence.values()}
    if len(commits)!=1 or len(tables)!=1:raise SystemExit(f'Evidence files carry renewals of different acquisitions: {sorted(commits)}')
    (commit,),((table_name,table_sha),)=commits,tables
    table_path=resolve(table_name,evidence_root)
    if bridge.digest(table_path)!=table_sha:raise SystemExit(f'Tier table SHA differs from the renewal: {table_path}')
    table=json.loads(table_path.read_text());accepted=table['accepted_t2'];files=bridge.group_files(source,group)
    return dict(schema=bridge.RECEIPT_SCHEMA,group=group,prefix=bridge.FEATURE_GROUPS[group],archive=Path(archive).parent.name,
                archive_source_sha256=proof['current_package_before']['source_sha256'],**predecessor,signature=bridge.group_signature(files),files=files,
                interface=bridge.group_interface(source,group),op_namespaces=bridge.op_namespaces(source,group),
                numerical_lock=dict(evidence=evidence,acquisition_commit=commit,tier_table=table_name,tier_table_sha256=table_sha,
                                    t1_bound=dict(relative=bridge.T1_BOUND,statement=bridge.T1_STATEMENT),
                                    t1_movers=[dict(field=m['field'],tier=m['tier'],max_rel_diff=m['max_rel_diff'],delta_over_bound=m['delta_over_bound'])
                                               for m in table['locked_movers'] if m['tier']=='T1'],
                                    accepted_t2=accepted),
                approval=approval or 'T1-auto',merge_declaration=merge_declaration),admitted,chain,table


def note(receipt,name,sha,length):
    """Merge-declaration note text: receipt, bound and every T1 mover with its max_rel_diff."""
    lock=receipt['numerical_lock']
    movers='; '.join(f'{m["field"]} max_rel_diff {m["max_rel_diff"]:.3g}' for m in lock['t1_movers']) or 'none'
    accepted='; '.join(f'{a["field"]} ({a["approval"]})' for a in lock['accepted_t2']) or 'none'
    return (f'Consumer feature group {receipt["group"]}: receipt {name}@{sha} (chain length {length}, signature {receipt["signature"]}, '
            f'predecessor {receipt["predecessor_signature"]}); numerical lock acquisition {lock["acquisition_commit"]}, tier table '
            f'{lock["tier_table"]} sha256 {lock["tier_table_sha256"]}; T1 bound {lock["t1_bound"]["statement"]}; T1 movers: {movers}; '
            f'accepted T2: {accepted}; approval {receipt["approval"]}.')


def issue(source,group,archive,merge_declaration,approval,evidence_root):
    source=Path(source).resolve()
    receipt,admitted,chain,table=build_receipt(source,group,archive,merge_declaration,approval,evidence_root)
    commit=receipt['numerical_lock']['acquisition_commit'];evidence=receipt['numerical_lock']['evidence']
    if table['acquisition_commit']!=commit:raise SystemExit('Tier table names another acquisition commit than the renewal')
    if table['unaccepted_t2'] or any(record['summary']['T2'] for record in evidence.values()):raise SystemExit(f'Unaccepted T2 in the tier table: {table["unaccepted_t2"]}')
    if table['bound']!=bridge.T1_STATEMENT:raise SystemExit(f'Tier table bound differs from the standard: {table["bound"]!r} != {bridge.T1_STATEMENT!r}')
    if table['accepted_t2'] and not (approval or '').strip():raise SystemExit('Accepted T2 entries need --approval with the user approval reference')
    if not table['accepted_t2'] and approval:raise SystemExit('--approval is only for accepted T2; T1 movers are auto-accepted')
    previous=chain[-1][2]['files'] if chain else admitted['files']
    if receipt['files']==previous:raise SystemExit('The group equals the chain head; there is nothing to admit')
    if receipt['interface']!=admitted['interface'] or not receipt['op_namespaces']==admitted['op_namespaces']==bridge.GROUP_OP_NAMESPACES[group]:
        raise SystemExit('The core-to-group interface or the op namespaces changed: a conventional re-receipt is required')
    if bridge.acquisition_tree(source,commit,group)!=receipt['files']:
        raise SystemExit('The working tree group differs from the acquisition commit: re-acquire and renew on this tree')
    directory=source/bridge.GROUP_RECEIPTS/group;directory.mkdir(parents=True,exist_ok=True)
    path=directory/f'{len(chain)+1:04d}.json'
    with path.open('x') as stream:json.dump(receipt,stream,indent=1,sort_keys=True);stream.write('\n')
    name=path.relative_to(source).as_posix();sha=bridge.digest(path)
    return dict(path=name,sha256=sha,declaration=f'consumer_group_receipt = "{name}@{sha}"',note=note(receipt,name,sha,len(chain)+1))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source-root',type=Path,default=ROOT)
    p.add_argument('--group',default='layered',choices=sorted(bridge.FEATURE_GROUPS))
    p.add_argument('--archive',type=Path,required=True,help='consumer archive record selected by tests/api/public/test_consumer_metadata.py (ARCHIVE)')
    p.add_argument('--merge-declaration',required=True,help='branch of the [[merge_declarations]] entry that will carry consumer_group_receipt')
    p.add_argument('--approval',help='user approval reference; only with accepted T2 entries')
    p.add_argument('--evidence-root',type=Path,default=os.environ.get('WITWIN_EVIDENCE_ROOT'),help='resolves <evidence-root>/ (default WITWIN_EVIDENCE_ROOT)')
    a=p.parse_args()
    if a.evidence_root is None:raise SystemExit('--evidence-root or WITWIN_EVIDENCE_ROOT is required')
    print(json.dumps(issue(a.source_root,a.group,a.archive,a.merge_declaration,a.approval,a.evidence_root),indent=1))
