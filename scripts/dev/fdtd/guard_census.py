"""Generate a guard review proposal; never automatically approve new debt.

Run with --proposal /tmp/guard-review.json after the source tree is frozen.
The proposal preserves reviews for unchanged guard identities and marks new or
changed boundaries as pending. A reviewer must supply the missing evidence.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT=Path(__file__).resolve().parents[3]
MANIFEST=ROOT/'docs/reference/fdtd-capability-guards.json'
BASELINE=ROOT/'docs/assessments/pcb-development-2026-10-22/guard-census/baseline.json'
BASELINE_COMMIT='10d351310b36e9681867347e772a21977e234bdf'
BASELINE_SHA256='ca0e1ed05e4dcb0a9c301369c3a70f7cfda44d96609168c98bc9aac6179e99f9'
BASELINE_DETAILED=BASELINE.with_name('baseline-detailed.json')
BASELINE_DETAILED_SHA256='f1751feb7d0b06c0c0be05b06e78d2e3c102e983fd5518dce9d793d737e1034c'
RENAME_RECEIPT=ROOT/'scripts/dev/mom_rename_receipt.json'
# 0.7.0 phase 3b: identities of raises migrated to UnsupportedFeatureError, written by
# scripts/dev/codemods/070_p3b_raise_migration.py (old identity -> new identity and expression).
RAISE_RECEIPT=ROOT/'scripts/dev/fdtd/raise_migration_receipt.json'
# 0.7.0 phase 7b: guard predicates read prepared declarations through prepared.scene; identities whose
# enclosing conditions changed (old identity -> new identity and conditions).
P7B_RECEIPT=ROOT/'scripts/dev/fdtd/p7b_guard_receipt.json'
# fix/strict-scene-reads-2: simulation.py reads Scene fields strictly; identities whose enclosing conditions
# changed (old identity -> new identity and conditions), applied after the p7b re-key.
STRICT_READS_2_RECEIPT=ROOT/'scripts/dev/fdtd/strict_reads_2_guard_receipt.json'
# the coordinator, 2026-10-31: 740c6fdd changed an invalid-input check to ValueError; contract list 29->28.
# 0.7.0 plan Q10 (user decision 2026-11-08), api070/p5b: the BJT/MOSFET placeholders are removed; 28->26.
CONTRACT_SHA256='fd1468d1baa2d4b36e80e6db4d76f906e069285236e6ddb92b0d9cc9094ec3e2'


# These safety boundaries are capability refusals even though their public
# contracts use ValueError or a dedicated runtime exception. Keep the ordinary
# input-validation ValueErrors outside the capability inventory.
SAFETY_GUARDS = {
    ("witwin/maxwell/fdtd/boundary/runtime.py", "resolve_absorber", "ValueError"),
    ("witwin/maxwell/fdtd/runtime/passivity.py", "check_passivity", "FDTDPassivityError"),
    ("witwin/maxwell/fdtd/runtime/passivity.py", "preflight_passivity", "ValueError"),
    # RFIC G1: graded vertical Green-table z admission fails closed at the last level.
    ("witwin/maxwell/mom/layered/vertical_refine.py", "admitted_vertical_table", "ConvergenceError"),
}


TRACKED_EXCEPTIONS = ("NotImplementedError", "UnsupportedFeatureError")


def tracked_exception(path, scope, name):
    return name in TRACKED_EXCEPTIONS or (path, scope, name) in SAFETY_GUARDS


def _without_scene_layers(node):
    class Strip(ast.NodeTransformer):
        def visit_Attribute(self, attribute):
            self.generic_visit(attribute)
            return attribute.value if attribute.attr == 'scene' else attribute
    return Strip().visit(node)


def p7b_normalized_condition(condition):
    """One enclosing condition with every ``.scene`` attribute layer removed (0.7.0 phase 7b indirection)."""
    def normalized(dump):
        return ast.dump(_without_scene_layers(eval(dump, vars(ast))), include_attributes=False)
    text, branch = condition
    if text.startswith('for '):
        body = text[len('for '):]
        for index in range(len(body)):
            if body.startswith(' in ', index):
                try:
                    target, iterator = eval(body[:index], vars(ast)), eval(body[index + 4:], vars(ast))
                except SyntaxError:
                    continue
                return ['for ' + normalized(ast.dump(target)) + ' in ' + normalized(ast.dump(iterator)), branch]
        raise ValueError(f'Unparsable loop condition {text!r}')
    if text.startswith('while '):
        return ['while ' + normalized(text[len('while '):]), branch]
    if text == 'try':
        return [text, branch]
    return [normalized(text), branch]


def p7b_guard_hashes(guard):
    """(body hash, normalized-condition hash) of a detailed guard for the 0.7.0 phase 7b receipt."""
    body = hashlib.sha256(json.dumps([guard['path'], guard['scope'], guard['expression'], guard['message']]).encode()).hexdigest()
    conditions = [p7b_normalized_condition(condition) for condition in guard['conditions']]
    return body, hashlib.sha256(json.dumps(conditions).encode()).hexdigest()


def migrated_identities():
    """Old -> new identity record of every raise the 0.7.0 migration rewrote."""
    if not RAISE_RECEIPT.exists():
        return {}
    return json.loads(RAISE_RECEIPT.read_text())['guard_id_map']


def literal_fragments(node):
    if isinstance(node,ast.Constant) and isinstance(node.value,str):
        return node.value
    if isinstance(node,ast.JoinedStr):
        return ''.join(value.value for value in node.values
                       if isinstance(value,ast.Constant) and isinstance(value.value,str))
    if isinstance(node,ast.BinOp):
        return literal_fragments(node.left)+literal_fragments(node.right)
    return ''


class _Guards(ast.NodeVisitor):
    def __init__(self,path):
        self.path=path
        self.scope=[]
        self.conditions=[]
        self.guards=[]

    def visit_ClassDef(self,node):
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_FunctionDef=visit_ClassDef
    visit_AsyncFunctionDef=visit_ClassDef

    def visit_If(self,node):
        condition=ast.dump(node.test,include_attributes=False)
        self.visit(node.test)
        for branch,statements in (('true',node.body),('false',node.orelse)):
            self.conditions.append((condition,branch))
            for statement in statements:
                self.visit(statement)
            self.conditions.pop()

    def visit_For(self,node):
        header='for '+ast.dump(node.target,include_attributes=False)+' in '+ast.dump(node.iter,include_attributes=False)
        for branch,statements in (('body',node.body),('else',node.orelse)):
            self.conditions.append((header,branch))
            for statement in statements:
                self.visit(statement)
            self.conditions.pop()

    visit_AsyncFor=visit_For

    def visit_While(self,node):
        header='while '+ast.dump(node.test,include_attributes=False)
        for branch,statements in (('body',node.body),('else',node.orelse)):
            self.conditions.append((header,branch))
            for statement in statements:
                self.visit(statement)
            self.conditions.pop()

    def visit_Try(self,node):
        branches=[('try',node.body),('else',node.orelse),('finally',node.finalbody)]
        branches.extend(('except '+ast.dump(handler.type,include_attributes=False) if handler.type is not None else 'except',handler.body)
                        for handler in node.handlers)
        for branch,statements in branches:
            self.conditions.append(('try',branch))
            for statement in statements:
                self.visit(statement)
            self.conditions.pop()

    def visit_Raise(self,node):
        exception=node.exc
        name=exception.func.id if isinstance(exception,ast.Call) and isinstance(exception.func,ast.Name) else getattr(exception,'id',None)
        if not tracked_exception(self.path, '.'.join(self.scope) or '<module>', name):
            return
        message=literal_fragments(exception.args[0]) if isinstance(exception,ast.Call) and exception.args else ''
        expression=ast.dump(exception,include_attributes=False)
        payload=(self.path,'.'.join(self.scope) or '<module>',expression,self.conditions)
        digest=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
        self.guards.append(dict(id=digest,path=self.path,scope=payload[1],message=message,
            expression=expression,conditions=[list(item) for item in self.conditions],line=node.lineno))


def collect_detailed_guards(package_root=None):
    package=ROOT/'witwin' if package_root is None else Path(package_root)
    result=[]
    for path in sorted(package.rglob('*.py')):
        visitor=_Guards(path.relative_to(package.parent).as_posix())
        visitor.visit(ast.parse(path.read_text(encoding='utf-8-sig')))
        occurrences=Counter()
        for guard in sorted(visitor.guards,key=lambda value:value['line']):
            digest=guard['id']
            guard['id']=f"{digest}:{occurrences[digest]}"
            occurrences[digest]+=1
            result.append(guard)
    return result


def is_contract(guard,contracts):
    return any(guard['path']==path and fragment in guard['message'] for path,fragment in contracts)


def load_contracts():
    tree=ast.parse((ROOT/'tests/api/public/test_guard_census.py').read_text())
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='CONTRACT_GUARDS' for target in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError('The explicit contract exclusion tuple is missing.')


def inherited_guards():
    if hashlib.sha256(BASELINE.read_bytes()).hexdigest()!=BASELINE_SHA256:
        raise ValueError('The original inherited-debt snapshot changed.')
    if hashlib.sha256(BASELINE_DETAILED.read_bytes()).hexdigest()!=BASELINE_DETAILED_SHA256:
        raise ValueError('The immutable baseline predicate/scope snapshot changed.')
    detailed=json.loads(BASELINE_DETAILED.read_text())
    contracts=load_contracts()
    if hashlib.sha256(json.dumps(contracts,separators=(',',':')).encode()).hexdigest()!=CONTRACT_SHA256:
        raise ValueError('The 26 contract exclusions changed; this reconciliation authorizes no reclassification.')
    # The baseline predates the SIE -> MoM rename; translate moved raise sites through the
    # line-paired identity map that scripts/dev/mom_rename.py recorded in its receipt.
    renamed=json.loads(RENAME_RECEIPT.read_text())['guard_id_map']
    migrated=migrated_identities()
    rekeyed={old:{'id':new['id'],'conditions':new['conditions']} for old,new in json.loads(P7B_RECEIPT.read_text())['guard_id_map'].items()}
    guards=[dict(guard,**renamed.get(guard['id'],{})) for guard in detailed['guards'] if not is_contract(guard,contracts)]
    guards=[dict(guard,**migrated.get(guard['id'],{})) for guard in guards]
    guards=[dict(guard,**rekeyed.get(guard['id'],{})) for guard in guards]
    strict={old:{'id':new['id'],'conditions':new['conditions']} for old,new in json.loads(STRICT_READS_2_RECEIPT.read_text())['guard_id_map'].items()}
    return [dict(guard,**strict.get(guard['id'],{})) for guard in guards]


def proposal(previous=None):
    baseline=json.loads(BASELINE.read_text())
    if baseline['commit']!=BASELINE_COMMIT or baseline['capabilities']!=300 or baseline['contracts']!=29:
        raise ValueError('The immutable inherited-debt baseline changed.')
    baseline_guards=inherited_guards()
    inherited={guard['id']:guard for guard in baseline_guards}
    previous_messages={(guard['path'],guard['message']) for guard in baseline_guards}
    contracts=load_contracts()
    prior={entry['id']:entry for entry in (previous or {}).get('active',[])}
    # A migrated raise keeps its review under the new identity.
    carried={record['id']:prior[old] for old,record in migrated_identities().items() if old in prior}
    active=[]
    for guard in collect_detailed_guards():
        if is_contract(guard,contracts):
            continue
        if guard['id'] in prior:
            entry=dict(prior[guard['id']],line=guard['line'])
        elif guard['id'] in carried:
            entry=dict(carried[guard['id']],id=guard['id'],expression=guard['expression'],line=guard['line'])
        else:
            key=(guard['path'],guard['message'])
            origin=('inherited_debt' if guard['id'] in inherited else
                    'modified_boundary' if key in previous_messages else 'feature_boundary')
            entry=dict(guard,origin=origin,review_status='pending',
                       unsupported_boundary=guard['message'],negative_tests=[],exit_acceptance='',rationale='')
        active.append(entry)
    current_ids={entry['id'] for entry in active}
    carried_from={carried[identity]['id'] for identity in current_ids if identity in carried}
    retired=list((previous or {}).get('retired',[]))
    for entry in (previous or {}).get('active',[]):
        if entry['id'] not in current_ids and entry['id'] not in carried_from:
            retired.append(dict(entry,retirement_review='pending',positive_tests=[],retirement_reason=''))
    retired_ids={entry['id'] for entry in retired}
    for entry in baseline_guards:
        if entry['id'] not in current_ids and entry['id'] not in retired_ids:
            retired.append(dict(entry,origin='inherited_debt',retirement_review='pending',
                                positive_tests=[],retirement_reason=''))
    return dict(schema='fdtd-capability-guard-manifest/1',snapshot_status='pending_final_source_freeze',
        baseline=dict(commit=BASELINE_COMMIT,path=BASELINE.relative_to(ROOT).as_posix(),
                      sha256=BASELINE_SHA256,detailed_path=BASELINE_DETAILED.relative_to(ROOT).as_posix(),
                      detailed_sha256=BASELINE_DETAILED_SHA256,capabilities=300,contracts=29),
        active=active,retired=retired)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal',type=Path,required=True)
    args=parser.parse_args()
    if args.proposal.resolve()==MANIFEST.resolve():
        parser.error('Write proposals to a separate file; approval requires an explicit review edit.')
    previous=json.loads(MANIFEST.read_text()) if MANIFEST.exists() else None
    record=proposal(previous)
    args.proposal.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(proposal=str(args.proposal),active=len(record['active']),
                         pending=sum(entry['review_status']=='pending' for entry in record['active']))))


if __name__=='__main__':
    main()
