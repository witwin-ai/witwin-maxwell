"""Freeze reviewed module-local inputs without changing any historical witness."""
from pathlib import Path
import argparse,hashlib,json,sys
CONTROL=Path(__file__).resolve().parent
sys.path.insert(0,str(CONTROL))
from module_environment import SCHEMA,load_policy
from shard_common import digest


def production_hash(root):
    value=hashlib.sha256()
    for path in sorted((root/'witwin/maxwell').rglob('*.py')):
        value.update(str(path.relative_to(root)).encode());value.update(path.read_bytes())
    return value.hexdigest()


def build(catalog_path,source,output,*,reconciliation=None,radial_bridge=None):
    catalog_path=Path(catalog_path).resolve();source=Path(source).resolve();output=Path(output).resolve()
    if output.exists():raise ValueError('Use a new policy path; original manifests are never overwritten')
    catalog=json.loads(catalog_path.read_text());files={};modules={};pending=[]
    def pin(path,expected=None):
        path=Path(path).resolve()
        if not path.is_file():raise ValueError(f'Missing original module input: {path}')
        actual=digest(path)
        if expected is not None and actual!=expected:raise ValueError(f'Original module input SHA mismatch: {path}')
        if str(path) in files and files[str(path)]!=actual:raise ValueError('Conflicting pins for one original file')
        files[str(path)]=actual
        return str(path)
    pin(catalog_path)
    for entry in catalog['entries']:
        if entry['status']!='ready_module_scoped_recorded_bundle':
            pending.append(dict(module=entry['module'],catalog_status=entry['status']));continue
        modules[entry['module']]=dict(environment=dict(entry['environment']),original_input_record=entry['selected_original_input_record'],qualification=False)
        pin(entry['selected_original_input_record'])
        for item in entry['input_files']:pin(item['path'],item['actual_sha256'])
    special='tests/mom/test_scalar_near_channels.py'
    if special in modules:modules[special]['sha_pairs']={'GSP_ORIGINAL_NEARFIELD_SHA256':'GSP_ORIGINAL_NEARFIELD_SOURCE'}
    if reconciliation:
        reconciliation=Path(reconciliation).resolve();review=json.loads(reconciliation.read_text());pin(reconciliation)
        for entry in review['additional_existing_originals']:
            module=entry['module']
            if module in modules:raise ValueError('Existing-original proposal overlaps a selected module')
            modules[module]=dict(environment=dict(entry['environment']),qualification=False,source_kind='existing_original_read_only_reconciliation')
            for field in ('manifest','historical_input','retained_copy','producer_xml'):
                if field in entry:pin(entry[field]['path'],entry[field]['sha256'])
            for pair in entry.get('actual_pairs',[]):
                for field in ('diagnostic','matrix'):pin(pair[field]['path'],pair[field]['sha256'])
        pending=[p for p in pending if p['module'] not in modules]
    current=production_hash(source);bridge_ready=False
    if radial_bridge:
        radial_bridge=Path(radial_bridge).resolve();bridge=json.loads(radial_bridge.read_text())
        if bridge.get('version')!='scalar_fmm_radial_order_bridge_v1' or bridge.get('production_source_hash')!=current or bridge.get('production_qualification') is not False:
            raise ValueError('Supply a new current-source consumer bridge; historical headers are never rewritten')
        modules['tests/mom/test_fmm_radial_order_bridge.py']['environment']['GSP_RADIAL_ORDER_BRIDGE']=pin(radial_bridge)
        for name in ('original_manifest','original_input','original_receipt','original_terminal'):
            pin(bridge[name]['path'],bridge[name]['sha256'])
        for item in bridge['records'].values():pin(item['path'],item['sha256'])
        original=json.loads(Path(bridge['original_input']['path']).read_text())
        for path,sha in original['source_before'].items():pin(path,sha)
        bridge_ready=True
    result=dict(schema=SCHEMA,modules=modules,files=files,source_production_hash=current,ready_for_final_execution=bridge_ready,pending_optional_reuse=pending,catalog_source=str(catalog_path),qualification=False,scope='Artifact availability and exact module-local input policy only. All historical source hashes/receipts remain unchanged; actual regression and physics qualification require separate execution.')
    output.write_text(json.dumps(result,indent=2)+'\n')
    try:load_policy(output,source)
    except BaseException:
        output.unlink();raise
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--catalog',type=Path,required=True);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--reconciliation',type=Path);p.add_argument('--radial-bridge',type=Path);a=p.parse_args();r=build(a.catalog,a.source_root,a.output,reconciliation=a.reconciliation,radial_bridge=a.radial_bridge);print(json.dumps(dict(modules=len(r['modules']),pinned_inputs=len(r['files']),ready_for_final_execution=r['ready_for_final_execution'],pending_optional_reuse=r['pending_optional_reuse'])))
