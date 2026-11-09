import hashlib
import json
import zipfile

import pytest

from benchmark.mom_archive import _provenance_records, verify_archive


def test_provenance_does_not_drop_a_conflicting_nested_segment():
    report = {'source_sha256': {'solver.py': 'first'},
              'segments': [[{'source_sha256': {'solver.py': 'different'}}]]}
    records = list(_provenance_records(report))
    assert records == [('$', {'solver.py': 'first'}),
                       ('$.segments[0][0]', {'solver.py': 'different'})]


def test_archive_rejects_changed_evidence_bytes(tmp_path):
    original = b'{"residual": 0.01}'
    manifest = {'schema_version': 1, 'files': {'evidence.json': {
        'bytes': len(original), 'sha256': hashlib.sha256(original).hexdigest()}}}
    path = tmp_path/'evidence.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('manifest.json', json.dumps(manifest))
        archive.writestr('evidence.json', original)
    assert verify_archive(path) == manifest
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('manifest.json', json.dumps(manifest))
        archive.writestr('evidence.json', b'{"residual": 0.00}')
    with pytest.raises(ValueError, match='integrity check failed'):
        verify_archive(path)


def test_archive_rejects_unmanifested_entries(tmp_path):
    path = tmp_path/'evidence.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('manifest.json', json.dumps({'schema_version': 1, 'files': {}}))
        archive.writestr('unrecorded.json', '{}')
    with pytest.raises(ValueError, match='do not match manifest'):
        verify_archive(path)
