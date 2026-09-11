"""Lightweight per-request file checks; no models or hardware."""
import os
from tests.test_parity_gate import serving_module


def test_metadata_detects_same_size_write_even_if_mtime_restored(tmp_path):
    api = serving_module(); p = tmp_path/'weights';p.write_bytes(b'abcd')
    before = api.file_metadata([p]);stat = p.stat()
    p.write_bytes(b'efgh');os.utime(p,ns=(stat.st_atime_ns,stat.st_mtime_ns))
    assert api.file_metadata([p]) != before


def test_metadata_detects_file_replacement_and_symlink_retarget(tmp_path):
    api=serving_module();p=tmp_path/'weights';q=tmp_path/'replacement'
    p.write_bytes(b'abcd');q.write_bytes(b'abcd');link=tmp_path/'link';link.symlink_to(p)
    before=api.file_metadata([p,link]);q.replace(p)
    assert api.file_metadata([p,link]) != before
    q.write_bytes(b'abcd');before=api.file_metadata([link]);link.unlink();link.symlink_to(q)
    assert api.file_metadata([link]) != before


def test_metadata_does_not_read_file_contents(tmp_path,monkeypatch):
    api=serving_module();p=tmp_path/'weights';p.write_bytes(b'abcd')
    def forbidden(*args,**kwargs):raise AssertionError('read of weight contents')
    monkeypatch.setattr(type(p),'open',forbidden)
    assert api.file_metadata([p]) == api.file_metadata([p])
