"""Restore verified Linux initial tensor bits on Windows; never relax a hash gate.

Old Torch stores data records lexicographically and omits two newer metadata
records. Its serialization ID contains a platform-dependent record-name hash.
Only this container metadata is translated for the historical SHA comparison.
"""
import copy,hashlib,io,struct,zipfile,zlib
import torch

def historical_hash(state,reference):
    ref=zipfile.ZipFile(reference);root=ref.namelist()[0].split('/')[0]
    prefix=ref.read(root+'/.data/serialization_id')[:20]
    buf=io.BytesIO();torch.save(copy.deepcopy(state),buf);z=zipfile.ZipFile(buf)
    assert z.read('archive/data.pkl')==ref.read(root+'/data.pkl'),'Historical tensor schema/pickle mismatch'
    out=io.BytesIO()
    with torch.serialization._open_zipfile_writer(out) as writer:
        for name in ref.namelist():
            key=name.split('/',1)[1]
            if key in ['version','.data/serialization_id']:continue
            value=z.read('archive/'+key);writer.write_record(key,value,len(value))
    raw=out.getvalue();zz=zipfile.ZipFile(out);old=zz.read('archive/.data/serialization_id');new=prefix+old[20:]
    oldcrc=struct.pack('<I',zlib.crc32(old));newcrc=struct.pack('<I',zlib.crc32(new))
    assert raw.count(old)==1 and raw.count(oldcrc)==2
    raw=raw.replace(old,new).replace(oldcrc,newcrc)
    # Independently load the translated serialization and compare every tensor.
    loaded=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    assert all(torch.equal(t.detach().cpu(),loaded[k]) for k,t in state.items())
    return hashlib.sha256(raw).hexdigest()

def construct_exact(constructor,seed_fn,reference,expected,device):
    original=torch.Tensor.uniform_
    def uniform_fma(t,a=0.,b=1.,*,generator=None):
        assert t.dtype==torch.float32 and t.device.type=='cpu'
        with torch.no_grad():
            u=torch.empty_like(t);original(u,0.,1.,generator=generator)
            lo=float(torch.tensor(a,dtype=t.dtype));hi=float(torch.tensor(b,dtype=t.dtype))
            return t.copy_((u.double()*(hi-lo)+lo).float())
    seed_fn(0)
    try:
        torch.Tensor.uniform_=uniform_fma
        model=constructor('LiteBN',62).to(device)
    finally:torch.Tensor.uniform_=original
    actual=historical_hash(model.state_dict(),reference)
    assert actual==expected,f'INITIALIZATION_MISMATCH: {actual} != {expected}'
    seed_fn(100000)
    return model,actual
