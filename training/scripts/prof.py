import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlx.core as mx
import chunked_delta as cd
B,T,H,D = 1, 2048, 16, 128
q = mx.random.normal((B,T,H,D))*0.1; k = mx.random.normal((B,T,H,D)); k = k/mx.linalg.norm(k,axis=-1,keepdims=True)
v = mx.random.normal((B,T,H,D)); g = -mx.abs(mx.random.normal((B,T,H)))*0.1; b = mx.sigmoid(mx.random.normal((B,T,H)))
def f(q,k,v): return cd.chunk_gated_delta(q,k,v,g,b,None)[0].sum()
gf = mx.grad(f, argnums=(0,1,2))
for chunk in [128, 64, 128, 64]:
    cd.CHUNK = chunk
    def f(q,k,v, c=chunk): return cd.chunk_gated_delta(q,k,v,g,b,None,chunk=c)[0].sum()
    gf = mx.grad(f, argnums=(0,1,2))
    for blk in [8]:
        cd.INV_BLOCK = blk
        mx.eval(gf(q,k,v)); t=time.time()
        for _ in range(5): mx.eval(gf(q,k,v))
        print("chunk", chunk, "blk", blk, f"fwd+bwd {(time.time()-t)/5*1000:.0f} ms")
        t=time.time()
        for _ in range(5): mx.eval(f(q,k,v))
        print("   fwd", f"{(time.time()-t)/5*1000:.0f} ms")
