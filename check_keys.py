"""Torch-free structural check of MSD_musicnn checkpoint keys/shapes.
Usage: python check_keys.py MSD_musicnn_optimized_fp16.pt"""

# ---------------- torch-free structural check ----------------
def _expected_shapes():
    exp = {}
    def bn(p, n):
        for k in ("weight", "bias", "running_mean", "running_var"):
            exp[f"{p}.{k}"] = (n,)
        exp[f"{p}.num_batches_tracked"] = ()
    bn("bn_input", 1)
    for name, k, c in [("timbral_1", (7, 38), 204), ("timbral_2", (7, 67), 204),
                       ("temp_1", (128, 1), 51), ("temp_2", (64, 1), 51), ("temp_3", (32, 1), 51)]:
        exp[f"{name}.conv.conv.weight"] = (c, 1) + k
        exp[f"{name}.conv.conv.bias"] = (c,)
        bn(f"{name}.conv.bn", c)
    for i, w in [(1, 561), (2, 64), (3, 64)]:
        exp[f"midend.c{i}_conv.weight"] = (64, 1, 7, w)
        exp[f"midend.c{i}_conv.bias"] = (64,)
        bn(f"midend.c{i}_bn", 64)
    bn("backend.bn_in", 1506)
    exp["backend.fc1.weight"] = (200, 1506); exp["backend.fc1.bias"] = (200,)
    bn("backend.bn_fc1", 200)
    exp["backend.fc2.weight"] = (50, 200); exp["backend.fc2.bias"] = (50,)
    return exp


def check_keys(path):
    """Compare checkpoint keys/shapes with this module's design, without torch."""
    import zipfile, pickle, collections
    z = zipfile.ZipFile(path)
    pkl = [n for n in z.namelist() if n.endswith("data.pkl")][0]

    def rebuild(storage, off, size, stride, *r):
        return ("T", tuple(size))

    class U(pickle.Unpickler):
        def find_class(self, mod, n):
            if n == "_rebuild_tensor_v2": return rebuild
            if n == "OrderedDict": return collections.OrderedDict
            if "Storage" in n: return n
            return lambda *a, **k: (mod, n, a)
        def persistent_load(self, pid): return str(pid[1])

    got = {k: v[1] for k, v in U(z.open(pkl)).load().items()}
    exp = _expected_shapes()
    bad = [(k, exp.get(k), got.get(k)) for k in sorted(set(exp) | set(got)) if exp.get(k) != got.get(k)]
    print("keys expected/found:", len(exp), len(got), "| mismatches:", len(bad))
    for b in bad: print("  ", b)
    return not bad



if __name__ == "__main__":
    import sys
    check_keys(sys.argv[1])
