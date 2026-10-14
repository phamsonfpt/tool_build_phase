"""Tóm tắt s1_output.txt thành một bảng gọn (đơn vị: Coverage trên tập kiểm thử)."""
import re
txt = open("s1_output.txt", encoding="utf-8").read()
for blk in txt.split("\n== ")[1:]:
    lines = blk.strip().splitlines()
    v = {l.split()[0]: float(re.search(r"Cov=([\d.]+)", l).group(1)) for l in lines[1:]}
    lam = re.search(r"'lam': ([\d.]+)", [l for l in lines if l.strip().startswith("P2")][0]).group(1)
    print(f"{lines[0].split('|')[0].strip():32s} B1 {v['B1_conf']:.3f}  B2 {v['B2_ens_conf']:.3f}  "
          f"B4 {v['B4_maha']:.3f}  v1.2 {v['P0_var_v12']:.3f}  A1 {v['A1_ens_maha']:.3f}  "
          f"P2 {v['P2_mean_plus_sd']:.3f} (lam={lam})  Hs {v['H_single_hybrid']:.3f}  He {v['H_ens_hybrid']:.3f}  "
          f"| He-Hs {100*(v['H_ens_hybrid']-v['H_single_hybrid']):+.1f}pp  Hs-B1 {100*(v['H_single_hybrid']-v['B1_conf']):+.1f}pp")
