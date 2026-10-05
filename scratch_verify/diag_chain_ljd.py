# fork-only diagnostic: per-point gap between numerical det(jacobian) baseline and analytical log_jac_det
import numpy as np, pytensor
import tests.distributions.test_transform as T
from pytensor.compile.mode import get_default_mode

chain = T.tr.Chain([T.tr.logodds, T.tr.ordered])
domain = T.Vector(T.R, 4)
y = T.pt.vector("y")
x = chain.backward(y)
jac = T.pt.log(T.pt.abs(T.pt.linalg.det(T.jacobian(x, [y]))))
actual = T.function([y], jac)
computed = T.function([y], T.pt.as_tensor_variable(chain.log_jac_det(y)), on_unused_input="ignore")
rows = []
for yval in domain.vals:
    a = float(actual(yval)); c = float(computed(yval))
    rows.append((abs(a - c), a, c, [float(v) for v in yval]))
print("DIAG pytensor", pytensor.__version__, "floatX", pytensor.config.floatX, "config.linker", pytensor.config.linker,
      "mode-linker", type(get_default_mode().linker).__name__, "points", len(rows))
nonfinite = [r for r in rows if not np.isfinite(r[0])]
print("DIAG nonfinite", len(nonfinite), nonfinite[:2])
fin = [r for r in rows if np.isfinite(r[0])]
for tol in (1e-5, 1e-4, 1e-3, 1e-2):
    bad = [r for r in fin if r[0] > tol + tol * abs(r[2])]
    print(f"DIAG tol={tol:g} failing_points={len(bad)}/{len(rows)}")
for r in sorted(fin, key=lambda r: -r[0])[:4]:
    print("DIAG worst absdiff=%.6g actual(numerical)=%.8g computed(analytical)=%.8g y=%s" % r)
