"""Copy of an exported openWakeWord classifier with a symbolic batch dimension (for batch scoring).

Usage: python -I dyn_batch.py IN.onnx OUT.onnx
"""

import sys

import onnx

m = onnx.load(sys.argv[1])
for v in list(m.graph.input) + list(m.graph.output):
    v.type.tensor_type.shape.dim[0].dim_param = "N"
for n in m.graph.node:  # Reshape nodes with a hard batch of 1 would break batching
    if n.op_type == "Reshape":
        print("reshape node present:", n.name)
onnx.checker.check_model(m)
onnx.save(m, sys.argv[2])
print(
    [
        (i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim])
        for i in m.graph.input
    ]
)
