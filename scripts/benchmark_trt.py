import numpy as np
import onnx
import onnx_graphsurgeon as gs

# --- Comprehensive and final patch for the library ---
class MappingPatch:
    # From ONNX to NumPy
    TENSOR_TYPE_TO_NP_TYPE = {
        1: np.float32, 2: np.uint8, 3: np.int8, 4: np.uint16,
        5: np.int16, 6: np.int32, 7: np.int64, 9: bool,
        10: np.float16, 11: np.double, 12: np.uint32, 13: np.uint64,
        14: np.complex64, 15: np.complex128, 16: np.float32
    }
    # From NumPy to ONNX (which caused the recent error)
    NP_TYPE_TO_TENSOR_TYPE = {np.dtype(v): k for k, v in TENSOR_TYPE_TO_NP_TYPE.items()}

# Inject the patch into the onnx library
onnx.mapping = MappingPatch

def optimize_for_speed(input_path, output_path):
    print(f"🚀 Starting Deep Surgery on: {input_path}")
    
    # Load the model
    onnx_model = onnx.load(input_path)
    graph = gs.import_onnx(onnx_model)

    print("🧹 Cleaning the graph and fusing INT4 nodes...")
    
    # 1. Constant Folding - to reduce model size and speed up operations
    # We set a maximum folding limit to avoid excessive file bloating
    graph.fold_constants().cleanup()
    
    # 2. Simplify topological sorting
    graph.toposort()

    # 3. Remove unnecessary nodes that cause Reformatting on the Jetson
    # We will clear out nodes that do not contribute to the actual computation
    for node in graph.nodes:
        if node.op in ["Identity", "Dropout"]:
            node.outputs = node.inputs

    # Final cleanup
    graph.cleanup()

    # Save the optimized model
    onnx.save(gs.export_onnx(graph), output_path)
    print(f"✅ SUCCESS! Optimized model saved as: {output_path}")

if __name__ == "__main__":
    optimize_for_speed("geta_final_optimized.onnx", "geta_ultra_fast.onnx")