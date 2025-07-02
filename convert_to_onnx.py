
import torch
import torch.onnx
import onnx
import onnxruntime
import numpy as np
import logging
from src.model import get_orientation_model
from src.utils import get_device
from config import IMAGE_SIZE, MODEL_NAME

def convert_to_onnx(model_path, onnx_file_name):
    # Instantiate and load the model
    device = get_device()
    model = get_orientation_model(pretrained=False)
    
    # Adjust state_dict keys if the model was compiled
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Create a dummy input tensor with the correct shape and type
    batch_size = 1 
    dummy_input = torch.randn(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE, requires_grad=True).to(device)

    # Export the model
    print("Exporting model to ONNX format...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_name,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input' : {0 : 'batch_size'},
                      'output' : {0 : 'batch_size'}}
    )
    print(f"Model successfully exported to {onnx_file_name}")

    # --- VERIFICATION PROCESS ---
    print("\nVerifying the ONNX model...")

    # Check that the ONNX model is well-formed
    onnx_model = onnx.load(onnx_file_name)
    onnx.checker.check_model(onnx_model)
    print("ONNX model check passed.")

    # Create an ONNX Runtime inference session
    ort_session = onnxruntime.InferenceSession(onnx_file_name)

    # Get the output from the PyTorch model
    with torch.no_grad():
        pytorch_out = model(dummy_input)

    # Get the output from the ONNX Runtime
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.detach().cpu().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)

    # Compare the outputs
    np.testing.assert_allclose(pytorch_out.cpu().numpy(), ort_outs[0], rtol=4e-02, atol=1e-05)

    print("Verification successful: PyTorch and ONNX Runtime outputs match.")

if __name__ == '__main__':
    convert_to_onnx(f"models/{MODEL_NAME}.pth", f"models/{MODEL_NAME}.onnx")
