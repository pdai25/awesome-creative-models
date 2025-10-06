from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from modelscope import dataset_snapshot_download, snapshot_download
import torch
from PIL import Image


pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
)

snapshot_download("DiffSynth-Studio/Qwen-Image-EliGen-V2", local_dir="models/DiffSynth-Studio/Qwen-Image-EliGen-V2", allow_file_pattern="model.safetensors")
pipe.load_lora(pipe.dit, "models/DiffSynth-Studio/Qwen-Image-EliGen-V2/model.safetensors")

global_prompt = "Poster for the Qwen-Image-EliGen Magic Café, featuring two magical coffees—one emitting flames and the other emitting ice spikes—against a light blue misty background. The poster includes text: 'Qwen-Image-EliGen Magic Café' and 'New Product Launch'"
entity_prompts = ["A red magic coffee with flames rising from the cup", 
                  "A red magic coffee surrounded by ice spikes", 
                  "Text: 'New Product Launch'", 
                  "Text: 'Qwen-Image-EliGen Magic Café'"]

dataset_snapshot_download(dataset_id="DiffSynth-Studio/examples_in_diffsynth", local_dir="./", allow_file_pattern=f"data/examples/eligen/qwen-image/example_6/*.png")
masks = [Image.open(f"./data/examples/eligen/qwen-image/example_6/{i}.png").convert('RGB').resize((1328, 1328)) for i in range(len(entity_prompts))]

image = pipe(
    prompt=global_prompt,
    seed=0,
    eligen_entity_prompts=entity_prompts,
    eligen_entity_masks=masks,
)
image.save("image.jpg")
