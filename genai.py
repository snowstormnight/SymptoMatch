from PIL import Image
import torch
from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8

# This part meant to initialize the preprocess and the tokenizer
model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

# This part meant to determine what hardware should be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def detection(names, image_name):
    # This part meant to give the model possible combinations of diseases

    # the name.txt should contain possible combinations of diseases
    labels = []

    for name in names:
        labels.append(name)
    template = "This is a photo of a ."
    context_length = 256
    texts = tokenizer([template + l for l in labels], context_length=context_length).to(device)

    # This part meant to load the image and the text
    local_image_path = image_name  # Update this to your image path
    # Assuming `local_image_path` is the path to your single image
    image = Image.open(local_image_path).convert("RGB")  # Ensure the image is in RGB format
    preprocessed_image = preprocess(image).unsqueeze(0).to(device)  # Preprocess the image, add a batch dimension, and send to the device

    model.to(device)
    model.eval()

    # Start to use the model to prediect the type of image
    with torch.no_grad():
        image_features, text_features, logit_scale = model(preprocessed_image, texts)

        logits = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)
        sorted_indices = torch.argsort(logits, dim=-1, descending=True)

        sorted_indices = sorted_indices.cpu().numpy()

    pred = labels[sorted_indices[0][0]]

    # This part meant to print the result
    print("According to your description and the image sent, you may have " + pred + ".")

