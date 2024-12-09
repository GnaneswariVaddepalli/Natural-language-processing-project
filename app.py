import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import (
    ImageCaptioningModelWithAttention,
    FlickrDataset,
    EncoderCNNWithAttention,
    DecoderRNNWithAttention
)


def load_model(model_path):
    """
    Load the pre-trained image captioning model
    """

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    dataset = FlickrDataset(
        root_dir='flickr8k/flickr8k/Flickr8k_Subset/Images',
        captions_file='flickr8k/flickr8k/Flickr8k_Subset/captions.txt',
        transform=transform
    )

    embed_size = 256
    hidden_size = 512
    attention_dim = 256
    encoder_dim = 256
    num_layers = 1

    model = ImageCaptioningModelWithAttention(
        embed_size=embed_size,
        hidden_size=hidden_size,
        vocab_size=len(dataset.vocab),
        attention_dim=attention_dim,
        encoder_dim=encoder_dim,
        num_layers=num_layers
    )

    model.load_model(model_path)

    return model, dataset, transform


def main():
    st.title('Image Captioning 🖼️✍️')

    @st.cache_resource
    def get_model():
        try:
            model, dataset, transform = load_model('model/image_captioning_model.pth')
            return model, dataset, transform
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None, None, None

    model, dataset, transform = get_model()

    if model is None:
        st.error("Could not load the model. Please check your model path and dependencies.")
        return

    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        help="Upload an image to generate a caption"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button('Generate Caption'):
            with st.spinner('Generating caption...'):
                try:
                    temp_image_path = "temp_uploaded_image.jpg"
                    image.save(temp_image_path)

                    caption = model.generate_caption(
                        temp_image_path,
                        transform,
                        dataset.vocab
                    )

                    st.success(f"Caption: {caption}")

                    os.remove(temp_image_path)

                except Exception as e:
                    st.error(f"Error generating caption: {e}")


if __name__ == '__main__':
    main()