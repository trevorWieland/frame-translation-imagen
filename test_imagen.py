import os
os.environ['TRANSFORMERS_CACHE'] = 'U:/HF_Cache/'
os.environ['HF_HOME'] = 'U:/HF_Cache/'

import torch
from torchvision.transforms import Compose, ColorJitter, ToTensor
from imagen_pytorch import Unet, Imagen, ImagenTrainer
from transformers import T5Tokenizer, T5EncoderModel
from transformers import pipeline
from tqdm.auto import tqdm
import datasets
from datasets import Image, load_from_disk

import numpy as np

BASE_MODEL = "google/t5-v1_1-base"

def load_data(use_full: bool = False):
    if use_full:
        data_dir = "B:/LargeDatasets/COCO/"
    else:
        data_dir = "./dummy_data/"

    ds = datasets.load_dataset("ydshieh/coco_dataset_script", "2017", data_dir=data_dir)

    return ds

def tokenize_data(base_model: str, train_dataset: datasets.Dataset):

    print("Building Text Tokenizing Models...")
    tokenizer = T5Tokenizer.from_pretrained(base_model)

    def tokenize_function(examples):
        inputs = examples["caption"]

        tokens = tokenizer(
            inputs,
            max_length=32,
            truncation=True,
            padding="max_length"
        )

        return tokens

    print("Tokenizing + Embedding Dataset in preparation for training...")
    tokenized_dataset = train_dataset.map(tokenize_function, batched=True, batch_size=1024, num_proc=8)

    return tokenized_dataset

def pixelize_image(tokenized_dataset: datasets.Dataset):

    def pixelize_function(examples):
        ims = np.array([
            np.moveaxis(np.asarray(image.convert("RGB").resize((256,256))), -1, 0)
            for image
            in examples["image_path"]
        ])
        examples["image"] = ims

        return examples

    print("Pixelizing image data in preparation for training...")
    pixelized_dataset = tokenized_dataset.map(pixelize_function, batched=True, batch_size=1024, num_proc=8)

    return pixelized_dataset

def embed_dataset(base_model: str, tokenized_dataset: datasets.Dataset, split: str):
    encoder = T5EncoderModel.from_pretrained(base_model).to("cuda")

    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    dataloader = torch.utils.data.DataLoader(tokenized_dataset[split], batch_size=512)

    embeddings = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            outputs = encoder(**batch)
            embedding = outputs.last_hidden_state
            embeddings.append(embedding.detach().cpu().numpy())

    dset_embed = datasets.Dataset.from_dict({"embeddings": np.concatenate(embeddings, axis=0)})

    return dset_embed

def load_or_create_data(data_path: str, force_new: bool, use_full: bool = False):

    if not force_new:
        try:
            ds = load_from_disk(data_path)
            build_new = False
        except Exception as e:
            print(e)
            print("Building from scratch due to error!")
            build_new = True

    if force_new or build_new:
        ds = load_data(use_full=use_full)

        ds["train"] = ds["train"].cast_column('image_path', Image(decode=True))
        ds["test"] = ds["test"].cast_column('image_path', Image(decode=True))
        ds["validation"] = ds["validation"].cast_column('image_path', Image(decode=True))

        ds = tokenize_data(BASE_MODEL, ds)

        train_embeddings = embed_dataset(BASE_MODEL, ds, "train")
        test_embeddings = embed_dataset(BASE_MODEL, ds, "test")
        val_embeddings = embed_dataset(BASE_MODEL, ds, "validation")

        ds["train"] = datasets.concatenate_datasets([ds["train"], train_embeddings], axis=1)
        ds["test"] = datasets.concatenate_datasets([ds["test"], test_embeddings], axis=1)
        ds["validation"] = datasets.concatenate_datasets([ds["validation"], val_embeddings], axis=1)

        ds = pixelize_image(ds)

        #ds.save_to_disk(data_path)

    return ds

def load_imagen_models():
    print("Building first UNET...")
    unet1 = Unet(
        dim = 32,
        cond_dim = 512,
        dim_mults = (1, 2, 4, 8),
        num_resnet_blocks = 3,
        layer_attns = (False, True, True, True),
        layer_cross_attns = (False, True, True, True)
    )

    print("Building second UNET...")
    unet2 = Unet(
        dim = 32,
        cond_dim = 512,
        dim_mults = (1, 2, 4, 8),
        num_resnet_blocks = (2, 4, 8, 8),
        layer_attns = (False, False, False, True),
        layer_cross_attns = (False, False, False, True)
    )

    # imagen, which contains the unets above (base unet and super resoluting ones)
    print("Building Imagen full model...")
    imagen = Imagen(
        unets = (unet1, unet2),
        image_sizes = (64, 256),
        timesteps = 1000,
        cond_drop_prob = 0.1
    )

    trainer = ImagenTrainer(imagen).cuda()

    return trainer

def collator(data):
    out_data = {}

    #Mask
    out_data["attention_mask"] = torch.tensor(np.array([np.array(d["attention_mask"]) for d in data])).type(torch.LongTensor)

    #Image
    out_data["image"] = torch.tensor(np.array([np.array(d["image"]) for d in data])).type(torch.FloatTensor)

    #Embedding
    out_data["embeddings"] = torch.tensor(np.array([np.array(d["embeddings"]) for d in data])).type(torch.FloatTensor)

    return out_data

def train_imagen(use_full: bool = False, force_reload: bool = True, batch_size: int = 8, epochs: int = 1):

    ds = load_or_create_data("U:/HF_DATA/COCO/", force_reload, use_full)
    trainer = load_imagen_models()

    ds["train"].set_format(columns=['embeddings', 'attention_mask', 'image'])
    dataloader = torch.utils.data.DataLoader(ds["train"], batch_size=batch_size, collate_fn=collator)

    pbar = tqdm(total=epochs*(len(dataloader)))

    for e in range(epochs):
        desc = f"EPOCH {e+1}/{epochs}"
        pbar.set_description(desc)
        for i, batch in enumerate(dataloader):

            for u in (1, 2):
                loss = trainer(
                    batch["image"].cuda(),
                    text_embeds=batch["embeddings"].cuda(),
                    text_masks=batch["attention_mask"].bool().cuda(),
                    unet_number=u
                )
                trainer.update(unet_number = u)

            pbar.update(1)
    pbar.close()

    trainer.save("models/short_model")

if __name__ == "__main__":
    train_imagen(force_reload=True, use_full=True, epochs=1)
