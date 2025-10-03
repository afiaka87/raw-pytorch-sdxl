"""
Data loaders for SDXL fine-tuning.

Supports:
- Simple image-caption pairs (folder structure)
- WebDataset format (tar shards)
- Custom datasets with __getitem__ interface
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from typing import Optional, Callable, Tuple
import torchvision.transforms as T
import io


class ImageCaptionDataset(Dataset):
    """
    Simple dataset for image-caption pairs.

    Directory structure:
        data/
            image1.jpg
            image1.txt
            image2.png
            image2.txt
            ...

    Each image file should have a corresponding .txt file with the caption.
    """

    def __init__(
        self,
        data_dir: str,
        image_size: int = 1024,
        center_crop: bool = True,
        random_flip: bool = True,
        image_extensions: tuple = (".jpg", ".jpeg", ".png", ".webp"),
    ):
        """
        Args:
            data_dir: Directory containing images and captions
            image_size: Target image size (square)
            center_crop: Whether to center crop images
            random_flip: Whether to randomly flip images horizontally
            image_extensions: Tuple of valid image extensions
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.center_crop = center_crop
        self.random_flip = random_flip

        # Find all image files
        self.image_files = []
        for ext in image_extensions:
            self.image_files.extend(self.data_dir.glob(f"*{ext}"))

        # Filter to only images with corresponding captions
        self.pairs = []
        for img_path in self.image_files:
            txt_path = img_path.with_suffix(".txt")
            if txt_path.exists():
                self.pairs.append((img_path, txt_path))

        if len(self.pairs) == 0:
            raise ValueError(f"No image-caption pairs found in {data_dir}")

        print(f"Found {len(self.pairs)} image-caption pairs in {data_dir}")

        # Setup transforms
        self.transform = self._build_transform()

    def _build_transform(self):
        """Build image transform pipeline."""
        transforms = []

        # Resize
        if self.center_crop:
            transforms.append(T.Resize(self.image_size, interpolation=T.InterpolationMode.BILINEAR))
            transforms.append(T.CenterCrop(self.image_size))
        else:
            transforms.append(T.Resize((self.image_size, self.image_size), interpolation=T.InterpolationMode.BILINEAR))

        # Random flip
        if self.random_flip:
            transforms.append(T.RandomHorizontalFlip(p=0.5))

        # Convert to tensor and normalize to [-1, 1]
        transforms.append(T.ToTensor())
        transforms.append(T.Normalize([0.5], [0.5]))

        return T.Compose(transforms)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, txt_path = self.pairs[idx]

        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return next item
            return self.__getitem__((idx + 1) % len(self))

        # Load caption
        try:
            with open(txt_path, "r") as f:
                caption = f.read().strip()
        except Exception as e:
            print(f"Error loading caption {txt_path}: {e}")
            caption = ""

        return {
            "image": image,
            "caption": caption,
        }


class BucketedImageCaptionDataset(Dataset):
    """
    Dataset with aspect ratio bucketing.

    Groups images by aspect ratio to minimize padding/cropping.
    Common buckets: 1024x1024, 1024x768, 768x1024, etc.
    """

    def __init__(
        self,
        data_dir: str,
        base_size: int = 1024,
        buckets: Optional[list[tuple[int, int]]] = None,
        random_flip: bool = True,
    ):
        """
        Args:
            data_dir: Directory containing images and captions
            base_size: Base resolution (1024 for SDXL)
            buckets: List of (width, height) buckets
            random_flip: Whether to randomly flip images
        """
        self.data_dir = Path(data_dir)
        self.base_size = base_size
        self.random_flip = random_flip

        # Default buckets if not provided
        if buckets is None:
            self.buckets = [
                (1024, 1024),  # Square
                (1024, 768),   # Landscape
                (768, 1024),   # Portrait
                (1024, 512),   # Wide
                (512, 1024),   # Tall
            ]
        else:
            self.buckets = buckets

        # Find image-caption pairs
        image_files = []
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            image_files.extend(self.data_dir.glob(f"*{ext}"))

        self.pairs = []
        self.bucket_indices = {bucket: [] for bucket in self.buckets}

        for i, img_path in enumerate(image_files):
            txt_path = img_path.with_suffix(".txt")
            if not txt_path.exists():
                continue

            # Determine bucket
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    bucket = self._find_bucket(w, h)
                    self.pairs.append((img_path, txt_path, bucket))
                    self.bucket_indices[bucket].append(len(self.pairs) - 1)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

        print(f"Found {len(self.pairs)} images in {len(self.buckets)} buckets:")
        for bucket, indices in self.bucket_indices.items():
            print(f"  {bucket}: {len(indices)} images")

    def _find_bucket(self, width: int, height: int) -> tuple[int, int]:
        """Find closest bucket for image dimensions."""
        aspect_ratio = width / height
        best_bucket = self.buckets[0]
        best_diff = float("inf")

        for bucket in self.buckets:
            bucket_ratio = bucket[0] / bucket[1]
            diff = abs(aspect_ratio - bucket_ratio)
            if diff < best_diff:
                best_diff = diff
                best_bucket = bucket

        return best_bucket

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, txt_path, bucket = self.pairs[idx]

        # Load and resize to bucket size
        try:
            image = Image.open(img_path).convert("RGB")
            image = image.resize(bucket, Image.Resampling.LANCZOS)

            # Random flip
            if self.random_flip and torch.rand(1).item() > 0.5:
                image = T.functional.hflip(image)

            # To tensor and normalize
            image = T.functional.to_tensor(image)
            image = T.functional.normalize(image, [0.5], [0.5])
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        # Load caption
        try:
            with open(txt_path, "r") as f:
                caption = f.read().strip()
        except Exception as e:
            print(f"Error loading caption {txt_path}: {e}")
            caption = ""

        return {
            "image": image,
            "caption": caption,
            "bucket": bucket,
        }


def create_webdataset_loader(
    data_dir: str,
    image_size: int = 1024,
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle: bool = True,
    center_crop: bool = True,
    random_flip: bool = True,
) -> DataLoader:
    """
    Create WebDataset loader for tar shards.

    Args:
        data_dir: Directory containing .tar shard files
        image_size: Target image size (square)
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle shards
        center_crop: Whether to center crop images
        random_flip: Whether to randomly flip images horizontally

    Returns:
        DataLoader instance
    """
    import webdataset as wds

    # Build list of shard URLs
    data_path = Path(data_dir)
    shard_files = sorted(data_path.glob("*.tar"))

    if len(shard_files) == 0:
        raise ValueError(f"No .tar files found in {data_dir}")

    # Convert to URLs (webdataset format)
    urls = [str(f) for f in shard_files]

    print(f"Found {len(urls)} webdataset shards in {data_dir}")

    def decode_sample(sample):
        """Decode webdataset sample to our format."""
        # WebDataset groups files by key (basename without extension)
        # After .decode(), images are PIL Images with keys like 'png', 'jpg', etc.
        # Text files remain as strings with keys like 'txt'

        # Handle different image formats (after decode, no dot prefix)
        image = None
        for ext in ['png', 'jpg', 'jpeg', 'webp']:
            if ext in sample:
                image = sample[ext]
                break

        if image is None:
            raise ValueError(f"No image found in sample with key {sample.get('__key__', 'unknown')}")

        caption = sample.get('txt', '')

        # Transform PIL image to tensor
        # Build transforms
        transforms = []
        if center_crop:
            transforms.append(T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR))
            transforms.append(T.CenterCrop(image_size))
        else:
            transforms.append(T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR))

        if random_flip:
            transforms.append(T.RandomHorizontalFlip(p=0.5))

        transforms.append(T.ToTensor())
        transforms.append(T.Normalize([0.5], [0.5]))

        transform = T.Compose(transforms)
        image_tensor = transform(image)

        return {
            "image": image_tensor,
            "caption": caption,
        }

    # Create webdataset pipeline
    # Use integer for shardshuffle (buffer size) or False to disable
    shard_shuffle = 100 if shuffle else False
    dataset = (
        wds.WebDataset(urls, shardshuffle=shard_shuffle)
        .decode("pil")  # Decode images as PIL, text as strings
        .map(decode_sample)
        .batched(batch_size, collation_fn=collate_fn)
    )

    # WebDataset returns an iterable, wrap in DataLoader
    loader = wds.WebLoader(
        dataset,
        batch_size=None,  # batching done in dataset
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader


def collate_fn(batch):
    """Custom collate function for batching."""
    images = torch.stack([item["image"] for item in batch])
    captions = [item["caption"] for item in batch]

    return {
        "images": images,
        "captions": captions,
    }


def collate_fn_bucketed(batch):
    """
    Collate function for bucketed dataset.

    All images in a batch should have the same bucket size.
    """
    # Check all images have same bucket
    buckets = [item["bucket"] for item in batch]
    if len(set(buckets)) > 1:
        raise ValueError("All images in batch must have same bucket size")

    images = torch.stack([item["image"] for item in batch])
    captions = [item["caption"] for item in batch]
    bucket = buckets[0]

    return {
        "images": images,
        "captions": captions,
        "bucket": bucket,
    }


def create_dataloader(
    data_dir: str,
    batch_size: int = 1,
    image_size: int = 1024,
    num_workers: int = 4,
    shuffle: bool = True,
    use_bucketing: bool = False,
    use_webdataset: bool = None,
    **dataset_kwargs,
) -> DataLoader:
    """
    Create data loader for training.

    Args:
        data_dir: Directory with image-caption pairs or webdataset tar shards
        batch_size: Batch size
        image_size: Image resolution
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        use_bucketing: Whether to use aspect ratio bucketing
        use_webdataset: Whether to use webdataset format (auto-detected if None)
        **dataset_kwargs: Additional arguments for dataset

    Returns:
        DataLoader instance
    """
    # Auto-detect webdataset format
    if use_webdataset is None:
        data_path = Path(data_dir)
        tar_files = list(data_path.glob("*.tar"))
        use_webdataset = len(tar_files) > 0

    # Use webdataset if tar files are present
    if use_webdataset:
        return create_webdataset_loader(
            data_dir=data_dir,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            **dataset_kwargs,
        )

    # Use regular dataset
    if use_bucketing:
        dataset = BucketedImageCaptionDataset(
            data_dir=data_dir,
            base_size=image_size,
            **dataset_kwargs,
        )
        collate = collate_fn_bucketed
    else:
        dataset = ImageCaptionDataset(
            data_dir=data_dir,
            image_size=image_size,
            **dataset_kwargs,
        )
        collate = collate_fn

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batches
    )

    return dataloader
