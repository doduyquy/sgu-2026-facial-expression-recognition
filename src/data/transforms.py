import tensorflow as tf

def build_transform(config, split="train"):
    """Build transforms for train / val / test.

    Fix 1: config key is 'input_size' (not 'image_size'). Falls back to
            'image_size' for backward compatibility with older configs.
    Fix 6: When semantic masks are used, TenCrop is incompatible because
            bounding-box coordinates are defined in the original image space
            and become invalid after any spatial crop. Simple resize is used
            instead so that bbox coordinates stay correct.

    Args:
        config: full config dict
        split: 'train' | 'val' | 'test'

    Returns:
        tf.data compatible transform function
    """
    data_cfg = config.get('data', {})
    image_size = data_cfg.get('input_size', data_cfg.get('image_size', 48))
    mu = 0.5
    st = 0.5

    use_semantic_masks = bool(data_cfg.get('use_semantic_masks', False))

    def transform_fn(image):
        # Convert to float and scale [0, 1]
        image = tf.cast(image, tf.float32) / 255.0

        if split == "train":
            if use_semantic_masks:
                # Non-spatial augmentations to preserve bbox alignment.
                # Horizontal flip and Affine are handled synchronously in the dataset generator.
                image = tf.image.random_brightness(image, 0.2)
                image = tf.image.random_contrast(image, 0.8, 1.2)
                image = tf.image.random_saturation(image, 0.8, 1.2)
            else:
                # Standard path without bounding boxes
                image = tf.image.resize(image, [int(image_size * 1.2), int(image_size * 1.2)])
                image = tf.image.random_crop(image, [image_size, image_size, tf.shape(image)[-1]])
                image = tf.image.random_flip_left_right(image)
        else:
            if use_semantic_masks:
                # No TenCrop — semantic bbox coords are in original image space
                # and would be wrong after any spatial crop. Simple resize preserves coords.
                image = tf.image.resize(image, [image_size, image_size])
            else:
                # TenCrop TTA for models that do not use bounding boxes.
                larger = int(image_size * 56 / 48)
                image = tf.image.resize(image, [larger, larger])
                h, w = larger, larger
                th, tw = image_size, image_size

                tl = tf.image.crop_to_bounding_box(image, 0, 0, th, tw)
                tr = tf.image.crop_to_bounding_box(image, 0, w - tw, th, tw)
                bl = tf.image.crop_to_bounding_box(image, h - th, 0, th, tw)
                br = tf.image.crop_to_bounding_box(image, h - th, w - tw, th, tw)
                center = tf.image.crop_to_bounding_box(image, (h - th) // 2, (w - tw) // 2, th, tw)

                crops = [tl, tr, bl, br, center]
                flips = [tf.image.flip_left_right(c) for c in crops]
                image = tf.stack(crops + flips, axis=0) # [10, th, tw, C]

        # Normalize
        image = (image - mu) / st
        return image

    return transform_fn