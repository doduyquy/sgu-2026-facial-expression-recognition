import tensorflow as tf
base = tf.keras.applications.ResNet50V2(include_top=False, weights=None, input_shape=(48,48,3))
for l in base.layers[:15]:
    s = ""
    if hasattr(l, 'strides'):
        s = f"strides={l.strides}"
    if hasattr(l, 'pool_size'):
        s += f" pool_size={l.pool_size}"
    print(f"{l.name} ({l.__class__.__name__}) {s}")
