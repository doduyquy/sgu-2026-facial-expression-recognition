import tensorflow as tf

def build_custom_resnet():
    # Load V1
    base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(48, 48, 3))
    
    def clone_fn(layer):
        if layer.name == 'conv1_conv':
            cfg = layer.get_config()
            cfg['strides'] = (1, 1)
            return tf.keras.layers.Conv2D.from_config(cfg)
        if layer.name == 'pool1_pool':
            # PyTorch uses nn.Identity(), we can use a 1x1 pool with stride 1
            return tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same', name=layer.name)
        return layer

    new_model = tf.keras.models.clone_model(base_model, clone_function=clone_fn)
    new_model.set_weights(base_model.get_weights())
    
    # Get layer 3 equivalent
    out = new_model.get_layer("conv4_block6_out").output
    model = tf.keras.Model(inputs=new_model.input, outputs=out)
    return model

m = build_custom_resnet()
dummy = tf.zeros((1, 48, 48, 3))
out = m(dummy)
print("Output shape:", out.shape)
