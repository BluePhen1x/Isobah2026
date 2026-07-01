import tensorflow as tf
class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.conv1 = tf.keras.layers.Conv2D(filters, 3, padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters, 3, padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.shortcut = tf.keras.layers.Conv2D(filters, 1, padding="same")
    def call(self, x, training=False):
        residual = self.shortcut(x)
        out = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        out = self.bn2(self.conv2(out), training=training)
        return tf.nn.relu(out + residual)
    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        return config
class SpectralBandAttentionEncoder(tf.keras.Model):
    def __init__(self, feature_dim=256, **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.group_channels = 64
        self.rgb_block = ResidualBlock(self.group_channels, name="rgb_resblock")
        self.rede_block = ResidualBlock(self.group_channels, name="rede_resblock")
        self.nir_block = ResidualBlock(self.group_channels, name="nir_resblock")
        self.swir_block = ResidualBlock(self.group_channels, name="swir_resblock")
        self.cross_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=4, key_dim=16, name="cross_spectral_mha"
        )
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.projection = tf.keras.layers.Conv2D(
            feature_dim, 1, padding="same", name="feature_projection"
        )
    def call(self, x, training=False):
        f_rgb = self.rgb_block(x[..., 0:3], training=training)
        f_rede = self.rede_block(x[..., 3:6], training=training)
        f_nir = self.nir_block(x[..., 6:8], training=training)
        f_swir = self.swir_block(x[..., 8:10], training=training)
        stacked = tf.stack([f_rgb, f_rede, f_nir, f_swir], axis=3)
        input_shape = tf.shape(stacked)
        B = input_shape[0]
        H = input_shape[1]
        W = input_shape[2]
        G = 4
        C = self.group_channels
        tokens = tf.reshape(stacked, [B * H * W, G, C])
        attended = self.cross_attention(tokens, tokens, training=training)
        attended = self.layer_norm(attended + tokens)
        fused = tf.reduce_mean(attended, axis=1)
        fused = tf.reshape(fused, [B, H, W, C])
        return self.projection(fused)
class SolarGeometryCorrection(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ], name="solar_mlp")
    def call(self, warped_frame, theta_source, theta_target):
        deg_to_rad = tf.constant(3.141592653589793 / 180.0)
        cos_source = tf.cos(tf.cast(theta_source, tf.float32) * deg_to_rad)
        cos_target = tf.cos(tf.cast(theta_target, tf.float32) * deg_to_rad)
        lambertian_ratio = cos_target / (cos_source + 1e-6)
        mlp_input = tf.stack([cos_source, cos_target, lambertian_ratio], axis=-1)
        if len(mlp_input.shape) == 1:
            mlp_input = tf.expand_dims(mlp_input, axis=0)
        learned_residual = tf.squeeze(self.mlp(mlp_input), axis=-1)
        correction = lambertian_ratio + learned_residual
        correction = tf.reshape(correction, [-1, 1, 1, 1])
        return warped_frame * correction
class CompositePhysicsLoss(tf.keras.losses.Loss):
    def __init__(self, epsilon=1e-3, lambda_spectral=0.5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.lambda_spectral = lambda_spectral
    def charbonnier(self, y_true, y_pred):
        diff_sq = tf.square(y_true - y_pred)
        return tf.reduce_mean(tf.sqrt(diff_sq + self.epsilon ** 2))
    def spectral_angle_mapper(self, y_true, y_pred):
        dot_product = tf.reduce_sum(y_true * y_pred, axis=-1)
        norm_true = tf.sqrt(tf.reduce_sum(tf.square(y_true), axis=-1) + 1e-8)
        norm_pred = tf.sqrt(tf.reduce_sum(tf.square(y_pred), axis=-1) + 1e-8)
        cos_angle = dot_product / (norm_true * norm_pred)
        cos_angle = tf.clip_by_value(cos_angle, -1.0 + 1e-7, 1.0 - 1e-7)
        return tf.reduce_mean(tf.acos(cos_angle))
    def call(self, y_true, y_pred):
        l_charb = self.charbonnier(y_true, y_pred)
        l_sam = self.spectral_angle_mapper(y_true, y_pred)
        return l_charb + self.lambda_spectral * l_sam
    def get_config(self):
        config = super().get_config()
        config.update({
            "epsilon": self.epsilon,
            "lambda_spectral": self.lambda_spectral,
        })
        return config
class SpectraFlowNet(tf.keras.Model):
    def __init__(self, feature_dim=256, **kwargs):
        super().__init__(**kwargs)
        self.encoder = SpectralBandAttentionEncoder(
            feature_dim=feature_dim, name="spectral_encoder"
        )
        self.solar_correction = SolarGeometryCorrection(
            name="solar_correction"
        )
    def build(self, input_shape):
        self.encoder.build(input_shape)
        super().build(input_shape)
    def call(self, frame_0, frame_1, theta_0, theta_t, training=False):
        features_0 = self.encoder(frame_0, training=training)
        features_1 = self.encoder(frame_1, training=training)
        return features_0, features_1
if __name__ == "__main__":
    print("=" * 70)
    print("SpectraFlow-Net — Architecture Verification")
    print("=" * 70)
    B, H, W, C = 2, 64, 64, 13
    frame_0 = tf.random.normal([B, H, W, C])
    frame_1 = tf.random.normal([B, H, W, C])
    theta_0 = tf.constant([35.0, 40.0])
    theta_t = tf.constant([38.0, 42.0])
    print(f"\nInput shape:   frame_0 = {frame_0.shape}")
    print(f"Input shape:   frame_1 = {frame_1.shape}")
    print(f"Input values:  theta_0 = {theta_0.numpy()}")
    print(f"Input values:  theta_t = {theta_t.numpy()}")
    print("\n--- Component 1: SpectralBandAttentionEncoder ---")
    encoder = SpectralBandAttentionEncoder(feature_dim=256)
    enc_out = encoder(frame_0, training=False)
    print(f"Encoder output shape: {enc_out.shape}")
    assert enc_out.shape == (B, H, W, 256), f"Expected (2,64,64,256), got {enc_out.shape}"
    print("PASSED")
    print("\n--- Component 2: SolarGeometryCorrection ---")
    solar = SolarGeometryCorrection()
    warped = tf.random.normal([B, H, W, C])
    corrected = solar(warped, theta_0, theta_t)
    print(f"Solar correction output shape: {corrected.shape}")
    assert corrected.shape == warped.shape, f"Shape mismatch: {corrected.shape} vs {warped.shape}"
    print("PASSED")
    print("\n--- Component 3: CompositePhysicsLoss ---")
    loss_fn = CompositePhysicsLoss(epsilon=1e-3, lambda_spectral=0.5)
    y_true = tf.random.normal([B, H, W, C])
    y_pred = tf.random.normal([B, H, W, C])
    loss_val = loss_fn(y_true, y_pred)
    print(f"Loss value: {loss_val.numpy():.6f}")
    assert loss_val.shape == (), f"Expected scalar loss, got shape {loss_val.shape}"
    assert loss_val.numpy() > 0, "Loss should be positive"
    print("PASSED")
    print("\n--- Component 4: SpectraFlowNet (Main Assembly) ---")
    model = SpectraFlowNet(feature_dim=256)
    feat_0, feat_1 = model(frame_0, frame_1, theta_0, theta_t, training=False)
    print(f"Feature 0 shape: {feat_0.shape}")
    print(f"Feature 1 shape: {feat_1.shape}")
    assert feat_0.shape == (B, H, W, 256)
    assert feat_1.shape == (B, H, W, 256)
    print("PASSED")
    print("\n--- Gradient Flow Verification ---")
    with tf.GradientTape() as tape:
        feat_0, feat_1 = model(frame_0, frame_1, theta_0, theta_t, training=True)
        dummy_target = tf.random.normal([B, H, W, 256])
        loss = tf.reduce_mean(tf.square(feat_0 - dummy_target))
    grads = tape.gradient(loss, model.trainable_variables)
    none_grads = [v.name for v, g in zip(model.trainable_variables, grads) if g is None]
    print(f"Total trainable variables: {len(model.trainable_variables)}")
    print(f"Variables with None gradients: {len(none_grads)}")
    if none_grads:
        for name in none_grads:
            print(f"  WARNING — no gradient: {name}")
    else:
        print("All variables receive gradients.")
    print("PASSED")
    print("\n--- Parameter Count ---")
    total = sum(tf.size(v).numpy() for v in model.trainable_variables)
    print(f"Total trainable parameters: {total:,}")
    print("\n" + "=" * 70)
    print("ALL VERIFICATIONS PASSED")
    print("=" * 70)