import tensorflow as tf
import tensorflow.keras as keras
import numpy as np


def mlp(inputs, in_features, hidden_features=None, out_features=None, drop=0.):
    out_f = out_features or in_features
    hidden_f = hidden_features or in_features
    fc1 = keras.layers.Dense(hidden_f)
    fc2 = keras.layers.Dense(out_f)
    drop = keras.layers.Dropout(drop)
    x = inputs
    x = fc1(x)
    x = keras.activations.relu(x)
    x = drop(x)
    x = fc2(x)
    x = drop(x)
    return x


def window_reverse(windows, window_size, H, W, C):
    x = tf.reshape(windows, shape=[-1, H // window_size,
                                   W // window_size, window_size, window_size, C])
    x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, shape=[-1, H, W, C])
    return x


def window_partition(x, window_size):
    B, H, W, C = x.get_shape().as_list()
    x = tf.reshape(x, shape=[-1, H // window_size,
                             window_size, W // window_size, window_size, C], name=f"take_{W}_{window_size}_{C}")
    x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    windows = tf.reshape(x, shape=[-1, window_size, window_size, C])
    return windows


def window_attention(inputs, dim, num_heads, window_size, attn_drop_val=0., proj_drop_val=0., mask=None, qkv_bias=True,
                     qk_scale=None):
    x = inputs
    B_, N, C = x.get_shape().as_list()
    QKV = keras.layers.Dense(dim * 3, use_bias=qkv_bias)
    head_dim = dim // num_heads
    scale = qk_scale or head_dim ** -0.5
    attn_drop = keras.layers.Dropout(attn_drop_val)
    proj = keras.layers.Dense(dim)
    proj_drop = keras.layers.Dropout(proj_drop_val)
    qkv_x = QKV(x)

    qkv = tf.transpose(tf.reshape(QKV(x), shape=[-1, N, 3, num_heads, C // num_heads]), perm=[2, 0, 3, 1, 4])
    q, k, v = qkv[0], qkv[1], qkv[2]

    q = q * scale

    zeros_initializer = tf.initializers.Zeros()
    relative_position_bias_table = tf.Variable(
        initial_value=zeros_initializer(shape=((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)),
        trainable=True
    )
    coords_h = np.arange(window_size[0])
    coords_w = np.arange(window_size[1])
    coords = np.stack(np.meshgrid(coords_h, coords_w, indexing='ij'))
    coords_flatten = coords.reshape(2, -1)

    relative_coords = coords_flatten[:, :,
                      None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.transpose([1, 2, 0])
    relative_coords[:, :, 0] += window_size[0] - 1
    relative_coords[:, :, 1] += window_size[1] - 1
    relative_coords[:, :, 0] *= 2 * window_size[1] - 1
    relative_position_index = relative_coords.sum(-1).astype(np.int64)
    relative_position_index = tf.Variable(initial_value=tf.convert_to_tensor(relative_position_index), trainable=False)

    attn = (q @ tf.transpose(k, perm=[0, 1, 3, 2]))
    relative_position_bias = tf.gather(relative_position_bias_table, tf.reshape(relative_position_index, shape=[-1]))
    relative_position_bias = tf.reshape(relative_position_bias,
                                        shape=[window_size[0] * window_size[1], window_size[0] * window_size[1], -1])
    relative_position_bias = tf.transpose(
        relative_position_bias, perm=[2, 0, 1])
    attn = attn + tf.expand_dims(relative_position_bias, axis=0)

    if mask is not None:
        nW = mask.get_shape()[0]  # tf.shape(mask)[0]
        attn = tf.reshape(attn, shape=[-1, nW, num_heads, N, N]) + tf.cast(
            tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), attn.dtype)
        attn = tf.reshape(attn, shape=[-1, num_heads, N, N])
        attn = tf.nn.softmax(attn, axis=-1)
    else:
        attn = tf.nn.softmax(attn, axis=-1)

    attn = attn_drop(attn)

    x = tf.transpose((attn @ v), perm=[0, 2, 1, 3])
    x = tf.reshape(x, shape=[-1, N, C])
    x = proj(x)
    x = proj_drop(x)
    return x


def window_cross_attention(input_x, input_y, dim, num_heads, window_size, attn_drop_val=0., proj_drop_val=0., mask=None, qkv_bias=True,
                     qk_scale=None):
    x = input_x
    y = input_y
    Bx_, Nx, Cx = x.get_shape().as_list()
    By_, Ny, Cy = y.get_shape().as_list()
    Q = keras.layers.Dense(dim * 1, use_bias=qkv_bias)
    KV = keras.layers.Dense(dim * 2, use_bias=qkv_bias)
    head_dim = dim // num_heads
    scale = qk_scale or head_dim ** -0.5
    attn_drop = keras.layers.Dropout(attn_drop_val)
    proj = keras.layers.Dense(dim)
    proj_drop = keras.layers.Dropout(proj_drop_val)

    qkv_x = tf.transpose(tf.reshape(Q(x), shape=[-1, Nx, 1, num_heads, Cx // num_heads]), perm=[2, 0, 3, 1, 4])
    qkv_y = tf.transpose(tf.reshape(KV(y), shape=[-1, Ny, 2, num_heads, Cy // num_heads]), perm=[2, 0, 3, 1, 4])
    q = qkv_x[0]
    k, v = qkv_y[0], qkv_y[1]
    print(q.get_shape().as_list(), k.get_shape().as_list(), v.get_shape().as_list())

    q = q * scale

    zeros_initializer = tf.initializers.Zeros()
    relative_position_bias_table = tf.Variable(
        initial_value=zeros_initializer(shape=((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)),
        trainable=True
    )
    coords_h = np.arange(window_size[0])
    coords_w = np.arange(window_size[1])
    coords = np.stack(np.meshgrid(coords_h, coords_w, indexing='ij'))
    coords_flatten = coords.reshape(2, -1)

    relative_coords = coords_flatten[:, :,
                      None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.transpose([1, 2, 0])
    relative_coords[:, :, 0] += window_size[0] - 1
    relative_coords[:, :, 1] += window_size[1] - 1
    relative_coords[:, :, 0] *= 2 * window_size[1] - 1
    relative_position_index = relative_coords.sum(-1).astype(np.int64)
    relative_position_index = tf.Variable(initial_value=tf.convert_to_tensor(relative_position_index), trainable=False)

    attn = (q @ tf.transpose(k, perm=[0, 1, 3, 2]))
    relative_position_bias = tf.gather(relative_position_bias_table, tf.reshape(relative_position_index, shape=[-1]))
    relative_position_bias = tf.reshape(relative_position_bias,
                                        shape=[window_size[0] * window_size[1], window_size[0] * window_size[1], -1])
    relative_position_bias = tf.transpose(
        relative_position_bias, perm=[2, 0, 1])
    print(relative_position_bias.get_shape().as_list())
    attn = attn + tf.expand_dims(relative_position_bias, axis=0)

    if mask is not None:
        nW = mask.get_shape()[0]  # tf.shape(mask)[0]
        attn = tf.reshape(attn, shape=[-1, nW, num_heads, Nx, Nx]) + tf.cast(
            tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), attn.dtype)
        attn = tf.reshape(attn, shape=[-1, num_heads, Nx, Nx])
        attn = tf.nn.softmax(attn, axis=-1)
    else:
        attn = tf.nn.softmax(attn, axis=-1)

    attn = attn_drop(attn)

    x = tf.transpose((attn @ v), perm=[0, 2, 1, 3])
    x = tf.reshape(x, shape=[-1, Nx, Cx])
    x = proj(x)
    x = proj_drop(x)
    return x


def drop_path_func(inputs, drop_prob, is_training):
    if (not is_training) or (drop_prob == 0.):
        return inputs

    # Compute keep_prob
    keep_prob = 1.0 - drop_prob

    # Compute drop_connect tensor
    random_tensor = keep_prob
    shape = (tf.shape(inputs)[0],) + (1,) * \
            (len(tf.shape(inputs)) - 1)
    random_tensor += tf.random.uniform(shape, dtype=inputs.dtype)
    binary_tensor = tf.floor(random_tensor)
    output = tf.math.divide(inputs, keep_prob) * binary_tensor
    return output


def drop_path(x, drop_prob=None, training=None):
    return drop_path_func(x, drop_prob, training)


def swin_transformer_block(inputs, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4.,
                           qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path_prob=0., out_f=None,
                           norm_layer=tf.keras.layers.LayerNormalization):
    if min(input_resolution) <= window_size:
        shift_size = 0
        window_size = min(input_resolution)
    assert 0 <= shift_size < window_size, "shift_size must in 0-window_size"

    H, W = input_resolution
    B, L, C = inputs.get_shape().as_list()
    print(H, W, B, L, C)

    assert L == H * W, "input feature has wrong size"

    x = inputs
    shortcut = x

    norm1 = norm_layer(epsilon=1e-5)
    x = norm1(x)
    x = tf.reshape(x, shape=[-1, H, W, C])

    # cyclic shift
    if shift_size > 0:
        shifted_x = tf.roll(
            x, shift=[-shift_size, -shift_size], axis=[1, 2])
    else:
        shifted_x = x

    # partition windows
    x_windows = window_partition(shifted_x, window_size)
    x_windows = tf.reshape(x_windows, shape=[-1, window_size * window_size, C])
    #{{node model_1/tf_op_layer_Reshape_5/Reshape_5}} = Reshape[T=DT_FLOAT, Tshape=DT_INT32, _cloned=true](model_1/dense/BiasAdd, model_1/tf_op_layer_Reshape_5/Reshape_5/shape)

    if shift_size > 0:
        H, W = input_resolution
        img_mask = np.zeros([1, H, W, 1])
        h_slices = (slice(0, -window_size),
                    slice(-window_size, -shift_size),
                    slice(-shift_size, None))
        w_slices = (slice(0, -window_size),
                    slice(-window_size, -shift_size),
                    slice(-shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        img_mask = tf.convert_to_tensor(img_mask)
        mask_windows = window_partition(img_mask, window_size)
        mask_windows = tf.reshape(
            mask_windows, shape=[-1, window_size * window_size])
        attn_mask = tf.expand_dims(
            mask_windows, axis=1) - tf.expand_dims(mask_windows, axis=2)
        attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
        attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
        attn_mask = tf.Variable(
            initial_value=attn_mask, trainable=False)
    else:
        attn_mask = None
    # W-MSA/SW-MSA
    attn_windows = window_cross_attention(input_x=x_windows, input_y=x_windows, dim=dim, window_size=(window_size, window_size),
                                    num_heads=num_heads,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop_val=attn_drop, proj_drop_val=drop,
                                    mask=attn_mask)

    # merge windows
    attn_windows = tf.reshape(
        attn_windows, shape=[-1, window_size, window_size, C])
    shifted_x = window_reverse(attn_windows, window_size, H, W, C)

    # reverse cyclic shift
    if shift_size > 0:
        x = tf.roll(shifted_x, shift=[shift_size, shift_size], axis=[1, 2])
    else:
        x = shifted_x
    x = tf.reshape(x, shape=[-1, H * W, C])

    # FFN
    x = shortcut + drop_path(x)
    norm2 = norm_layer(epsilon=1e-5)
    mlp_hidden_dim = int(dim * mlp_ratio)
    mlp_layer = mlp(inputs=norm2(x), in_features=dim, hidden_features=mlp_hidden_dim, drop=drop, out_features=out_f)
    drop_path_layer = drop_path(mlp_layer, drop_path_prob if drop_path_prob > 0. else 0.)
    x = x + drop_path_layer
    return x


def cross_swin_input(inputs, input_resolution, window_size=7, shift_size=0,
                           norm_layer=tf.keras.layers.LayerNormalization):
    if min(input_resolution) <= window_size:
        shift_size = 0
        window_size = min(input_resolution)
    assert 0 <= shift_size < window_size, "shift_size must in 0-window_size"

    H, W = input_resolution
    B, L, C = inputs.get_shape().as_list()
    print(H, W, B, L, C)

    assert L == H * W, "input feature has wrong size"

    x = inputs
    shortcut = x

    norm1 = norm_layer(epsilon=1e-5)
    x = norm1(x)
    x = tf.reshape(x, shape=[-1, H, W, C], name=f"randomr_{H}_{W}_{C}")

    # cyclic shift
    if shift_size > 0:
        shifted_x = tf.roll(
            x, shift=[-shift_size, -shift_size], axis=[1, 2])
    else:
        shifted_x = x

    # partition windows
    # window_size = 8 if window_size == 7 else window_size
    x_windows = window_partition(shifted_x, window_size)
    x_windows = tf.reshape(x_windows, shape=[-1, window_size * window_size, C])
    #{{node model_1/tf_op_layer_Reshape_5/Reshape_5}} = Reshape[T=DT_FLOAT, Tshape=DT_INT32, _cloned=true](model_1/dense/BiasAdd, model_1/tf_op_layer_Reshape_5/Reshape_5/shape)
    # window_size = 7 if window_size == 8 else window_size
    if shift_size > 0:
        H, W = input_resolution
        img_mask = np.zeros([1, H, W, 1])
        h_slices = (slice(0, -window_size),
                    slice(-window_size, -shift_size),
                    slice(-shift_size, None))
        w_slices = (slice(0, -window_size),
                    slice(-window_size, -shift_size),
                    slice(-shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        img_mask = tf.convert_to_tensor(img_mask)
        mask_windows = window_partition(img_mask, window_size)
        mask_windows = tf.reshape(
            mask_windows, shape=[-1, window_size * window_size])
        attn_mask = tf.expand_dims(
            mask_windows, axis=1) - tf.expand_dims(mask_windows, axis=2)
        attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
        attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
        attn_mask = tf.Variable(
            initial_value=attn_mask, trainable=False)
    else:
        attn_mask = None
    # W-MSA/SW-MSA
    print("windows")
    print(x_windows.get_shape().as_list())
    return x_windows, attn_mask


def cross_swin_transformer_block(input_x, input_y, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4.,
                           qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path_prob=0., out_f=None,
                           norm_layer=tf.keras.layers.LayerNormalization):
    if min(input_resolution) <= window_size:
        shift_size = 0
        window_size = min(input_resolution)
    assert 0 <= shift_size < window_size, "shift_size must in 0-window_size"

    H, W = input_resolution
    Bx, Lx, Cx = input_x.get_shape().as_list()
    By, Ly, Cy = input_x.get_shape().as_list()

    print(H, W, Bx, Lx, Cx, By, Ly, Cy)

    assert Lx == H * W, "input feature x has wrong size"
    assert Ly == H * W, "input feature x has wrong size"

    x = input_x
    y = input_y
    print("shapes")
    print(x.get_shape().as_list(), y.get_shape().as_list())

    shortcut = x

    norm1 = norm_layer(epsilon=1e-5)
    x = norm1(x)
    x = tf.reshape(x, shape=[-1, H, W, Cx])
    y = norm1(y)
    y = tf.reshape(y, shape=[-1, H, W, Cy])
    # cyclic shift
    if shift_size > 0:
        shifted_x = tf.roll(
            x, shift=[-shift_size, -shift_size], axis=[1, 2])
        shifted_y = tf.roll(
            y, shift=[-shift_size, -shift_size], axis=[1, 2])
    else:
        shifted_x = x
        shifted_y = y

    # partition windows
    x_windows = window_partition(shifted_x, window_size)
    x_windows = tf.reshape(x_windows, shape=[-1, window_size * window_size, Cx])
    y_windows = window_partition(shifted_y, window_size)
    y_windows = tf.reshape(y_windows, shape=[-1, window_size * window_size, Cy])

    # {{node model_1/tf_op_layer_Reshape_5/Reshape_5}} = Reshape[T=DT_FLOAT, Tshape=DT_INT32, _cloned=true](model_1/dense/BiasAdd, model_1/tf_op_layer_Reshape_5/Reshape_5/shape)

    if shift_size > 0:
        H, W = input_resolution
        img_mask = np.zeros([1, H, W, 1])
        h_slices = (slice(0, -window_size),
                    slice(-window_size, -shift_size),
                    slice(-shift_size, None))
        w_slices = (slice(0, -window_size),
                    slice(-window_size, -shift_size),
                    slice(-shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        img_mask = tf.convert_to_tensor(img_mask)
        mask_windows = window_partition(img_mask, window_size)
        mask_windows = tf.reshape(
            mask_windows, shape=[-1, window_size * window_size])
        attn_mask = tf.expand_dims(
            mask_windows, axis=1) - tf.expand_dims(mask_windows, axis=2)
        attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
        attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
        attn_mask = tf.Variable(
            initial_value=attn_mask, trainable=False)
    else:
        attn_mask = None
    # W-MSA/SW-MSA
    attn_windows = window_cross_attention(input_x=x_windows, input_y=y_windows, dim=dim,
                                          window_size=(window_size, window_size),
                                          num_heads=num_heads,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop_val=attn_drop,
                                          proj_drop_val=drop,
                                          mask=attn_mask)

    # merge windows
    attn_windows = tf.reshape(
        attn_windows, shape=[-1, window_size, window_size, Cx])
    shifted_x = window_reverse(attn_windows, window_size, H, W, Cx)

    # reverse cyclic shift
    if shift_size > 0:
        x = tf.roll(shifted_x, shift=[shift_size, shift_size], axis=[1, 2])
    else:
        x = shifted_x
    x = tf.reshape(x, shape=[-1, H * W, Cx])

    # FFN
    x = shortcut + drop_path(x)
    norm2 = norm_layer(epsilon=1e-5)
    mlp_hidden_dim = int(dim * mlp_ratio)
    mlp_layer = mlp(inputs=norm2(x), in_features=dim, hidden_features=mlp_hidden_dim, drop=drop, out_features=out_f)
    drop_path_layer = drop_path(mlp_layer, drop_path_prob if drop_path_prob > 0. else 0.)
    x = x + drop_path_layer
    return x



def basic_layer(inputs_x, inputs_y, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4., qkv_bias=True,
                qk_scale=None,p_m_merging=None, out_f=None,
                drop=0., attn_drop=0., drop_path_prob=0., norm_layer=keras.layers.LayerNormalization, downsample=None,
                use_checkpoint=False):
    x = inputs_x
    y = inputs_y
    for i in range(depth):
        x = cross_swin_transformer_block(x, y, dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                                   window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2,
                                   mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop,
                                   attn_drop=attn_drop, drop_path_prob=drop_path_prob, norm_layer=norm_layer, out_f=out_f)
    if downsample:
        return x, downsample(x, input_resolution, dim=p_m_merging if p_m_merging else dim, norm_layer=norm_layer)
    return x, downsample


def patch_embed(inputs, img_size=(224, 224), patch_size=(4, 4), in_chans=3, embed_dim=96, norm_layer=None):
    x = inputs
    B, H, W, C = x.get_shape().as_list()
    assert H == img_size[0] and W == img_size[
        1], f"Input image size ({H}*{W}) doesn't match model ({img_size[0]}*{img_size[1]})."
    patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
    # num_patches = patches_resolution[0] * patches_resolution[1]
    proj = keras.layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size)
    x = proj(x)
    x = tf.reshape(x, shape=[-1, (H // patch_size[0]) * (W // patch_size[0]), embed_dim])
    if norm_layer:
        norm = norm_layer(epsilon=1e-5)
        x = norm(x)
    return x


def patch_merging(inputs, input_resolution, dim, norm_layer=keras.layers.LayerNormalization):
    x = inputs
    H, W = input_resolution
    B, L, C = x.get_shape().as_list()
    assert L == H * W, "input feature has wrong size"
    assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
    x = tf.reshape(x, shape=[-1, H, W, C])
    x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
    x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
    x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
    x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
    x = tf.concat([x0, x1, x2, x3], axis=-1)
    x = tf.reshape(x, shape=[-1, (H // 2) * (W // 2), 4 * C])
    reduction = keras.layers.Dense(2 * dim, use_bias=False)
    norm = norm_layer(epsilon=1e-5)
    x = norm(x)
    x = reduction(x)

    return x
