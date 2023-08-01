import tensorflow as tf

# 定義 Transformer 模型的 Encoder Layer
class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.multi_head_attention = tf.keras.layers.MultiHeadAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(d_ff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training=True):
        attn_output = self.multi_head_attention(inputs, inputs, attention_mask=None, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layer_norm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layer_norm2(out1 + ffn_output)
        return out2

# 定義 Transformer 模型
class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_vocab_size, maximum_position_encoding, dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = self.positional_encoding(maximum_position_encoding, d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.enc_layers = [TransformerEncoderLayer(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_layers)]

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(tf.range(position, dtype=tf.float32)[:, tf.newaxis],
                                     tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
                                     d_model)
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return pos * angle_rates

    def call(self, inputs, training=True):
        seq_len = tf.shape(inputs)[1]
        word_embedding = self.embedding(inputs)
        word_embedding *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        word_embedding += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(word_embedding, training=training)
        for layer in self.enc_layers:
            x = layer(x, training=training)
        return x

# 定義 Transformer 模型的參數
num_layers = 4
d_model = 128
num_heads = 8
d_ff = 512
input_vocab_size = 10000
maximum_position_encoding = 1000

# 建立 Transformer 模型
transformer = TransformerEncoder(num_layers, d_model, num_heads, d_ff, input_vocab_size, maximum_position_encoding)

# 輸入範例
sample_input = tf.random.uniform((64, 100))  # 假設輸入長度為 100 的序列，總共 64 個樣本

# 使用 Transformer 模型進行預測
outputs = transformer(sample_input)

# 輸出 Transformer 模型的預測結果
print(outputs.shape)  # (64, 100, 128)