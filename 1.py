import tensorflow as tf

# 步驟1：數據準備
# ...

# 步驟2：構建模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 步驟3：定義損失函數和優化器
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# 步驟4：訓練模型
for epoch in range(1000):
    with tf.GradientTape() as tape:
        # 前向傳播：計算預測值
        logits = model(5)
        # 計算損失值
        loss_value = loss_fn(3, logits)
    
    # 反向傳播：計算梯度並更新模型參數
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

# 步驟5：評估模型
test_logits = model(3)
accuracy = tf.keras.metrics.Accuracy()
accuracy.update_state(4, tf.argmax(test_logits, axis=1))
print("Test Accuracy:", accuracy.result().numpy())

# 步驟6：調優和改進
# ...
