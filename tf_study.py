import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from utils import TrainLogger, TestLogger

import os
import random

data = load_iris()
X = data.data
y = data.target
y_onehot=tf.keras.utils.to_categorical(y,num_classes=3)
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.5, random_state=42)

clf = LogisticRegression()
clf.fit(X_train, np.argmax(y_train,axis=1))
train_acc = clf.score(X_train, np.argmax(y_train,axis=1))
test_acc = clf.score(X_test, np.argmax(y_test,axis=1))
print(f'Logistic Regression - Train Accuracy: {train_acc}, Test Accuracy: {test_acc}')

# model=tf.keras.models.Sequential([
#     #tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Dense(16, input_dim=4,activation='relu'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Dense(16, input_dim=16,activation='relu'),
#     tf.keras.layers.Dense(3, input_dim=16,activation='softmax',
#                           kernel_regularizer=tf.keras.regularizers.l2(1e-2)),
# ])

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 初始化器
he_init = tf.keras.initializers.HeNormal(seed=SEED)          # 用于 ReLU 层
glorot_init = tf.keras.initializers.GlorotUniform(seed=SEED) # 用于输出层
bias_zero = tf.keras.initializers.Zeros()

# 模型定义（只在第一层声明输入形状）
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(4,)),
    tf.keras.layers.Dense(16, activation='relu',
                          kernel_initializer=he_init,
                          bias_initializer=bias_zero),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(16, activation='relu',
                          kernel_initializer=he_init,
                          bias_initializer=bias_zero),
    tf.keras.layers.Dense(3, activation='softmax',
                          kernel_initializer=glorot_init,
                          bias_initializer=bias_zero,
                          kernel_regularizer=tf.keras.regularizers.l2(1e-2)),
])

optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,weight_decay=1e-3)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

init_loss,init_acc=model.evaluate(X_train,y_train)
print('Initial Loss:', init_loss)
print('Initial Accuracy:', init_acc)

# To ensure the model performs well on the test set
# L2 regularization, and large enough epochs as well as batch size are used.
interval=10
train_logger,test_logger=TrainLogger(interval),TestLogger(X_test,y_test,interval)
model.fit(X_train,y_train,epochs=300,batch_size=64,verbose=0,
          callbacks=[train_logger,test_logger])

loss,accuracy=model.evaluate(X_test,y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')
predictions=model.predict(X_test[:5])
print('Predictions:', tf.argmax(predictions,axis=1))
print('True Labels:', y_test[:5])



plt.figure()
plt.plot(train_logger.losses, label='Train Loss')
plt.plot(test_logger.losses, label='Test Loss')
#plt.scatter(np.arange(0,len(train_logger.losses),interval),test_logger.losses, label='Test Loss',marker='^')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()


plt.figure()
plt.plot(train_logger.accs, label='Train Accuracy')
plt.plot(test_logger.accs, label='Test Accuracy')
#plt.scatter(np.arange(0,len(train_logger.accs),interval),test_logger.accs, label='Test Accuracy',marker='^')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
