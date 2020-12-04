import numpy as np
import random
import tensorflow as tf
import os
import GridWorld
import matplotlib.pyplot as plt

env = GridWorld.gameEnv(size=5)


# Deep Q-Network
class Qnetwork():
    def __init__(self, h_size):
        # 輸入時被扁平化的長度為84*84*3=21168的向量
        self.scalarInput = tf.placeholder(shape=[None, 21168], dtype=tf.float32)
        # 將輸入恢復成多個84*84*3尺寸的圖片
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, 84, 84, 3])
        # 第1個卷積層：卷積核尺寸8*8，步長為4*4，輸出通道數為32，padding模式VALID，輸出維度20*20*32
        self.conv1 = tf.contrib.layers.convolution2d(inputs=self.imageIn, num_outputs=32, kernel_size=[8, 8],
                                                     stride=[4, 4], padding='VALID',
                                                     biases_initializer=None)
        # 第2個卷積層：卷積核尺寸4*4，步長為2*2，輸出通道數為64，padding模式VALID，輸出維度9*9*64
        self.conv2 = tf.contrib.layers.convolution2d(inputs=self.conv1, num_outputs=64, kernel_size=[4, 4],
                                                     stride=[2, 2], padding='VALID',
                                                     biases_initializer=None)
        # 第3個卷積層：卷積核尺寸3*3，步長為1*1，輸出通道數為64，padding模式VALID，輸出維度7*7*64
        self.conv3 = tf.contrib.layers.convolution2d(inputs=self.conv2, num_outputs=64, kernel_size=[3, 3],
                                                     stride=[1, 1], padding='VALID',
                                                     biases_initializer=None)
        # 第4個卷積層：卷積核尺寸7*7，步長為1*1，輸出通道數為512，padding模式VALID，輸出維度1*1*512
        self.conv4 = tf.contrib.layers.convolution2d(inputs=self.conv3, num_outputs=512, kernel_size=[7, 7],
                                                     stride=[1, 1], padding='VALID',
                                                     biases_initializer=None)
        # 將卷積層輸出平均分拆成兩段，AC和VC，分別對應Action價值和環境本身價值
        self.streamAC, self.streamVC = tf.split(self.conv4, 2, 3)  # 2段，第3維度
        # 扁平化處理AC
        self.streamA = tf.contrib.layers.flatten(self.streamAC)
        # 扁平化處理VC
        self.streamV = tf.contrib.layers.flatten(self.streamVC)
        # 創建線性全連接層初始化權重W
        self.AW = tf.Variable(tf.random_normal([h_size // 2, env.actions]))
        self.VW = tf.Variable(tf.random_normal([h_size // 2, 1]))
        # 獲得Advantage結果
        self.Advantage = tf.matmul(self.streamA, self.AW)
        # 獲得Value結果
        self.Value = tf.matmul(self.streamV, self.VW)
        # Q值由Advantage和Value複合而成
        self.Qout = self.Value + tf.subtract(self.Advantage,
                                             tf.reduce_mean(self.Advantage, reduction_indices=1, keep_dims=True))
        # 計算Q最大的Action
        self.predict = tf.argmax(self.Qout, 1)
        # 定義目標Q值輸入
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        # 定義動作action輸入
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        # actions轉化為onehot編碼模式
        self.actions_onehot = tf.one_hot(self.actions, env.actions, dtype=tf.float32)
        # 將Qout和actions_onehot相乘得到Q值
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), reduction_indices=1)
        # 計算targetQ和Q的均方誤差
        self.td_error = tf.square(self.targetQ - self.Q)
        # 定義loss損失函數
        self.loss = tf.reduce_mean(self.td_error)
        # 使用Adam優化器
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        # 最小化loss函數
        self.updateModel = self.trainer.minimize(self.loss)


# Experience策略
class experience_buffer():
    # 存儲樣本最大容量buffer size
    def __init__(self, buffer_size=50000):
        self.buffer = []
        self.buffer_size = buffer_size

    # 存儲樣本越界的話，清空早期一些樣本
    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    # 隨機抽樣一些樣本
    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])


# 扁平化資料84*84*3函數
def processState(states):
    return np.reshape(states, [21168])


# 更新DQN模型參數方法：全部參數、主DQN學習率
def updateTargetGraph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    # 取前一半參數即主DQN模型參數
    for idx, var in enumerate(tfVars[0:total_vars // 2]):
        # 緩慢更新參數
        op_holder.append(tfVars[idx + total_vars // 2].assign(
            (var.value() * tau) + ((1 - tau) * tfVars[idx + total_vars // 2].value())))
    return op_holder


# 執行更新模型參數操作
def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)


batch_size = 32  # 從experience buffer中獲取樣本批次尺寸
update_freq = 4  # 每隔4步更新一次模型參數
y = .99  # Q值衰減係數
startE = 1  # 起始隨機Action概率，訓練需要隨機搜索，預測時不需要
endE = 0.1  # 最終執行Action概率
anneling_steps = 10000.  # 初始隨機概率到最終隨機概率下降的步數
num_episodes = 10000  # 實驗次數
pre_train_steps = 10000  # 使用DQN選擇Action前進行多少步隨機測試
max_epLength = 50  # 每個episode執行多少次Action
load_model = False  # 是否讀取之前訓練的模型
path = "./dqn"  # 模型存儲路徑
h_size = 512  # DQN全連接層隱含節點數
tau = 0.001  # target DQN向主DQN學習的學習率

tf.reset_default_graph()
# 初始化主DQN
mainQN = Qnetwork(h_size)
# 初始化輔助targetDQN
targetQN = Qnetwork(h_size)

# 初始化全域變數
init = tf.global_variables_initializer()

# 獲得所有可訓練參數
trainables = tf.trainable_variables()

# 創建更新targetDQN參數的操作
targetOps = updateTargetGraph(trainables, tau)

# 初始化experience buffer
myBuffer = experience_buffer()

# 設置當前學習率
e = startE
# 計算每一步衰減值
stepDrop = (startE - endE) / anneling_steps

# 初始化存儲episode的reward清單
rList = []
# 初始化總步數
total_steps = 0

# 創建模型記憶體並檢驗保存路徑
saver = tf.train.Saver()
if not os.path.exists(path):
    os.makedirs(path)


with tf.Session() as sess:
    # 如果已有模型
    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    # 初始化全部參數
    sess.run(init)
    # 執行更新參數操作
    updateTarget(targetOps, sess)
    # 創建實驗迴圈
    for i in range(num_episodes + 1):
        episodeBuffer = experience_buffer()
        # 重置環境
        s = env.reset()
        # 獲取環境資訊並將其扁平化
        s = processState(s)
        # done標記
        d = False
        # episode內部的總reward
        rAll = 0
        # episode內部的步數
        j = 0
        # 創建內迴圈，反覆運算每一次執行的Action
        while j < max_epLength:
            j += 1
            # 步數小於pre_train_steps時，強制使用隨機Action
            if np.random.rand(1) < e or total_steps < pre_train_steps:
                a = np.random.randint(0, 4)
            # 達到pre_train_steps後，保留一個較小概率隨機選擇Action，若不選擇隨機Action，將當前狀態s傳入DQN，預測Action
            else:
                a = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: [s]})[0]
            # 執行一步Action，和獲取接下來狀態s1，reward和done標記
            s1, r, d = env.step(a)
            # 扁平化處理s1
            s1 = processState(s1)
            # 總步數+1
            total_steps += 1
            # 將資料和結果傳入episodeBuffer存儲
            episodeBuffer.add(
                np.reshape(np.array([s, a, r, s1, d]), [1, 5]))  # Save the experience to our episode buffer.
            # 如果總步數超過pre_train_steps
            if total_steps > pre_train_steps:
                # 持續降低隨機選擇Action概率直到endE
                if e > endE:
                    e -= stepDrop
                # 步數達到update_freq整倍數時，進行一次訓練，更新一次參數
                if total_steps % (update_freq) == 0:
                    # 從myBuffer獲取樣本
                    trainBatch = myBuffer.sample(batch_size)
                    # 訓練樣本中第3列資訊即下一個狀態s1傳入主DQN，得到主模型選擇的Action
                    A = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 3])})
                    # 再將s1傳入輔助的targetDQN，得到s1狀態下所有Action的Q值
                    Q = sess.run(targetQN.Qout, feed_dict={targetQN.scalarInput: np.vstack(trainBatch[:, 3])})
                    # 將主DQN輸出的Action和targetDQN輸出的Q值，得到doubleQ
                    doubleQ = Q[range(batch_size), A]
                    # 使用訓練樣本第二列reward值+doubleQ*衰減係數y，獲得targetQ
                    targetQ = trainBatch[:, 2] + y * doubleQ
                    # 傳入當前狀態s，學習targetQ和這一步的Action
                    _ = sess.run(mainQN.updateModel,
                                 feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 0]), mainQN.targetQ: targetQ,
                                            mainQN.actions: trainBatch[:, 1]})
                    # 更新一次主DQN參數
                    updateTarget(targetOps, sess)
            # 累計當前獲取的reward
            rAll += r
            # 更新當前狀態為下一步實驗做準備
            s = s1
            # 如果done為True，則終止實驗
            if d == True:
                break

        # episode內部的episodeBuffer添加到myBuffer中
        myBuffer.add(episodeBuffer.buffer)
        # episode中reward添加到rList中
        rList.append(rAll)
        # 每25次展示一次平均reward
        if i > 0 and i % 25 == 0:
            print('episode', i, ', average reward of last 25 episode', np.mean(rList[-25:]))
        # 每1000次保存當前模型
        if i > 0 and i % 1000 == 0:
            saver.save(sess, path + '/model-' + str(i) + '.cptk')
            print("Saved Model")
    saver.save(sess, path + '/model-' + str(i) + '.cptk')

# 計算每100個episode平均reward
rMat = np.resize(np.array(rList), [len(rList) // 100, 100])
# 使用plot繪製reward變化趨勢
rMean = np.average(rMat, 1)
plt.plot(rMean)