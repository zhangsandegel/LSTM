# %% [markdown]
# ## Classification

# %%
# load packages
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
import platform
if platform.system() == "Windows":
    plt.rcParams['font.family'] = ['SimHei'] # Windows
elif platform.system() == "Darwin":
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # MacOS
plt.rcParams['axes.unicode_minus']=False 


d_list = []
folder = '/kaggle/input/gas-sensor-array-drift-dataset'
files = sorted([i for i in os.listdir(folder) if i.endswith('.dat')])
for file in files:
    print(file)
    df = pd.read_csv(os.path.join(folder,file),header=None,sep=' ')
    df = df.dropna(axis=1)
    df = df.astype(str).applymap(lambda x:x.split(':')[-1])
    df['label'] = df[0].map(lambda x:int(x.split(';')[0]))
    df['var'] = df[0].map(lambda x:float(x.split(';')[1]))
    df = df.drop(columns=[0])
    df = df.astype(float)
    d_list.append(df)

# %%
df = pd.concat(d_list,axis=0).reset_index(drop=True)
df = df[~df['label'].isin([0])]
df['label'] = df['label'] - 1

# %%
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(df.iloc[:,:128])
X = X.reshape(-1,X.shape[1],1)

# %%
y = df.iloc[:,128].values

# %%
df.iloc[:,128].unique()

# %%
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

# 数据预处理
# 对类别标签y进行独热编码
y_encoded = to_categorical(y)

# 拆分数据集为训练集和测试集（8:2）
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 构建LSTM多分类模型
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(32, activation='relu'))
model.add(Dense(6, activation='softmax'))  # 假设有3个类别，输出层使用softmax激活函数

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
# 训练模型
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test,y_test))

# 评估模型性能
loss = model.evaluate(X_test, y_test)
print("Test Loss:", loss)

# %% [markdown]
# ## Regression

# %%
import os
# load packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
import platform
if platform.system() == "Windows":
    plt.rcParams['font.family'] = ['SimHei'] # Windows
elif platform.system() == "Darwin":
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # MacOS
plt.rcParams['axes.unicode_minus']=False 

# 数据文件地址
folder = '/kaggle/input/gas-sensor-array-temperature-modulation/gas-sensor-array-temperature-modulation'
files = sorted([i for i in os.listdir(folder) if i.endswith('.csv')])
data = pd.concat(
    [pd.read_csv(os.path.join(folder,file)) for file in files],axis=0
)

# %%
data = data.sort_values(by=['Time (s)']).head(10000).copy()

# %%
# 对数据进行归一化处理
from sklearn.preprocessing import StandardScaler,MinMaxScaler

features = [ 'Humidity (%r.h.)','Flow rate (mL/min)', 'Heater voltage (V)', 'R1 (MOhm)', 'R2 (MOhm)',
       'R3 (MOhm)', 'R4 (MOhm)', 'R5 (MOhm)', 'R6 (MOhm)', 'R7 (MOhm)',
       'R8 (MOhm)', 'R9 (MOhm)', 'R10 (MOhm)', 'R11 (MOhm)', 'R12 (MOhm)',
       'R13 (MOhm)', 'R14 (MOhm)']
target = 'Temperature (C)'


x_scaler = MinMaxScaler()
data[features] = x_scaler.fit_transform(data[features])
y_scaler = MinMaxScaler()
data[[target]] = y_scaler.fit_transform(data[[target]])

# %%

import numpy as np
class WindowGenerator():
    """A class that generates time series data"""
    def __init__(self, input_width, label_width, shift,
               data = None,label_columns=None,feature_columns=None):
        # Store the raw data. pd.DataFrame type
        self.data = data
        # Work out the label column indices.
        self.label_columns = label_columns
        self.feature_columns = feature_columns
        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        # all cols
        self.col_dic = {col:i for i,col in enumerate(data.columns)}
        self.label_col_idx = [self.col_dic[col] for col in self.label_columns]
        self.feature_col_idx = [self.col_dic[col] for col in self.feature_columns]
        # change to numpy array
        self.arr = data.values
        self.total_window_size = input_width + shift
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
        self.split_window()
        print(self)

    def split_window(self):
        self.X = []
        self.y = []
        for i in range(self.total_window_size,len(self.arr)):
            window_data = self.arr[i-self.total_window_size:i]
            self.X.append(window_data[np.ix_(self.input_indices,self.feature_col_idx)])
            self.y.append(window_data[np.ix_(self.label_indices,self.label_col_idx)])
        self.X = np.asarray(self.X)
        self.y = np.asarray(self.y)

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}',
            f'Label column index(s): {self.label_col_idx}',
            f'Feature column name(s): {self.feature_columns}',
            f'Feature column index(s): {self.feature_col_idx}',
            f'Origin dataset shape: {self.arr.shape}',
            f'X shape: {self.X.shape}',
            f'y shape: {self.y.shape}',
            ])

# 传入一个dataframe类型
wg = WindowGenerator(input_width=10,
					label_width=1,
					shift=1,
					label_columns=[target],
					feature_columns=features,
					data=data
					)

# %%
X = wg.X
y = wg.y

# %%
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 拆分数据集为训练集和测试集（8:2）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建LSTM回归模型
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))  # 输出层没有使用激活函数，因为是回归问题

# 编译模型
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mae'])

print(model.summary())
# 训练模型
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data = (X_test,y_test))


# %%
# 评估模型性能
loss = model.evaluate(X_test, y_test)
print("Test Loss:", loss)

# %%



