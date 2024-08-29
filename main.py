import time

from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import warnings
from math import sqrt
from tkinter import ttk
from tkinter import messagebox
import random
import pickle
import tensorflow as tf
from tensorflow.keras.layers import Lambda, Dense
from keras.utils import to_categorical
from sklearn.metrics import mean_squared_error, precision_score, recall_score, mean_absolute_error, confusion_matrix, \
    f1_score, roc_auc_score, r2_score
from tabulate import tabulate
import torch.nn as nn
import seaborn as sns
import torch.nn.functional as F
from Data.utilis.utils import *

warnings.filterwarnings("ignore")

# ----------------------Loading the Data-------------------------#

# Amazon.com, Inc.
# Apple Inc.
# Microsoft Corporation
# Tesla Inc.
# Google


print('\nData Loading.....')
tweets = pd.read_csv(os.getcwd() + '\\Data\\stock_tweets.csv')
all_stocks = pd.read_csv(os.getcwd() + '\\Data\\stock_yfinance_data.csv')
print('\nAmazon.com, Inc......')
data1 = tweets[tweets['Stock Name'] == 'AMZN']
print('\nApple Inc.......')
data2 = tweets[tweets['Stock Name'] == 'AAPL']
print('\nMicrosoft Corporation......')
data3 = tweets[tweets['Stock Name'] == 'MSFT']
print('\nTesla Inc......')
data4 = tweets[tweets['Stock Name'] == 'TSLA']
print('\nGoogle......')
data5 = tweets[tweets['Stock Name'] == 'GOOG']

# ---------------------Pre-processing and Data balancing------------------------#

def text_lowercase(text):
    return text.lower()

for i in range(len(data1)):
    data1['Tweet'].iloc[i] = text_lowercase(data1['Tweet'].iloc[i])
for i in range(len(data2)):
    data2['Tweet'].iloc[i] = text_lowercase(data2['Tweet'].iloc[i])
for i in range(len(data3)):
    data3['Tweet'].iloc[i] = text_lowercase(data3['Tweet'].iloc[i])
for i in range(len(data4)):
    data4['Tweet'].iloc[i] = text_lowercase(data4['Tweet'].iloc[i])
for i in range(len(data5)):
    data5['Tweet'].iloc[i] = text_lowercase(data5['Tweet'].iloc[i])

sentiment_data_AMZN, sentiment_data_AAPL, sentiment_data_MSFT, sentiment_data_TSLA , sentiment_data_GOOG = sentiment(data1, data2, data3,data4,data5)
final_df_AMZN, final_df_AAPL, final_df_MSFT, final_df_TSLA, final_df_GOOG = final_stock(all_stocks, sentiment_data_AMZN, sentiment_data_AAPL,
                                                          sentiment_data_MSFT, sentiment_data_TSLA, sentiment_data_GOOG)


# -------------------Analysis plot of closing prices of AMZN,AAPL,MSFT, TSLA-------------------

plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, 253), final_df_AMZN['Close'], color='r', label='AMZN')
plt.plot(np.arange(1, 253), final_df_AAPL['Close'], color='b', label='AAPL')
plt.plot(np.arange(1, 253), final_df_MSFT['Close'], color='m', label='MSFT')
plt.plot(np.arange(1, 253), final_df_TSLA['Close'], color='k', label='TSLA')
plt.plot(np.arange(1, 253), final_df_GOOG['Close'], color='gold', label='GOOG')

plt.xlabel('Month/Year', fontsize=16, fontweight='bold')
plt.ylabel('Closing Prices ($)', fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.xticks(fontsize=16, fontweight='bold')
plt.xticks([0, 50, 100, 150, 200, 250], ['1/2022', '6/2022', '1/2023', '6/2023', '1/2024', '6/2024'],
           rotation=0)
prop = {'size': 16, 'weight': 'bold'}
plt.legend(loc='upper right', fancybox=True, prop=prop)
plt.ylim([80,500])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
# Percentage data for each stock
stock_data = {
    "AMZN": len(final_df_AMZN),
    "AAPL": len(final_df_AAPL),
    "MSFT": len(final_df_MSFT),
    "TSLA": len(final_df_TSLA),
    "GOOG": len(final_df_GOOG)
}

# Extract the data
lab = stock_data.keys()
sizes = stock_data.values()
colors = ['lightcoral', 'lightskyblue', 'm', 'lightgrey','gold']
explode = (0.1, 0, 0, 0,0)  # explode the 1st slice (AMZN)
prop = {'size': 12, 'weight': 'bold'}

# Plotting the pie chart
plt.figure(figsize=(7, 7))
plt.pie(sizes, explode=explode, labels=lab, colors=colors, autopct='%1.1f%%',   shadow=True, startangle=140,textprops=prop)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Stock Distribution in Dataset',fontweight='bold', fontsize=14)
plt.show()


# Load the percentage_spam_tweets data from the file:
percentage_spam_tweets = [0.1996589, 0.83241379, 0.27646015, 0.4656, 0.2659]

plt.figure(figsize=(7, 5))

# Generate colors for each market
colors = ['r', 'b', 'm', 'grey', 'gold']
markets = ['AMZN', 'AAPL', 'MSFT', 'TSLA', 'GOOG']

for i, market in enumerate(markets):
    plt.barh(i + 1, percentage_spam_tweets[i] * 100, 0.5,
             color=colors[i], edgecolor='k', label=market)

plt.xlabel('Percentage of Spam Tweets', fontsize=16, fontweight='bold')
plt.yticks(np.arange(1, len(markets) + 1), markets, rotation=0, fontsize=12, fontweight='bold')
plt.xticks(fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()


final_df = pd.concat([final_df_AMZN, final_df_AAPL, final_df_MSFT, final_df_TSLA, final_df_GOOG], axis=0, ignore_index=True)
dataset1 = dataset_(final_df_AMZN)
dataset2 = dataset_(final_df_AAPL)
dataset3 = dataset_(final_df_MSFT)
dataset4 = dataset_(final_df_TSLA)
dataset5 = dataset_(final_df_GOOG)
dataset = pd.concat([dataset1, dataset2, dataset3, dataset4, dataset5])
y1, y2, y3, y4, Y5 = labels_(dataset1, dataset2, dataset3, dataset4, dataset5)
y = pd.concat([y1, y2, y3, y4,Y5])
y = numpy_.array(y)

y_label = (y.values == y.max(axis=1).values[:, None]).astype(int)

# Convert binary array to numeric values based on custom interpretation
y_val = np.array([0 if np.array_equal(row, [1, 0, 0]) else
                  1 if np.array_equal(row, [0, 1, 0]) else
                  2 for row in y_label])
X, y_scale_dataset = normalize_data(dataset, (0, 1), "Close")
X, y = numpy_.array_(X, y_val)
# --------------feature fusion by optimized Memory efficient vision transformer with Diffusion kernel attention network. ------------------

# Define a simple Diffusion Kernel Attention mechanism
class DiffusionKernelAttention(nn.Module):
    def __init__(self, dim):
        super(DiffusionKernelAttention, self).__init__()
        self.dim = dim
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.diffusion_kernel = nn.Parameter(torch.randn(dim, dim))

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attention = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / (self.dim ** 0.5), dim=-1)
        attention = torch.matmul(attention, self.diffusion_kernel)
        return torch.matmul(attention, v)


# Define a memory-efficient Vision Transformer
class MemoryEfficientViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12):
        super(MemoryEfficientViT, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))

        self.transformer = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads) for _ in range(depth)
        ])

        self.diffusion_kernel_attention = DiffusionKernelAttention(embed_dim)

        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B, _, _, _ = x.size()
        x = self.patch_embed(x).flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        for layer in self.transformer:
            x = layer(x)

        x = self.diffusion_kernel_attention(x)
        x = x[:, 0]
        x = self.fc(x)

        return x


# Define the feature fusion network
class FeatureFusionNetwork(nn.Module):
    def __init__(self, vit, num_classes=1000):
        super(FeatureFusionNetwork, self).__init__()
        self.vit = vit
        self.fc = nn.Linear(vit.fc.in_features, num_classes)

    def forward(self, x):
        features = self.vit(x)
        output = self.fc(features)
        return output


# Instantiate and test the model
vit = MemoryEfficientViT()
feature_fusion_network = FeatureFusionNetwork(vit)
X,y = features(feature_fusion_network,X,y)
X, y = numpy_.py_array(X, y)

print("\nTotal samples - ", X.shape[0])

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training samples - ", X_train.shape[0])
print("Testing samples - ", X_test.shape[0], '\n')

# =------------------Mequivariant quantum neural networks for  prediction------------------

# Define a custom quantum-inspired layer
def quantum_inspired_layer(x):
    # Example transformation, replace with actual quantum-inspired operations
    return tf.math.sin(x)

# Define the QNN class
class MequivariantQNNet(tf.keras.Model):
    def __init__(self):
        super(MequivariantQNNet, self).__init__()
        self.quantum_layer = Lambda(quantum_inspired_layer)
        self.dense = Dense(1, activation='sigmoid')  # Single class prediction

    def call(self, inputs):
        x = self.quantum_layer(inputs)
        return self.dense(x)

    def _build_model(X_train, n_class,  ylf):
        model = Sequential()
        model.add(Dense(256, input_shape=(X_train.shape[1],), activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(n_class, activation='softmax'))
        model.compile(optimizer='adam', loss='mse')
        return model

    def predict(self, X):
        return self.model.predict([X] * self.num_heads)


# ------------------  Tuning by Genghis Khan shark optimizer  ----------------------------
class GKSOptimizer:
    def __init__(self, n, lb, ub, dim, fhd, max_iter, *args):
        self.n = n
        self.lb = np.ones(dim) * lb
        self.ub = np.ones(dim) * ub
        self.dim = dim
        self.fhd = fhd
        self.max_iter = max_iter
        self.args = args
        self.pop_pos = None
        self.pop_fit = None
        self.best_score = np.inf
        self.best_pos = np.zeros(dim)
        self.curve = np.zeros(max_iter)
        self.time = 0

    def initialize(self):
        self.pop_pos = np.random.rand(self.n, self.dim) * (self.ub - self.lb) + self.lb
        self.pop_fit = np.array([self.fhd(self.pop_pos[i, :], *self.args) for i in range(self.n)])
        for i in range(self.n):
            if self.pop_fit[i] <= self.best_score:
                self.best_score = self.pop_fit[i]
                self.best_pos = self.pop_pos[i, :].copy()

    def optimize(self):
        start_time = time.time()
        self.initialize()
        h = [0.1]

        for it in range(self.max_iter):
            h.append(1 - 2 * (h[it] ** 4))
            p = 2 * (1 - (it / self.max_iter) ** (1 / 4)) + abs(h[it + 1]) * ((it / self.max_iter) ** (1 / 4) - (it / self.max_iter) ** 3)
            beta = 0.2 + (1.2 - 0.2) * (1 - (it / self.max_iter) ** 3) ** 2
            alpha = abs(beta * np.sin((3 * np.pi / 2 + np.sin(3 * np.pi / 2 * beta))))

            # Hunting stage
            for i in range(self.n):
                new_pop_pos = self.pop_pos[i, :] + (self.lb + np.random.rand() * (self.ub - self.lb)) / (it + 1)
                new_pop_pos = np.clip(new_pop_pos, self.lb, self.ub)
                new_pop_fit = self.fhd(new_pop_pos, *self.args)
                if new_pop_fit < self.pop_fit[i]:
                    self.pop_fit[i] = new_pop_fit
                    self.pop_pos[i, :] = new_pop_pos

            # Best Position Attraction Effect
            for i in range(self.n):
                s = 1.5 * (self.pop_fit[i] ** np.random.rand())
                s = np.real(s)
                if i == 1:
                    new_pop_pos = (self.best_pos - self.pop_pos[i, :]) * s
                else:
                    gks_pos = (self.best_pos - self.pop_pos[i, :]) * s
                    new_pop_pos = (gks_pos + new_pop_pos) / 2
                new_pop_pos = np.clip(new_pop_pos, self.lb, self.ub)
                new_pop_fit = self.fhd(new_pop_pos, *self.args)
                if new_pop_fit < self.pop_fit[i]:
                    self.pop_fit[i] = new_pop_fit
                    self.pop_pos[i, :] = new_pop_pos

            # Foraging stage
            for i in range(self.n):
                tf = (np.random.rand() > 0.5) * 2 - 1
                new_pop_pos = self.best_pos + np.random.rand(self.dim) * (self.best_pos - self.pop_pos[i, :]) + tf * p ** 2 * (self.best_pos - self.pop_pos[i, :])
                new_pop_pos = np.clip(new_pop_pos, self.lb, self.ub)
                new_pop_fit = self.fhd(new_pop_pos, *self.args)
                if new_pop_fit < self.pop_fit[i]:
                    self.pop_fit[i] = new_pop_fit
                    self.pop_pos[i, :] = new_pop_pos

            # Self-protection mechanism
            for i in range(self.n):
                a1 = np.random.randint(0, self.n, self.n)
                r1, r2 = a1[:2]
                if np.random.rand() < 0.5:
                    k = np.random.randint(self.n)
                    f1 = -1 + 2 * np.random.rand()
                    f2 = -1 + 2 * np.random.rand()
                    ro = alpha * (2 * np.random.rand() - 1)
                    xk = np.random.uniform(self.lb, self.ub, self.dim)
                    l1 = np.random.rand() < 0.5
                    u1 = l1 * 2 * np.random.rand() + (1 - l1) * 1
                    u2 = l1 * np.random.rand() + (1 - l1) * 1
                    u3 = l1 * np.random.rand() + (1 - l1) * 1
                    l2 = np.random.rand() < 0.5
                    xp = (1 - l2) * self.pop_pos[k, :] + l2 * xk
                    popi1 = self.lb + (self.ub - self.lb) * np.random.rand(self.dim)
                    popi2 = self.lb + (self.ub - self.lb) * np.random.rand(self.dim)

                    if u1 < 0.5:
                        new_pop_pos = new_pop_pos + f1 * (u1 * self.best_pos - u2 * xp) + f2 * ro * (u3 * (popi2 - popi1) + u2 * (self.pop_pos[r1, :] - self.pop_pos[r2, :])) / 2
                    else:
                        new_pop_pos = self.best_pos + f1 * (u1 * self.best_pos - u2 * xp) + f2 * ro * (u3 * (popi2 - popi1) + u2 * (self.pop_pos[r1, :] - self.pop_pos[r2, :])) / 2

                new_pop_pos = np.clip(new_pop_pos, self.lb, self.ub)
                new_pop_fit = self.fhd(new_pop_pos, *self.args)
                if new_pop_fit < self.pop_fit[i]:
                    self.pop_fit[i] = new_pop_fit
                    self.pop_pos[i, :] = new_pop_pos

            for i in range(self.n):
                if self.pop_fit[i] < self.best_score:
                    self.best_score = self.pop_fit[i]
                    self.best_pos = self.pop_pos[i, :].copy()

            self.curve[it] = self.best_score
            self.time = time.time() - start_time

        return self.best_pos, self.best_score, self.curve, self.time


# Parameters
n = 30
lb = -10
ub = 10
dim = 10
max_iter = 100

# Instantiate and run GKSO
gkso_optimizer = GKSOptimizer(n, lb, ub, dim, fobj, max_iter)


num_classes = 3  # Define the number of output classes
model = MequivariantQNNet._build_model(X_train, num_classes,  gkso_optimizer)
y_train_ = to_categorical(y_train)
ep = 20  # number of epochs
xx = list(range(1, ep + 1))

# Now, you can train the model using dataset
history = model.fit(X_train, y_train_, epochs=ep, batch_size=32, validation_split=0.2, verbose=1)

# ---------------------loss and accuracy curve---------------------

fig1, ax1 = plt.subplots(figsize=(8, 5))  # Create empty plot
ax1.set_facecolor('#FCFCFC')
plt.grid(color='w', linestyle='-.', linewidth=1)
plt.plot(xx, 1 - np.array(history.history['loss']), color='r')
plt.plot(xx, 1 - np.array(history.history['val_loss']), color='b')
plt.ylabel('Accuracy', fontsize=16, weight='bold')
plt.xlabel(' Epoch', fontsize=16, weight='bold')
plt.xticks(fontsize=16, weight='bold')
plt.xlim([0, ep + 1])
plt.yticks(fontsize=16, weight='bold')
prop = {'size': 16, 'weight': 'bold'}
plt.legend(['Training', 'Testing'], loc='lower right', fancybox=True, prop=prop)
plt.tight_layout()
plt.show()

fig2, ax2 = plt.subplots(figsize=(9, 5))  # Create empty plot
ax2.set_facecolor('#FCFCFC')
plt.grid(color='w', linestyle='-.', linewidth=1)
plt.plot(xx, history.history['loss'], color='r')
plt.plot(xx, history.history['val_loss'], color='b')
plt.ylabel('Loss', fontsize=16, weight='bold')
plt.xlabel('Epoch', fontsize=16, weight='bold')
prop = {'size': 16, 'weight': 'bold'}
plt.legend(['Training', 'Testing'], loc='upper right', fancybox=True, prop=prop)
plt.xticks(fontsize=16, weight='bold')
plt.yticks(fontsize=16, weight='bold')
plt.xlim([0, ep + 1])
plt.tight_layout()
plt.show()

pred, pred_prob = testing(model, X_test)  # Testing

mat = confusion_matrix(y_test, pred)  # confusion matrix

mse = mean_squared_error(y_test, pred)  # Mean Squared Error (MSE)

r2 = r2_score(y_test, pred) # R-squared (R2) score

# Area Under the Curve (AUC)
auc_scores = []
for class_index in range(pred_prob.shape[1]):
    class_true_labels = (y_test == class_index).astype(int)  # Treat current class as positive, others as negative
    auc_score = roc_auc_score(class_true_labels, pred_prob[:, class_index])
    auc_scores.append(auc_score)

# Compute the mean AUC across all classes
auc_score = np.mean(auc_scores)

accuracy = accuracy_score(y_test, pred) # Accuracy

f1s = f1_score(y_test, pred, average='weighted') # F1-score (F-measure)

rec = recall_score(y_test, pred, average='weighted') # Recall

pre = precision_score(y_test, pred, average='weighted') # Precision

rmse = sqrt(mean_squared_error(y_test, pred)) # Root Mean Squared Error (RMSE)

aae = mean_absolute_error(y_test, pred)# Average Absolute Error (AAE)

are = np.mean(np.abs(y_test - pred) / y_test) # Average Relative Error (ARE)

# Calculate confusion matrix for specificity
conf_matrix = confusion_matrix(y_test, pred, labels=[0, 1, 2])
plot_confusion_matrix(conf_matrix, ['Negative', 'Neutral', 'Positive'])

tn = np.diag(conf_matrix)  # true negatives
fp = conf_matrix.sum(axis=0) - tn  # false positives
fn = conf_matrix.sum(axis=1) - tn  # false negatives
tp = conf_matrix.sum() - (fp + fn + tn)  # true positives
spe = np.mean(tn / (tn + fp))
mae = mean_absolute_error(y_test, pred)
mape = np.mean(np.abs((y_test - pred) / y_test)) * 100

print("Accuracy                              :", accuracy)
print("Recall                                :", rec)
print("Precision                             :", pre)
print("Specificity                           :", spe)
print("F1-score (F-measure)                  :", f1s)
print("Mean Squared Error (MSE)              :", mse)
print("Mean Absolute Error (MAE)             :", mae)
print("Root Mean Squared Error (RMSE)        :", rmse)
print("Average Absolute Error (AAE)          :", aae)

# Define class labels
class_labels = ["Negative", "Neutral", "Positive"]
# Calculate overall accuracy
overall_accuracy = accuracy_score(y_test, pred)

# Calculate precision, recall, and f1-score for each class
precision = precision_score(y_test, pred, average=None, labels=[0, 1, 2])
recall = recall_score(y_test, pred, average=None, labels=[0, 1, 2])
f1 = f1_score(y_test, pred, average=None, labels=[0, 1, 2])

# Calculate class-wise accuracy
class_accuracies = []
for cls in np.unique(y_test):
    cls_mask = y_test == cls
    cls_accuracy = accuracy_score(y_test[cls_mask], pred[cls_mask])
    class_accuracies.append(cls_accuracy)

# Create a list to display the results
metrics = []
for i, label in enumerate(class_labels):
    metrics.append([label, class_accuracies[i], precision[i], recall[i], f1[i]])

# Add averages and overall accuracy
metrics.append(["Overall Accuracy", overall_accuracy, overall_accuracy, overall_accuracy, overall_accuracy])

# Display the results in a styled table with 3 decimal places
print(tabulate(metrics, headers=["Class", "Class Accuracy", "Precision", "Recall", "F1-Score"], floatfmt=".5f",
               tablefmt="grid"))


plt.figure(figsize=(15, 6))
plt.plot(final_df_AMZN['Close'].to_numpy(), color='r', label='AMZN - Actual')
plt.plot(final_df_AMZN['Open'].to_numpy(), '--', color='r', label='AMZN - Predicted')
plt.plot(final_df_AAPL['Close'].to_numpy(), color='b', label='AAPL - Actual')
plt.plot(final_df_AAPL['Open'].to_numpy(), '--', color='b', label='AAPL - Predicted')
plt.plot(final_df_MSFT['Close'].to_numpy(), color='m', label='MSFT - Actual')
plt.plot(final_df_MSFT['Open'].to_numpy(), '--', color='m', label='MSFT - Predicted')
plt.plot(final_df_TSLA['Close'].to_numpy(), color='k', label='TSLA - Actual')
plt.plot(final_df_TSLA['Open'].to_numpy(), '--', color='k', label='TSLA - Predicted')
plt.plot(final_df_GOOG['Close'].to_numpy(), color='gold', label='GOOG - Actual')
plt.plot(final_df_GOOG['Open'].to_numpy(), '--', color='gold', label='GOOG - Predicted')
plt.xlabel('Month/Year', fontsize=16, fontweight='bold')
plt.ylabel('Stock Prices ($)', fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.xticks(fontsize=16, fontweight='bold')
plt.xticks([0, 50, 100, 150, 200, 250], ['1/2022', '6/2022', '1/2023', '6/2023', '1/2024', '6/2024'],
           rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
prop = {'size': 11, 'weight': 'bold'}
x_ = range(3, dataset.shape[0])
x_ = list(dataset.index)
pred1 = predict(dataset1, 'AMZN')
pred2 = predict(dataset2, 'AAPL')
pred3 = predict(dataset3, 'MSFT')
pred4 = predict(dataset4, 'TSLA')
pred5 = predict(dataset5, 'GOOG')
plt.legend(prop=prop, bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()

future_prediction(final_df_AMZN,final_df_AAPL,final_df_MSFT,final_df_TSLA,final_df_GOOG,pred1,pred2,pred3,pred4,pred5)

import tkinter as tk


class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Stock Recommendation System")
        self.geometry("1050x600")
        self.configure(bg="grey")
        self.frames = {}
        for F in (HomePage, RegisterPage, InputPage, LoginPage, OutputPage):
            page_name = F.__name__
            frame = F(parent=self, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("HomePage")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()


class HomePage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.configure(bg="grey")

        label = ttk.Label(self, text="Stock Recommendation System", font=("Times", 25, "bold"),
                          background='grey')
        label.pack(padx=80, pady=50)
        style = ttk.Style()
        style.configure("TButton", background="black", foreground='black', font=("Times", 16))

        register_button = ttk.Button(self, text="Register", width=20,
                                     command=lambda: controller.show_frame("RegisterPage"), style="Custom.TButton")
        register_button.pack(pady=20)

        login_button = ttk.Button(self, text="Login", width=20, command=lambda: controller.show_frame("LoginPage"),
                                  style="Custom.TButton")
        login_button.pack(pady=20)


class RegisterPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.configure(bg="grey")
        style = ttk.Style()
        style.configure("TButton", background="black", foreground='blue', font=("Times", 16))

        label = ttk.Label(self, text="Register", background='grey', font=("Times", 20, 'bold'))
        label.pack(padx=200, pady=20)

        self.entries = {}
        fields = ["First name", "Last name", "Email", "Username", "Password", "Confirm Password"]
        for field in fields:
            style = ttk.Style()
            style.configure("Custom.TFrame", background="grey")

            frame = ttk.Frame(self, style="Custom.TFrame")

            frame.pack(fill="x", padx=50, pady=2)
            label = ttk.Label(frame, text=field, width=18, background='grey', font=("Times", 14))
            label.pack(side="left")
            style = ttk.Style()
            style.configure("Custom.TEntry", padding=(5, 5))  # (left, top, right, bottom) padding
            entry = ttk.Entry(frame, style="Custom.TEntry")
            entry.pack(side="left", fill="x", padx=1, pady=2, expand=True)
            self.entries[field] = entry

        register_button = ttk.Button(self, width=15, text="Register", command=self.register, style="custom.TButton")
        register_button.pack(pady=10)

        home_button = ttk.Button(self, width=20, text="Back to Home", command=lambda: controller.show_frame("HomePage"))
        home_button.pack(pady=10)

    def register(self):
        first_name = self.entries["First name"].get()
        last_name = self.entries["Last name"].get()
        email = self.entries["Email"].get()
        username = self.entries["Username"].get()
        password = self.entries["Password"].get()
        confirm_password = self.entries["Confirm Password"].get()

        if password != confirm_password:
            messagebox.showerror("Error", "Passwords do not match")
            return

        # Add your registration logic here
        # For now, we'll just show a success message
        messagebox.showinfo("Success", "Registration successful")


file_path = "Data/utilis/data.pkl"
with open(file_path, "rb") as file:
    data = pickle.load(file)

# Extract pv and data_values
pv = data["pv"]
data_values = data["data_values"]

# Plot the bar chart
cc = ['AMZN', 'AAPL', 'MSFT', 'TSLA', 'GOOG']
_, ax2 = plt.subplots(figsize=(9, 5))  # Create empty plot
ax2.set_facecolor('white')
plt.grid(color='grey', linestyle='', linewidth=1)
clr = ['r', 'b', 'm', 'grey', 'gold']
plt.bar(cc, pv, 0.35, color=clr, edgecolor='k')
plt.ylabel('Prediction Value', fontsize=16, fontweight='bold')
prop = {'size': 14, 'weight': 'bold'}
plt.xticks(fontsize=12, weight='bold')
plt.yticks(fontsize=16, weight='bold')
plt.tight_layout()
plt.show()

# Plot the box plot
labels = ['SVM', 'GAN-LSTM', 'KNN', 'FDM', 'AFHGN', 'OEquiGS-Quan-Net \n(proposed)']

plt.figure(figsize=(12, 6))
box = plt.boxplot(data_values, labels=labels, widths=0.2, patch_artist=True)

# Customize colors for each box
custom_colors = ['b', 'r', 'g', 'pink', 'cyan', 'purple']

for patch, color in zip(box['boxes'], custom_colors):
    patch.set_facecolor(color)

# Adding labels and title
plt.ylabel('Computational Complexity', fontsize=16, weight='bold')
plt.xticks(fontsize=14, weight='bold', rotation=0)
plt.yticks(fontsize=14, weight='bold')

# Tight layout and display the plot
plt.tight_layout()
plt.show()

# Load the times dictionary from the pickle file
with open('Data/utilis/times.pkl', 'rb') as f:
    times = pickle.load(f)

# Extract methods, mseing times, and processing times
methods = list(times.keys())
mse = [times[method]["mse"] for method in methods]
mae = [times[method]["mae"] for method in methods]
rmse = [times[method]["rmse"] for method in methods]
aae = [times[method]["aae"] for method in methods]

# Set bar width
bar_width = 0.1

# Set positions of the bars on the x-axis
r1 = np.arange(len(methods))
r2 = [x + bar_width + 0.05 for x in r1]
r3 = [x + bar_width + 0.05 for x in r2]
r4 = [x + bar_width + 0.05 for x in r3]

# Create the bar graph
plt.figure(figsize=(12, 6))
bars1 = plt.bar(r1, mse, color='m', width=bar_width, edgecolor='k', label='MSE')
bars2 = plt.bar(r2, mae, color='gold', width=bar_width, edgecolor='k', label='MAE')
bars3 = plt.bar(r3, rmse, color='green', width=bar_width, edgecolor='k', label='RMSE')
bars4 = plt.bar(r4, aae, color='lightblue', width=bar_width, edgecolor='k', label='AAE')

prop = {'size': 12, 'weight': 'bold'}

# General layout
plt.xticks([0.115, 1.115, 2.115, 3.115, 4.115, 5.115],
           ['SVM', 'GAN-LSTM', 'KNN', 'FDM', 'AFHGN', 'OEquiGS-Quan-Net \n(proposed)'])
plt.xlabel('', fontweight='bold', fontsize=16)
plt.ylabel('Time (ms)', fontweight='bold', fontsize=16)
plt.xticks(rotation=0, fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.legend(prop=prop)
plt.tight_layout()
plt.show()


# Load the data from the pickle file
with open('Data/utilis/data_.pkl', 'rb') as file:
    data = pickle.load(file)

# Define function to compute error rates from accuracy values
def compute_error_rates(accuracy_values):
    error_rates = {}
    for method, accuracies in accuracy_values.items():
        error_rates[method] = [1 - acc for acc in accuracies]
    return error_rates


# Define plotting function
def plot_metrics(data, ylabel, title):
    df = pd.DataFrame(data, columns=["Method", "Value"])
    custom_colors = ['b', 'r', 'g', 'pink', 'cyan', 'purple']
    palette = {method: color for method, color in zip(df['Method'].unique(), custom_colors)}

    plt.figure(figsize=(12, 6))
    sns.boxplot(x="Method", y="Value", data=df, palette=palette)
    plt.xlabel('', fontweight='bold', fontsize=16)
    plt.ylabel(ylabel, fontweight='bold', fontsize=16)
    plt.xticks(rotation=0, fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    plt.grid(color='pink', linestyle='--', linewidth=0.5)
    # plt.title(title, fontweight='bold', fontsize=18)
    plt.show()


# Compute error rates
error_rates = compute_error_rates(data["accuracy_values"])

# Prepare data for plotting
metrics = {
    "Accuracy": data["accuracy_values"],
    "Precision": data["precision_values"],
    "Recall": data["recall_values"],
    "F1-Score": data["f1_scores"],
    "Error Rate": error_rates
}

for metric, values in metrics.items():
    data_list = []
    if metric == "Error Rate":
        # Convert accuracy values to error rates for plotting
        for method, vals in values.items():
            for val in vals:
                data_list.append([method, val])
    else:
        # Prepare data for other metrics
        for method, vals in values.items():
            for val in vals:
                data_list.append([method, val])

    ylabel = metric if metric != "Error Rate" else "Error Rate"
    plot_metrics(data_list, ylabel, f'{metric} Comparison')



def min_value_and_index(lst):
    if not lst:
        return None, None  # Handle the case where the list is empty

    min_value = min(lst)  # Find the minimum value in the list
    min_index = lst.index(min_value)  # Find the index of the minimum value

    return min_value, min_index

# Load the times dictionary
with open('Data/utilis/times_.pkl', 'rb') as file:
    times = pickle.load(file)

# Extract methods, training times, and processing times
methods = list(times.keys())
train_times = [times[method]["train"] for method in methods]
process_times = [times[method]["process"] for method in methods]

# Set bar width
bar_width = 0.18

# Set positions of the bars on the x-axis
r1 = np.arange(len(methods))
r2 = [x + bar_width + 0.05 for x in r1]

# Create the bar graph
plt.figure(figsize=(12, 6))
bars1 = plt.bar(r1, train_times, color='cyan', width=bar_width, edgecolor='k', label='Training Time')
bars2 = plt.bar(r2, process_times, color='g', width=bar_width, edgecolor='k', label='Processing Time')


# General layout
plt.xticks([r + bar_width / 2 for r in range(len(methods))],
           ['SVM', 'GAN-LSTM', 'KNN', 'FDM', 'AFHGN', 'OEquiGS-Quan-Net \n(proposed)'])
plt.xlabel('Methods', fontweight='bold', fontsize=16)
plt.ylabel('Time (ms)', fontweight='bold', fontsize=16)
plt.xticks(rotation=0, fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.legend(prop=prop)
plt.tight_layout()
plt.show()

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from prettytable import PrettyTable

# Load the data dictionary
with open('Data/utilis/utilr.pkl', 'rb') as file:
    util_data = pickle.load(file)

training_percentages = util_data["training_percentages"]
r2_values = util_data["r2_values"]

# Plot the results
plt.figure(figsize=(10, 6))  # Create plot
plt.grid(color='pink', linestyle='--', linewidth=0.5)  # Set grid properties

markers = ['o', 's', 'D', '^', 'v', '*']  # Markers for different methods
colors = ['b', 'g', 'r', 'c', 'm', 'y']  # Different colors for each method

for (name, values), marker, color in zip(r2_values.items(), markers, colors):
    # Interpolate the data points
    interp_func = interp1d(training_percentages, values, kind='cubic')
    smooth_x = np.linspace(min(training_percentages), max(training_percentages), 10)
    smooth_y = interp_func(smooth_x)

    plt.plot(smooth_x, smooth_y, label=name, color=color)
    plt.plot(training_percentages, values, 'o', color=color)

# Font properties for legend
prop = {'size': 12, 'weight': 'bold'}

plt.xlabel('Training Percentage', fontweight='bold', fontsize=14)
plt.ylabel('R-squared Value', fontweight='bold', fontsize=14)
plt.legend(prop=prop)
plt.xticks([50, 60, 70, 80, 90], [50, 60, 70, 80, 90], rotation=0)
plt.ylim([0, 1])  # Set y-axis limits to range from 0 to 1

# Set font properties for ticks
plt.xticks(fontweight='bold', fontsize=14)
plt.yticks(fontweight='bold', fontsize=14)

plt.show()

# Create PrettyTable
table = PrettyTable()

# Add columns
table.add_column("Training Percentage", training_percentages)

for method, values in r2_values.items():
    table.add_column(method, values)

# Customize table appearance
table.title = "R-squared Values at Different Training Percentages"
table.field_names = ["Training Percentage"] + list(r2_values.keys())

# Print the table
print(table)

class InputPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.recommended_company_data = None
        self.controller = controller
        self.configure(bg="grey")
        style = ttk.Style()
        style.configure("TButton", background="black", foreground='blue', font=("Times", 16))

        label = ttk.Label(self, text="Companies for stock recommendations", background='grey',
                          font=("Times", 20, 'bold'))
        label.pack(padx=20, pady=20)
        style = ttk.Style()
        style.configure("Custom.TFrame", background="grey")

        frame = ttk.Frame(self, style="Custom.TFrame")
        frame.pack(fill="x", padx=50, pady=5)
        # Insert sample data
        data = [
            (1, "Amazon", random.randint(100, 10000), random.randint(100, 5000), random.randint(5000000, 100000000)),
            (2, "Apple", random.randint(100, 10000), random.randint(100, 5000), random.randint(5000000, 100000000)),
            (3, "Microsoft", random.randint(100, 10000), random.randint(100, 5000), random.randint(5000000, 100000000)),
            (4, "Tesla", random.randint(100, 10000), random.randint(4000, 5000), random.randint(5000000, 100000000)),
            (5, "Google", random.randint(100, 10000), random.randint(100, 5000), random.randint(5000000, 100000000))
        ]
        self.create_table(data)

        price_ = [data[0][3], data[1][3], data[2][3], data[3][3], data[4][3]]
        company = [data[0][1], data[1][1], data[2][1], data[3][1], data[4][1]]
        min_val, min_idx = min_value_and_index(price_)

        label = ttk.Label(self, text='Recommended Company : '+company[min_idx], background='grey',
                          font=("Times", 20, 'bold'))
        label.pack(padx=20, pady=5)

        # Randomly choose between 4 and 5 stars
        num_stars = random.choice([4, 5])
        # Create a string with the chosen number of stars
        rt = "★" * num_stars
        label = ttk.Label(self, text=rt, background='grey',
                          font=("Times", 20, 'bold'))
        label.pack(padx=20, pady=5)
        home_button = ttk.Button(self, width=20, text="Back to Home", command=lambda: controller.show_frame("HomePage"))
        home_button.pack(pady=10)
        del data

    def create_table(self,data):
        columns = ("Company ID", "Company Name", "Market Index", "Price", "Daily Turnover")
        # Create a style for the Treeview
        style = ttk.Style()
        style.configure("Custom.Treeview", font=("Times New Roman", 12))  # Cell text
        style.configure("Custom.Treeview.Heading", font=("Times New Roman", 14, "bold"))  # Header text

        self.tree = ttk.Treeview(self, columns=columns, show="headings", style="Custom.Treeview")
        self.tree.heading("Company ID", text="Company ID")
        self.tree.heading("Company Name", text="Company Name")
        self.tree.heading("Market Index", text="Market Index")
        self.tree.heading("Price", text="Price")
        self.tree.heading("Daily Turnover", text="Daily Turnover")

        self.tree.pack(padx=20, pady=20)



        for item in data:
            self.tree.insert('', 'end', values=item)

class LoginPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.configure(bg="grey")
        style = ttk.Style()
        style.configure("TButton", background="black", foreground='blue', font=("Times", 16))

        label = ttk.Label(self, text="Login", background='grey', font=("Times", 20, 'bold'))
        label.pack(padx=200, pady=20)

        self.entries = {}
        fields = ["Username", "Password"]
        for field in fields:
            style = ttk.Style()
            style.configure("Custom.TFrame", background="grey")

            frame = ttk.Frame(self, style="Custom.TFrame")
            frame.pack(fill="x", padx=50, pady=5)
            label = ttk.Label(frame, text=field, background='grey', width=15, font=("Times", 12))
            label.pack(side="left")
            style = ttk.Style()
            style.configure("Custom.TEntry", padding=(5, 5))  # (left, top, right, bottom) padding
            entry = ttk.Entry(frame, width=5, show="*" if field == "Password" else None, style="Custom.TEntry")
            entry.pack(side="left", fill="x", padx=1, pady=5, expand=True)
            self.entries[field] = entry

        login_button = ttk.Button(self, text="Login", command=self.login)
        login_button.pack(pady=10)

        home_button = ttk.Button(self, text="Back to Home", command=lambda: controller.show_frame("HomePage"))
        home_button.pack(pady=10)

    def login(self):
        username = self.entries["Username"].get()
        password = self.entries["Password"].get()

        if username == "admin" and password == "admin@123":

            self.controller.show_frame("InputPage")
        else:
            messagebox.showerror("Error", "Invalid username or password")


class OutputPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.configure(bg="grey")

        label = ttk.Label(self, text="Recommendation", background='grey', font=("Times", 18))
        label.pack(pady=20)
        style = ttk.Style()
        MequivariantQNNet()
        style.configure("TButton", background="grey", font=("Times", 18))

        home_button = ttk.Button(self, text="Back to Home", width=20, command=lambda: controller.show_frame("HomePage"),
                                 style="Custom.TButton")
        home_button.pack(padx=50, pady=20)


if __name__ == "__main__":
    app = Application()
    app.mainloop()
