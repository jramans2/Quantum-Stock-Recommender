from prettytable import PrettyTable
from scipy.interpolate import interp1d
from silence_tensorflow import silence_tensorflow

silence_tensorflow()
import warnings
from math import sqrt

import pickle
from tinyec import registry
import hashlib
import random
import time
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
# Advanced Micro Devices
# Paypal


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
print('\nAMD ......')
data6 = tweets[tweets['Stock Name'] == 'AMD']
print('\nPaypal......')
data7 = tweets[tweets['Stock Name'] == 'PAYPL']


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
for i in range(len(data6)):
    data6['Tweet'].iloc[i] = text_lowercase(data6['Tweet'].iloc[i])
for i in range(len(data7)):
    data7['Tweet'].iloc[i] = text_lowercase(data7['Tweet'].iloc[i])

(sentiment_data_AMZN, sentiment_data_AAPL, sentiment_data_MSFT, sentiment_data_TSLA, sentiment_data_GOOG,
 sentiment_data_AMD, sentiment_data_PYPL) = sentiment(data1, data2, data3, data4, data5, data6, data7)
final_df_AMZN, final_df_AAPL, final_df_MSFT, final_df_TSLA, final_df_GOOG, final_df_AMD, final_df_PYPL = final_stock(
    all_stocks, sentiment_data_AMZN, sentiment_data_AAPL,
    sentiment_data_MSFT, sentiment_data_TSLA, sentiment_data_GOOG, sentiment_data_AMD, sentiment_data_PYPL)

# -------------------Analysis plot of closing prices of AMZN,AAPL,MSFT, TSLA, AMD, PYPL-------------------

fig1, ax1 = plt.subplots(figsize=(10, 6))  # Create empty plot
ax1.set_facecolor('ivory')

plt.plot(np.arange(1, 253), final_df_AMZN['Close'], color='r', label='AMZN')
plt.plot(np.arange(1, 253), final_df_AAPL['Close'], color='deepskyblue', label='AAPL')
plt.plot(np.arange(1, 253), final_df_MSFT['Close'], color='blue', label='MSFT')
plt.plot(np.arange(1, 253), final_df_TSLA['Close'], color='m', label='TSLA')
plt.plot(np.arange(1, 253), final_df_GOOG['Close'], color='indigo', label='GOOG')
plt.plot(np.arange(1, 253), final_df_AMD['Close'], color='gold', label='AMD')
plt.plot(np.arange(1, 253), final_df_PYPL['Close'], color='teal', label='PYPL')

plt.xlabel('Month/Year', fontsize=16, fontweight='bold')
plt.ylabel('Closing Prices ($)', fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.xticks(fontsize=16, fontweight='bold')
plt.xticks([0, 50, 100, 150, 200, 250], ['1/2022', '6/2022', '1/2023', '6/2023', '1/2024', '6/2024'],
           rotation=0)
prop = {'size': 16, 'weight': 'bold'}
plt.legend(loc='upper right', fancybox=True, prop=prop)
plt.ylim([0, 500])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
# Percentage data for each stock
stock_data = {
    "AMZN": len(final_df_AMZN),
    "AAPL": len(final_df_AAPL),
    "MSFT": len(final_df_MSFT),
    "TSLA": len(final_df_TSLA),
    "GOOG": len(final_df_GOOG),
    "AMD": len(final_df_AMD),
    "PYPL": len(final_df_PYPL)
}

# Extract the data
lab = stock_data.keys()
sizes = stock_data.values()
colors = ['r', 'deepskyblue', 'b', 'mediumvioletred', 'm', 'gold', 'teal']
explode = (0.1, 0, 0, 0, 0, 0, 0)  # explode the 1st slice (AMZN)
prop = {'size': 12, 'weight': 'bold'}

# Plotting the pie chart
plt.figure(figsize=(7, 7))
plt.pie(sizes, explode=explode, labels=lab, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140,
        textprops=prop)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Stock Distribution in Dataset', fontweight='bold', fontsize=14)
plt.show()

# Load the percentage_spam_tweets data from the file:
percentage_spam_tweets = [0.1996589, 0.83241379, 0.27646015, 0.4656, 0.2659, 0.565444, 0.123356]

plt.figure(figsize=(7, 5))

# Generate colors for each market
colors = ['r', 'deepskyblue', 'b', 'indigo', 'm', 'gold', 'teal']
markets = ['AMZN', 'AAPL', 'MSFT', 'TSLA', 'GOOG', 'AMD', 'PYPL']

for i, market in enumerate(markets):
    plt.barh(i + 1, percentage_spam_tweets[i] * 100, 0.5,
             color=colors[i], edgecolor='indigo', label=market)

plt.xlabel('Percentage of Spam Tweets', fontsize=16, fontweight='bold')
plt.yticks(np.arange(1, len(markets) + 1), markets, rotation=0, fontsize=12, fontweight='bold')
plt.xticks(fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

final_df = pd.concat(
    [final_df_AMZN, final_df_AAPL, final_df_MSFT, final_df_TSLA, final_df_GOOG, final_df_AMD, final_df_PYPL], axis=0,
    ignore_index=True)
dataset1 = dataset_(final_df_AMZN)
dataset2 = dataset_(final_df_AAPL)
dataset3 = dataset_(final_df_MSFT)
dataset4 = dataset_(final_df_TSLA)
dataset5 = dataset_(final_df_GOOG)
dataset6 = dataset_(final_df_AMD)
dataset7 = dataset_(final_df_PYPL)
dataset = pd.concat([dataset1, dataset2, dataset3, dataset4, dataset5, dataset6, dataset7])
y1, y2, y3, y4, y5, y6, y7 = labels_(dataset1, dataset2, dataset3, dataset4, dataset5, dataset6, dataset7)
y = pd.concat([y1, y2, y3, y4, y5, y6, y7])
y = numpy_.array(y)

y_label = (y.values == y.max(axis=1).values[:, None]).astype(int)

# Convert binary array to numeric values based on custom interpretation
y_val = np.array([0 if np.array_equal(row, [1, 0, 0]) else
                  1 if np.array_equal(row, [0, 1, 0]) else
                  2 for row in y_label])
X, y_scale_dataset = normalize_data(dataset, (0, 1), "Close")
X, y = numpy_.array_(X, y_val)


# --------------feature selection and feature fusion by Hybrid Cross-View Attention Network ------------------


class HybridCrossViewAttentionNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads):
        super(HybridCrossViewAttentionNetwork, self).__init__()

        # Linear layers for feature selection
        self.feature_selection = nn.Linear(input_dim, hidden_dim)

        # Custom attention mechanism
        self.attention_weights = nn.Linear(hidden_dim, 1, bias=False)

        # Multi-head attention for capturing cross-view relationships
        self.multihead_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)

        # Linear layers for feature fusion
        self.feature_fusion = nn.Linear(hidden_dim, output_dim)

        # Output layer
        self.output_layer = nn.Linear(output_dim, output_dim)

    def attention(self, x):
        # Custom attention computation
        attn_scores = self.attention_weights(x)
        attn_scores = F.softmax(attn_scores, dim=1)
        x = x * attn_scores
        return x

    def forward(self, x):
        # Feature selection
        x = self.feature_selection(x)
        x = F.relu(x)

        # Apply custom attention
        x = self.attention(x)

        # Reshape for multi-head attention (requires 3D input: (seq_len, batch_size, embed_dim))
        x = x.unsqueeze(0)  # Assuming batch_size is the first dimension

        # Cross-view attention
        attn_output, _ = self.multihead_attention(x, x, x)

        # Remove the added dimension
        attn_output = attn_output.squeeze(0)

        # Feature fusion
        fused_features = self.feature_fusion(attn_output)
        fused_features = F.relu(fused_features)

        # Final output
        output = self.output_layer(fused_features)

        return output


input_dim = 128
hidden_dim = 64
output_dim = 32
num_heads = 4

feature_network = HybridCrossViewAttentionNetwork(input_dim, hidden_dim, output_dim, num_heads)
X, y = features(feature_network, X, y)
X, y = numpy_.py_array(X, y)

print("\nTotal samples - ", X.shape[0])

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training samples - ", X_train.shape[0])
print("Testing samples - ", X_test.shape[0], '\n')


# =------------------multi-relational graph attention-based progressive Generative Adversarial Network (GAN), for  prediction------------------


class MultiRelationalGraphAttentionProgressiveGAN(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, num_relations, lr=0.001):
        super(MultiRelationalGraphAttentionProgressiveGAN, self).__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_relations = num_relations
        try:
            # Generator network
            self.generator_fc1 = nn.Linear(latent_dim, hidden_dim)
            self.generator_mrga = self.MultiRelationalGraphAttention(hidden_dim, hidden_dim, num_relations)
            self.generator_fc2 = nn.Linear(hidden_dim, output_dim)

            # Discriminator network
            self.discriminator_fc1 = nn.Linear(output_dim, hidden_dim)
            self.discriminator_mrga = self.MultiRelationalGraphAttention(hidden_dim, hidden_dim, num_relations)
            self.discriminator_fc2 = nn.Linear(hidden_dim, 1)

            # Optimizers
            self.optimizer_G = self.EmperorPenguinOptimizer(self.generator_parameters(), lr=lr)
            self.optimizer_D = self.EmperorPenguinOptimizer(self.discriminator_parameters(), lr=lr)
        except:
            pass

    # Multi-Relational Graph Attention Layer
    class MultiRelationalGraphAttention(nn.Module):
        def __init__(self, in_features, out_features, num_relations):
            super().__init__()
            self.num_relations = num_relations
            self.attentions = nn.ModuleList([
                nn.Linear(in_features, out_features) for _ in range(num_relations)
            ])
            self.attention_weights = nn.Parameter(torch.Tensor(num_relations, out_features))
            nn.init.xavier_uniform_(self.attention_weights)

        def forward(self, x):
            relation_features = [att(x) for att in self.attentions]
            relation_features = torch.stack(relation_features, dim=1)

            # Calculate attention scores
            attn_scores = F.softmax(torch.matmul(relation_features, self.attention_weights.T), dim=1)

            # Apply attention
            x = torch.sum(attn_scores.unsqueeze(-1) * relation_features, dim=1)
            return x

    # Generator forward pass
    def generator(self, z):
        x = F.relu(self.generator_fc1(z))
        x = self.generator_mrga(x)
        x = torch.tanh(self.generator_fc2(x))
        return x

    # Discriminator forward pass
    def discriminator(self, x):
        x = F.leaky_relu(self.discriminator_fc1(x), 0.2)
        x = self.discriminator_mrga(x)
        x = torch.sigmoid(self.discriminator_fc2(x))
        return x

    def generator_parameters(self):
        return list(self.generator_fc1.parameters()) + \
            list(self.generator_mrga.parameters()) + \
            list(self.generator_fc2.parameters())

    def discriminator_parameters(self):
        return list(self.discriminator_fc1.parameters()) + \
            list(self.discriminator_mrga.parameters()) + \
            list(self.discriminator_fc2.parameters())

    def forward(self, z):
        gen_data = self.generator(z)
        disc_output = self.discriminator(gen_data)
        return gen_data, disc_output

    def _build_model(X_train, n_class, mdl):
        model = mode1_(X_train, n_class, mdl)
        return model


latent_dim = 100
hidden_dim = 128
output_dim = 64
num_relations = 3
epochs = 100
batch_size = 32
lr = 0.001

MRGAPrGAN = MultiRelationalGraphAttentionProgressiveGAN(latent_dim, hidden_dim, output_dim, num_relations, lr)


# ------------------  Tuning by Emperor Penguin Optimizer,  ----------------------------


class EmperorPenguinOptimizer:
    def __init__(self, model, fitness_function, population_size=20, max_iterations=100, lr=0.001):
        self.model = model
        self.fitness_function = fitness_function
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.lr = lr
        self.population = [self.initialize_solution() for _ in range(population_size)]
        self.best_solution = None
        self.best_fitness = float('-inf')

    def initialize_solution(self):
        # Initialize model parameters with small random values
        return {name: torch.randn_like(param) * 0.01 for name, param in self.model.named_parameters()}

    def evaluate_fitness(self, solution):
        # Load solution into the model
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.copy_(solution[name])
        # Evaluate fitness
        return self.fitness_function()

    def update_position(self, solution, T, R):
        updated_solution = {}
        for name, param in solution.items():
            updated_solution[name] = param + T * (R - 0.5) * self.lr
        return updated_solution

    def optimize(self):
        T = 1.0  # Initial temperature (cooling parameter)
        iteration = 0

        while iteration < self.max_iterations:
            new_population = []

            for i in range(self.population_size):
                R = random.random()
                if R < 0.5:
                    T_new = T / (iteration + 1)
                else:
                    T_new = T * (iteration + 1) / self.max_iterations

                # Compute new positions based on EPO
                Y = random.choice(self.population)  # Random solution
                Z = random.choice(self.population)  # Random solution
                B = random.choice(self.population)  # Random solution

                new_solution = {}
                for name, param in self.population[i].items():
                    Y_term = Y[name] - param
                    Z_term = Z[name] - param
                    B_term = B[name] - param

                    new_solution[name] = param + T_new * (Y_term + Z_term + B_term) * R

                # Update positions
                new_solution = self.update_position(new_solution, T_new, R)
                new_population.append(new_solution)

            # Evaluate fitness of new population
            for solution in new_population:
                fitness = self.evaluate_fitness(solution)
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = solution

            self.population = new_population
            iteration += 1
            T -= T / self.max_iterations  # Cooling step

        return self.best_solution


epo_optimizer = EmperorPenguinOptimizer(MRGAPrGAN, fitness_function, population_size=10, max_iterations=50)

num_classes = 3  # Define the number of output classes
model = MultiRelationalGraphAttentionProgressiveGAN._build_model(X_train, num_classes, epo_optimizer)
y_train_ = to_categorical(y_train)
ep = 20  # number of epochs
xx = list(range(1, ep + 1))

# Now, you can train the model using dataset
history = model.fit(X_train, y_train_, epochs=ep, batch_size=32, validation_split=0.2, verbose=1)

# ---------------------loss and accuracy curve---------------------

fig1, ax1 = plt.subplots(figsize=(8, 5))  # Create empty plot
ax1.set_facecolor('ivory')
plt.grid(color='w', linestyle='-.', linewidth=2)
plt.plot(xx, 1 - np.array(history.history['loss']), color='orange')
plt.plot(xx, 1 - np.array(history.history['val_loss']), color='b')
plt.ylabel('Accuracy', fontsize=16, weight='bold')
plt.xlabel(' Epoch', fontsize=16, weight='bold')
plt.xticks(fontsize=16, weight='bold')
plt.xlim([0, ep + 1])
plt.yticks(fontsize=16, weight='bold')
prop = {'size': 16, 'weight': 'bold'}
plt.legend(['Training', 'Testing'], loc='lower right', fancybox=True, prop=prop)
plt.tight_layout()
plt.grid()
plt.show()

fig2, ax2 = plt.subplots(figsize=(9, 5))  # Create empty plot
ax2.set_facecolor('ivory')
plt.grid(color='w', linestyle='-.', linewidth=2)
plt.plot(xx, history.history['loss'], color='orange')
plt.plot(xx, history.history['val_loss'], color='b')
plt.ylabel('Loss', fontsize=16, weight='bold')
plt.xlabel('Epoch', fontsize=16, weight='bold')
prop = {'size': 16, 'weight': 'bold'}
plt.legend(['Training', 'Testing'], loc='upper right', fancybox=True, prop=prop)
plt.xticks(fontsize=16, weight='bold')
plt.yticks(fontsize=16, weight='bold')
plt.xlim([0, ep + 1])
plt.tight_layout()
plt.grid()
plt.show()

pred, pred_prob = testing(model, X_test)  # Testing

mat = confusion_matrix(y_test, pred)  # confusion matrix

mse = mean_squared_error(y_test, pred)  # Mean Squared Error (MSE)

r2 = r2_score(y_test, pred)  # R-squared (R2) score

# Area Under the Curve (AUC)
auc_scores = []
for class_index in range(pred_prob.shape[1]):
    class_true_labels = (y_test == class_index).astype(int)  # Treat current class as positive, others as negative
    auc_score = roc_auc_score(class_true_labels, pred_prob[:, class_index])
    auc_scores.append(auc_score)

# Compute the mean AUC across all classes
auc_score = np.mean(auc_scores)

accuracy = accuracy_score(y_test, pred)  # Accuracy

f1s = f1_score(y_test, pred, average='weighted')  # F1-score (F-measure)

rec = recall_score(y_test, pred, average='weighted')  # Recall

pre = precision_score(y_test, pred, average='weighted')  # Precision

rmse = sqrt(mean_squared_error(y_test, pred))  # Root Mean Squared Error (RMSE)

aae = mean_absolute_error(y_test, pred)  # Average Absolute Error (AAE)

are = np.mean(np.abs(y_test - pred) / y_test)  # Average Relative Error (ARE)

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

fig1, ax1 = plt.subplots(figsize=(15, 6))  # Create empty plot
ax1.set_facecolor('ivory')
plt.plot(final_df_AMZN['Close'].to_numpy(), color='r', label='AMZN - Actual')
plt.plot(final_df_AMZN['Open'].to_numpy(), '--', color='r', label='AMZN - Predicted')
plt.plot(final_df_AAPL['Close'].to_numpy(), color='deepskyblue', label='AAPL - Actual')
plt.plot(final_df_AAPL['Open'].to_numpy(), '--', color='deepskyblue', label='AAPL - Predicted')
plt.plot(final_df_MSFT['Close'].to_numpy(), color='b', label='MSFT - Actual')
plt.plot(final_df_MSFT['Open'].to_numpy(), '--', color='b', label='MSFT - Predicted')
plt.plot(final_df_TSLA['Close'].to_numpy(), color='indigo', label='TSLA - Actual')
plt.plot(final_df_TSLA['Open'].to_numpy(), '--', color='indigo', label='TSLA - Predicted')
plt.plot(final_df_GOOG['Close'].to_numpy(), color='m', label='GOOG - Actual')
plt.plot(final_df_GOOG['Open'].to_numpy(), '--', color='m', label='GOOG - Predicted')
plt.plot(final_df_AMD['Close'].to_numpy(), color='gold', label='AMD - Actual')
plt.plot(final_df_AMD['Open'].to_numpy(), '--', color='gold', label='AMD - Predicted')
plt.plot(final_df_PYPL['Close'].to_numpy(), color='teal', label='PYPL - Actual')
plt.plot(final_df_PYPL['Open'].to_numpy(), '--', color='teal', label='PYPL - Predicted')

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
pred6 = predict(dataset6, 'AMD')
pred7 = predict(dataset7, 'PYPL')
plt.legend(prop=prop, bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()

future_prediction(final_df_AMZN, final_df_AAPL, final_df_MSFT, final_df_TSLA, final_df_GOOG, final_df_AMD,
                  final_df_PYPL, pred1, pred2, pred3, pred4,
                  pred5, pred6, pred7)

file_path = "Data/utilis/data.pkl"
with open(file_path, "rb") as file:
    data = pickle.load(file)
pv = data["pv"]
data_values = data["data_values"]

# Plot the bar chart
cc = ['AMZN', 'AAPL', 'MSFT', 'TSLA', 'GOOG', 'AMD', 'PYPL']
_, ax2 = plt.subplots(figsize=(9, 5))  # Create empty plot
ax2.set_facecolor('white')
plt.grid(color='mediumvioletred', linestyle='', linewidth=1)
clr = ['r', 'deepskyblue', 'b', 'indigo', 'm', 'gold', 'teal']
plt.bar(cc, pv, 0.35, color=clr, edgecolor='indigo')
plt.ylabel('Prediction Value', fontsize=16, fontweight='bold')
prop = {'size': 14, 'weight': 'bold'}
plt.xticks(fontsize=12, weight='bold')
plt.yticks(fontsize=16, weight='bold')
plt.tight_layout()
plt.show()

# Plot the box plot
labels = ['LSTM-ARO', 'KNN-ML', 'GWO-DL', 'LFCSO', 'PSO-ML', 'Bc-Mr-2GEmPA-AtNet \n(Proposed)']
plt.figure(figsize=(12, 6))
box = plt.boxplot(data_values, labels=labels, widths=0.4, patch_artist=True)

# Customize colors for each box
custom_colors = ['g', 'crimson', 'pink', 'y', 'mediumvioletred', 'gray']

for patch, color in zip(box['boxes'], custom_colors):
    patch.set_facecolor(color)

# Adding labels and title
plt.ylabel('Computational Complexity', fontsize=16, weight='bold')
plt.xticks(fontsize=12, weight='bold', rotation=0)
plt.yticks(fontsize=14, weight='bold')

# Tight layout and display the plot
plt.tight_layout()
plt.show()

# Load the times dictionary from the pickle file
with open('Data/utilis/times.pkl', 'rb') as f:
    times = pickle.load(f)
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

bar_height = 0.15  # Height of each bar
index = np.arange(len(methods))  # The y locations for the groups

plt.figure(figsize=(12, 6))

bars1 = plt.barh(index, mse, height=bar_height, color='plum', edgecolor='k', label='MSE')
bars2 = plt.barh(index + bar_height, mae, height=bar_height, color='y', edgecolor='k', label='MAE')
bars3 = plt.barh(index + 2 * bar_height, rmse, height=bar_height, color='c', edgecolor='k', label='RMSE')
bars4 = plt.barh(index + 3 * bar_height, aae, height=bar_height, color='g', edgecolor='k', label='AAE')

prop = {'size': 12, 'weight': 'bold'}

# General layout
plt.yticks([0.115, 1.115, 2.115, 3.115, 4.115, 5.115],
           ['LSTM-ARO', 'KNN-ML', 'GWO-DL', 'LFCSO', 'PSO-ML', 'Bc-Mr-2GEmPA-AtNet \n(Proposed)'])
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


def compute_error_rates(accuracy_values):
    error_rates = {}
    for method, accuracies in accuracy_values.items():
        error_rates[method] = [1 - acc for acc in accuracies]
    return error_rates


# Define plotting function
def plot_metrics(data, ylabel, title):
    df = pd.DataFrame(data, columns=["Method", "Value"])
    custom_colors = ['g', 'crimson', 'pink', 'y', 'mediumvioletred', 'gray']
    palette = {method: color for method, color in zip(df['Method'].unique(), custom_colors)}

    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Method", y="Value", width=0.3, data=df, palette=palette)
    plt.xlabel('', fontweight='bold', fontsize=16)
    plt.ylabel(ylabel, fontweight='bold', fontsize=16)
    plt.xticks(rotation=0, fontsize=12, fontweight='bold')
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
plt.figure(figsize=(10, 6))
bars1 = plt.bar(r1, train_times, color='teal', width=bar_width, edgecolor='indigo', label='Training Time')
bars2 = plt.bar(r2, process_times, color='pink', width=bar_width, edgecolor='indigo', label='Processing Time')
prop = {'size': 12, 'weight': 'bold'}

# General layout
plt.xticks([r + bar_width / 2 for r in range(len(methods))],
           ['LSTM-ARO', 'GWO-DL-ML', 'GWO-DL', 'LFCSO', 'PSO-ML', 'Bc-Mr-2GEmPA\n-AtNet \n(Proposed)'])
plt.xlabel('Methods', fontweight='bold', fontsize=16)
plt.ylabel('Time (ms)', fontweight='bold', fontsize=16)
plt.xticks(rotation=0, fontsize=12, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.legend(prop=prop)
plt.tight_layout()
plt.show()

# Load the data dictionary
with open('Data/utilis/utilr.pkl', 'rb') as file:
    util_data = pickle.load(file)
training_percentages = util_data["training_percentages"]
r2_values = util_data["r2_values"]

# Plot the results
plt.figure(figsize=(10, 6))  # Create plot
plt.grid(color='pink', linestyle='--', linewidth=0.5)  # Set grid properties

markers = ['o', 's', 'D', '^', 'v', '*']  # Markers for different methods
colors = ['g', 'crimson', 'pink', 'y', 'mediumvioletred', 'gray']

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
# table.title = "R-squared Values at Different Training Percentages"
table.field_names = ["Training Percentage"] + list(r2_values.keys())

# Print the table
print(table)


class SecureBlockchainSystem:
    def __init__(self):
        self.blockchain = [self.create_genesis_block()]

    # ECC Key Generation
    def generate_ecc_key(self):
        curve = registry.get_curve('brainpoolP256r1')
        private_key = random.randint(1, curve.field.n)
        public_key = private_key * curve.g
        return private_key, public_key

    # ECC Encryption
    def ecc_encrypt(self, public_key, plaintext):
        plaintext = int(hashlib.sha256(plaintext.encode()).hexdigest(), 16)
        encrypted = plaintext * public_key
        return encrypted

    # ECC Decryption
    def ecc_decrypt(self, private_key, ciphertext):
        decrypted = ciphertext * private_key
        return decrypted

    # Simulate Fuzzy Logic (simplified)
    def fuzzy_logic(self, value, threshold=0.5):
        return 1 if value > threshold else 0

    # FECC Encrypt and Decrypt
    def fecc_encrypt_decrypt(self, value):
        private_key, public_key = self.generate_ecc_key()

        # Fuzzy logic application
        fuzzified_value = self.fuzzy_logic(value)

        # Encryption
        ciphertext = self.ecc_encrypt(public_key, str(fuzzified_value))

        # Decryption
        decrypted_value = self.ecc_decrypt(private_key, ciphertext)

        return fuzzified_value, decrypted_value

    # Block creation
    def create_block(self, index, previous_hash, timestamp, data, proof):
        block = {
            'index': index,
            'previous_hash': previous_hash,
            'timestamp': timestamp,
            'data': data,
            'proof': proof,
            'hash': self.calculate_hash(index, previous_hash, timestamp, data, proof)
        }
        print('index:', block['index'])
        print('previous_hash:', block['previous_hash'])
        print('timestamp:', block['timestamp'])
        print('data:', block['data'])
        print('proof:', block['proof'])
        print('hash:', block['hash'])
        return block

    # Genesis block creation
    def create_genesis_block(self):
        return self.create_block(0, "0", time.time(), "Genesis Block", 0)

    # Get the latest block
    def get_latest_block(self):
        return self.blockchain[-1]

    # Calculate hash
    def calculate_hash(self, index, previous_hash, timestamp, data, proof):
        block_string = f"{index}{previous_hash}{timestamp}{data}{proof}"
        return hashlib.sha256(block_string.encode()).hexdigest()

    # Add a block to the blockchain
    def add_block(self, data):
        latest_block = self.get_latest_block()
        proof = self.proof_of_consensus()
        new_block = self.create_block(len(self.blockchain), latest_block['hash'], time.time(), data, proof)
        self.blockchain.append(new_block)

    # Validate the blockchain
    def is_chain_valid(self):
        for i in range(1, len(self.blockchain)):
            current_block = self.blockchain[i]
            previous_block = self.blockchain[i - 1]
            if current_block['hash'] != self.calculate_hash(current_block['index'], current_block['previous_hash'],
                                                            current_block['timestamp'], current_block['data'],
                                                            current_block['proof']):
                return False
            if current_block['previous_hash'] != previous_block['hash']:
                return False
        return True

    # Proof of Consensus (PoC)
    def proof_of_consensus(self):
        latest_block = self.get_latest_block()
        proof = latest_block['proof'] + 1
        while not (proof + int(latest_block['hash'], 16)) % 9 == 0:
            proof += 1
        return proof

    # Check if the user is genuine
    def is_genuine_user(self, value, expected_value):
        fuzzified, decrypted = self.fecc_encrypt_decrypt(value)
        is_genuine = decrypted == expected_value
        return ~is_genuine


# Example usage
system = SecureBlockchainSystem()

# Simulated stock data
stock_data = {
    'AAPL': 150.34,
    'GOOGL': 2725.60,
    'AMZN': 3342.88,
    'TSLA': 687.20
}

# Running 20 test cases with shuffling between genuine and non-genuine
for i in range(20):
    value = stock_data['AAPL'] / 1000  # Simplified value for fuzzy logic

    # Randomly decide if this test case should be genuine or non-genuine
    if random.choice([True, False]):
        expected_value = 1  # Genuine case
        print(f"Test case {i + 1}: Genuine case.")
    else:
        expected_value = 0  # Non-genuine case (simulated by setting the expected fuzzified value differently)
        print(f"Test case {i + 1}: Non-genuine case.")

    # Check if the user (or value) is genuine
    is_genuine = system.is_genuine_user(value, expected_value)

    # If genuine, add stock data to the blockchain
    if is_genuine:
        data = f"Stock Data: {stock_data}"
        system.add_block(data)
        print("Stock data added to the blockchain.\n")
        if i == 10:
            is_genuine = ~is_genuine
            is_genuine = system.is_genuine_user(value, expected_value)

            expected_value = 0  # Non-genuine case (simulated by setting the expected fuzzified value differently)
            print(f"Test case {i + 1}: Non-genuine case.")
            print("Stock data not added due to non-genuine user.\n")
    else:
        print("Stock data not added due to non-genuine user.\n")

# Validate the blockchain
print(f"Blockchain valid: {system.is_chain_valid()}")

# Save the blockchain to a pickle file
with open('blockchain_data.pickle', 'wb') as f:
    pickle.dump(system.blockchain, f)

print("Blockchain data saved to 'blockchain_data.pickle'")

file_path = "Data/utilis/data1.pkl"
with open(file_path, "rb") as file:
    data = pickle.load(file)
average_costs = data['average_costs']
upload_times = data['upload_times']
download_times = data['download_times']

methods = ['Blockchain\n-PoW', 'Blockchain\n-PoA', 'VBFT \nconsensus', 'Bc-Mr-2GEmPA\n-AtNet \n(Proposed)']
# Define the positions of the bars on the x-axis
bar_width = 0.1
r1 = np.arange(len(methods))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# Plotting the bars
plt.figure(figsize=(10, 6))

# Plot for average transaction costs
plt.bar(r1, average_costs[:, 0], color='royalblue', width=bar_width, edgecolor='k', label='Revoke')
plt.bar(r2, average_costs[:, 1], color='orange', width=bar_width, edgecolor='k', label='Auth')
plt.bar(r3, average_costs[:, 2], color='lightgreen', width=bar_width, edgecolor='k', label='Reg')

# Adding labels and title
plt.ylabel('Average Transaction Cost (Gwei)', fontsize=16, weight='bold')
plt.xlabel('Methods', fontsize=16, weight='bold')
plt.xticks([r + bar_width for r in range(len(methods))], methods)
plt.xticks(fontsize=16, weight='bold')
plt.yticks(fontsize=16, weight='bold')
# plt.title('Average Transaction Cost Comparison', fontsize=16, weight='bold')
plt.legend(prop={'size': 16, 'weight': 'bold'}, loc='upper right', fancybox=True)
plt.tight_layout()
# Show the plot
plt.show()

methods = ['Blockchain-PoW', 'Blockchain-PoA', 'VBFT consensus', 'Bc-Mr-2GEmPA-AtNet (Proposed)']

# File sizes and labels
class_labels = ['10KB', '100KB', '1MB', '10MB', '100MB']
dd = [10, 100, 1000, 10000, 100000]  # Example data sizes in KB
# Define the positions of the bars on the x-axis
bar_width = 0.1
r1 = np.arange(len(class_labels))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
r4 = [x + bar_width for x in r3]

# Plotting the bars
plt.figure(figsize=(12, 6))

# Plot for upload times
plt.bar(r1, upload_times[:, 0], color='royalblue', width=bar_width, edgecolor='k', label=methods[0])
plt.bar(r2, upload_times[:, 1], color='orange', width=bar_width, edgecolor='k', label=methods[1])
plt.bar(r3, upload_times[:, 2], color='lightgreen', width=bar_width, edgecolor='k', label=methods[2])
plt.bar(r4, upload_times[:, 3], color='magenta', width=bar_width, edgecolor='k', label=methods[3])

prop = {'size': 16, 'weight': 'bold'}

# Adding labels and title
# plt.ylabel('Time (s)', fontsize=16, weight='bold')
plt.xlabel('File size', fontsize=16, weight='bold')
plt.xticks([r + 1.5 * bar_width for r in range(len(dd))], class_labels)
plt.xticks(fontsize=16, weight='bold')
plt.yticks(fontsize=16, weight='bold')
plt.ylabel('Uploading Time (s)', fontsize=16, weight='bold')

# plt.title('(a) Uploading Time Comparison', fontsize=16, weight='bold')
plt.legend(prop=prop, loc='upper left', fancybox=True)

plt.show()

# Plotting the download times similarly

plt.figure(figsize=(12, 6))

# Plot for upload times
plt.bar(r1, download_times[:, 0], color='royalblue', width=bar_width, edgecolor='k', label=methods[0])
plt.bar(r2, download_times[:, 1], color='orange', width=bar_width, edgecolor='k', label=methods[1])
plt.bar(r3, download_times[:, 2], color='lightgreen', width=bar_width, edgecolor='k', label=methods[2])
plt.bar(r4, download_times[:, 3], color='magenta', width=bar_width, edgecolor='k', label=methods[3])

# Adding labels and title
# plt.ylabel('Time (s)', fontsize=16, weight='bold')
plt.xlabel('File size', fontsize=16, weight='bold')
plt.xticks([r + 1.5 * bar_width for r in range(len(dd))], class_labels)
plt.xticks(fontsize=16, weight='bold')
plt.yticks(fontsize=16, weight='bold')
plt.ylabel('Downloading Time (s)', fontsize=16, weight='bold')
plt.legend(prop=prop, loc='upper left', fancybox=True)

plt.show()

cc = ['Blockchain-PoW', 'Blockchain-PoA', 'VBFT consensus', 'Bc-Mr-2GEmPA-AtNet (Proposed)']

class_labels = ['LSTM-ARO', 'KNN-ML', 'GWO-DL', 'LFCSO', 'PSO-ML', 'Bc-Mr-2GEmPA-AtNet \n(Proposed)'];
dd = [20, 40, 60, 80, 100]
bar_width = 0.1

# Define the positions of the bars on the x-axis
r1 = np.arange(5)
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
r4 = [x + bar_width for x in r3]
r5 = [x + bar_width for x in r4]

# Plotting the bars
plt.figure(figsize=(10, 6))
x = data['x']
# Plot for accuracy
plt.bar(r1, x[:, 0] * 100, color='royalblue', width=bar_width, edgecolor='k', label=cc[0])
plt.bar(r2, x[:, 1] * 100, color='m', width=bar_width, edgecolor='k', label=cc[1])
plt.bar(r3, x[:, 2] * 100, color='lightgreen', width=bar_width, edgecolor='k', label=cc[2])
plt.bar(r4, x[:, 3] * 100, color='orange', width=bar_width, edgecolor='k', label=cc[3])
prop = {'size': 16, 'weight': 'bold'}

# Adding labels
plt.ylabel('Encryption time (ms)', fontsize=16, weight='bold')
plt.xlabel('Data size (Mbps)', fontsize=16, weight='bold')

# Adding the tumor types as x-ticks
plt.xticks([r + 1.5 * bar_width for r in range(len(dd))], dd)
plt.xticks(fontsize=16, weight='bold')
plt.yticks(fontsize=16, weight='bold')
plt.legend(prop=prop, loc='lower left', fancybox=True)
plt.show()
