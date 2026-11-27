import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

class DecisionTree:
    def __init__(self, max_depth = None, min_samples_split = 0, max_features  = None):
        self.tree = None # weights
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features

    # reusable: calculate entropy for (one) value of a feature, according to the target proportion
    def entropy(self, y):
        no_instances = len(y)
        unique_values = y.unique()
        entropy = 0
        for value in unique_values:
            no_values = len(y[y == value])
            proportion = no_values / no_instances
            entropy -= proportion * np.log2(proportion)
        return entropy

    # reusable: calculate IG of a splitted feature
    def information_gain(self, data, feature, y, threshold):
        entropy_S = self.entropy(data[y])
        left = data[data[feature]<=threshold]
        right = data[data[feature]>threshold]
        weighted_entropy = (len(left)/len(data))*self.entropy(left[y]) + (len(right)/len(data))*self.entropy(right[y])
        return  entropy_S - weighted_entropy

    def calc_split_info(self, data, feature, threshold):
        p_left = (data[feature] <= threshold).mean()
        split_info = 0
        for p in (p_left, 1-p_left):
            if p > 0:
                split_info -= p * np.log2(p)
        return split_info
        

    # finding optimal slicing point for a feature
    def numeric_thresholding(self, df, feature, y):
        values = np.sort(df[feature].unique())
        # unique value handler
        if len(values) < 2:
            return None, 0
        slicing_candidates = (values[:-1]+values[1:])/2 # a list of possible values to be a slicing [L/R] threshold
        best_th, best_gain = None, -1
        # iterate through each slicing candidate as the threshold
        for t in slicing_candidates:
            gain = self.information_gain(df, feature, y,t)
            if gain > best_gain:
                best_th, best_gain = t, gain
        return best_th, best_gain

    # chosing best features based on IG (highest)
    # target for modification ID3 -> C4.5: choose_best_feature criteria
    # shift from plain IG to Gain Ratio
    def choose_best_feature(self, data, features, y):

        # for RF --> random feature subsampling mechanism
        # making each recursive calls from build method narrows available passed feature into randomly chosen feature of size k
        if self.max_features is not None:
            k = min(self.max_features, len(features))
            features = np.random.choice(features, size=k, replace=False)

        features_thresh_gains = {}
        for feature in features:
            th,gain = self.numeric_thresholding(data, feature, y)
            if th is not None:
                features_thresh_gains[feature] = (th, gain)
        gain_ratios = {}
        for feature, (th, gain) in features_thresh_gains.items():
            split_info = self.calc_split_info(data, feature, th)
            if split_info > 0:
                gain_ratios[feature] = gain / split_info
        if not gain_ratios:
            return None, None
        
        best_feature = max(gain_ratios, key=gain_ratios.get)
        return best_feature, features_thresh_gains[best_feature][0]
        


    # force split into left right
    def split(self, df, feature, threshold):
        left = df[df[feature]<=threshold]
        right = df[df[feature]>threshold]
        return left,right

    # recursive function, 
    # smaller problem: creating decision stump over and over
    # base: maxdepth reached, (modify, target): minimum data not reached
    # recursive: evaluating IG of all features and split   
    def build(self,df, features, target, depth = 0):
        if self.max_depth is not None and depth >= self.max_depth:#maxdepth reached
            return df[target].mode().iloc[0] # majority vote
        if len(np.unique(df[target]))==1: # pure node
            return df[target].iloc[0]
        if len(df)<self.min_samples_split:#minimum split escape
            return df[target].mode().iloc[0] # majority vote

        node = {}
        best_feature, threshold = self.choose_best_feature(df, features, target)
        if best_feature is None:
            return df[target].mode().iloc[0] # majority vote

        node['feature'] = best_feature
        node['children'] = {}
        node['split_value'] = threshold
        left, right = self.split(df, best_feature, threshold)
        node['children']['left'] = self.build(left, features, target, depth + 1)
        node['children']['right'] = self.build(right, features, target, depth + 1)
        return node
            
    def fit(self, X, y):
        # per-row evaluation, previously detached X,y data is merged again
        merged_df = X.copy()
        merged_df['target'] = y
        features = merged_df.columns[:-1]
        target = merged_df.columns[-1]
        self.tree = self.build(merged_df, features, target)

    def predict_instance(self, X, tree):
        # loop until the children is a label
        while isinstance(tree, dict): 
            feature = tree['feature']
            # match condition of the feature value of an instance
            if X[feature] <= tree['split_value']:
                tree = tree['children']['left']
            else:
                tree = tree['children']['right']
        return tree

    # predict data
    def predict(self,X):
        predictions = []
        for _, row in X.iterrows():
            predictions.append(self.predict_instance(row, self.tree))
        return predictions

class RandomForest:
    def __init__(self, no_tree = 10, max_depth = 3, min_samples_split = 0):
        self.forest = []
        self.no_tree = no_tree
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self,X,y):
        # need: bootstraping data, feature sampling, flexible according to the number of tree
        # bootstraping data: 
        merged_df = X.copy()
        merged_df['target'] = y
        feature_subsampling_sz = max(1,int(np.sqrt(X.shape[1])))
        
        for i in range(self.no_tree):
            print(f"\rTree {i + 1} of {self.no_tree}...", end="", flush=True)
            # bootstrap sampling using pandas
            # frac 1 results 100% df length, 
            bagging_df = merged_df.sample(frac=1.0, replace=True).reset_index(drop=True)
            bagging_df_X = bagging_df.drop(columns=['target'])
            bagging_df_y = bagging_df['target']

            tree = DecisionTree(max_depth = self.max_depth, min_samples_split=self.min_samples_split, max_features = feature_subsampling_sz)
            tree.fit(bagging_df_X, bagging_df_y)
            self.forest.append(tree)

    def predict_instance(self,X,forest):
        vote = []
        for tree in forest:
            vote.append(tree.predict_instance(X,tree.tree))
        # majority voting
        return pd.Series(vote).mode().iloc[0] # majority vote

        
    def predict(self,X):
        predictions = []
        for _, row in X.iterrows():
            predictions.append(self.predict_instance(row, self.forest))
        return predictions

class ArtificialNeuralNetwork:
    def __init__(
        self,
        input_dim,
        hidden_layers=(64, 32),
        output_dim=3,
        activations='relu',
        learning_rate=0.001,
        epochs=100,
        batch_size=32,
        l2_lambda=0.0,
        random_state=42
    ):
        self.input_dim = input_dim
        self.hidden_layers = list(hidden_layers)
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2_lambda = l2_lambda
        self.random_state = random_state
        if isinstance(activations, str):
            self.activations = [activations.lower()] * len(self.hidden_layers)
        else:
            if len(activations) != len(self.hidden_layers):
                raise ValueError("Length of activations must match number of hidden layers.")
            self.activations = [act.lower() for act in activations]
        self.layer_dims = [self.input_dim] + self.hidden_layers + [self.output_dim]
        self.num_layers = len(self.layer_dims) - 1  # Number of weight matrices
        self.weights = {}
        self.biases = {}
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        self._initialize_parameters()

    def _initialize_parameters(self):
        rng = np.random.default_rng(self.random_state)
        for l in range(1, self.num_layers + 1):
            in_dim = self.layer_dims[l - 1]
            out_dim = self.layer_dims[l]

            if l <= len(self.activations) and self.activations[l - 1] in ('relu', 'leaky_relu'):
                std = np.sqrt(2.0 / in_dim)
            else:
                std = np.sqrt(1.0 / in_dim)

            self.weights[l] = rng.normal(0.0, std, size=(in_dim, out_dim))
            self.biases[l] = np.zeros((1, out_dim), dtype=np.float64)

    def _apply_activation(self, Z, name):
        if name == 'relu':
            return Activations.relu(Z)
        elif name == 'leaky_relu':
            return Activations.leaky_relu(Z)
        elif name == 'sigmoid':
            return Activations.sigmoid(Z)
        elif name == 'tanh':
            return Activations.tanh(Z)
        elif name == 'softmax':
            return Activations.softmax(Z)
        else:
            raise ValueError(f"Unsupported activation function: {name}")

    def _apply_activation_derivative(self, Z, name):
        if name == 'relu':
            return Activations.relu_derivative(Z)
        elif name == 'leaky_relu':
            return Activations.leaky_relu_derivative(Z)
        elif name == 'sigmoid':
            return Activations.sigmoid_derivative(Z)
        elif name == 'tanh':
            return Activations.tanh_derivative(Z)
        else:
            raise ValueError(f"Unsupported activation derivative: {name}")

    def forward(self, X):
        """Perform forward propagation across all layers and cache intermediate values."""
        cache = {'A0': X}
        A = X
        for l in range(1, self.num_layers):
            Z = A @ self.weights[l] + self.biases[l]
            A = self._apply_activation(Z, self.activations[l - 1])
            cache[f'Z{l}'] = Z
            cache[f'A{l}'] = A
        L = self.num_layers
        ZL = A @ self.weights[L] + self.biases[L]
        AL = Activations.softmax(ZL)
        cache[f'Z{L}'] = ZL
        cache[f'A{L}'] = AL
        return AL, cache

    def compute_loss(self, y_true_oh, y_pred_probs):
        m = y_true_oh.shape[0]
        eps = 1e-12
        probs_clipped = np.clip(y_pred_probs, eps, 1.0 - eps)
        ce_loss = -np.sum(y_true_oh * np.log(probs_clipped)) / m
        if self.l2_lambda > 0:
            l2_cost = (self.l2_lambda / (2 * m)) * sum(
                np.sum(np.square(self.weights[l])) for l in range(1, self.num_layers + 1)
            )
            return ce_loss + l2_cost
        return ce_loss

    def backward(self, y_true_oh, cache):
        m = y_true_oh.shape[0]
        grads_W = {}
        grads_b = {}
        L = self.num_layers
        dZ = (cache[f'A{L}'] - y_true_oh) / m
        grads_W[L] = cache[f'A{L - 1}'].T @ dZ + (self.l2_lambda / m) * self.weights[L]
        grads_b[L] = np.sum(dZ, axis=0, keepdims=True)
        for l in range(L - 1, 0, -1):
            dA = dZ @ self.weights[l + 1].T
            dZ = dA * self._apply_activation_derivative(cache[f'Z{l}'], self.activations[l - 1])
            grads_W[l] = cache[f'A{l - 1}'].T @ dZ + (self.l2_lambda / m) * self.weights[l]
            grads_b[l] = np.sum(dZ, axis=0, keepdims=True)

        return grads_W, grads_b

    def _update_parameters(self, grads_W, grads_b):
        for l in range(1, self.num_layers + 1):
            self.weights[l] -= self.learning_rate * grads_W[l]
            self.biases[l] -= self.learning_rate * grads_b[l]

    def fit(self, X_train, y_train_oh, X_val=None, y_val_oh=None, verbose=True,patience=15,monitor="val_loss",min_delta=1e-4):
        rng = np.random.default_rng(self.random_state)
        n_samples = X_train.shape[0]
        for epoch in range(1, self.epochs + 1):
            # Shuffle training data each epoch
            indices = rng.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train_oh[indices]
            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                _, cache = self.forward(X_batch)
                grads_W, grads_b = self.backward(y_batch, cache)
                self._update_parameters(grads_W, grads_b)
            train_probs, _ = self.forward(X_train)
            train_loss = self.compute_loss(y_train_oh, train_probs)
            train_acc = np.mean(np.argmax(train_probs, axis=1) == np.argmax(y_train_oh, axis=1))
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            if X_val is not None and y_val_oh is not None:
                val_probs, _ = self.forward(X_val)
                val_loss = self.compute_loss(y_val_oh, val_probs)
                val_acc = np.mean(np.argmax(val_probs, axis=1) == np.argmax(y_val_oh, axis=1))
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                if verbose and (epoch % 10 == 0 or epoch == 1 or epoch == self.epochs):
                    print(
                        f"Epoch {epoch:3d}/{self.epochs} | "
                        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
                    )
            else:
                if verbose and (epoch % 10 == 0 or epoch == 1 or epoch == self.epochs):
                    print(
                        f"Epoch {epoch:3d}/{self.epochs} | "
                        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}"
                    )
        return self

    def predict_proba(self, X):
        probs, _ = self.forward(X)
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

def gradient_check(model, X_sample, y_sample_oh, epsilon=1e-7):
    _, cache = model.forward(X_sample)
    grads_W, grads_b = model.backward(y_sample_oh, cache)

    for l in range(1, model.num_layers + 1):
        W = model.weights[l]
        dW_num = np.zeros_like(W)
        it = np.nditer(W, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            orig = W[idx]

            W[idx] = orig + epsilon
            p_plus, _ = model.forward(X_sample)
            loss_plus = model.compute_loss(y_sample_oh, p_plus)

            W[idx] = orig - epsilon
            p_minus, _ = model.forward(X_sample)
            loss_minus = model.compute_loss(y_sample_oh, p_minus)

            dW_num[idx] = (loss_plus - loss_minus) / (2 * epsilon)
            W[idx] = orig
            it.iternext()
        rel_error = np.linalg.norm(grads_W[l] - dW_num) / (
            np.linalg.norm(grads_W[l]) + np.linalg.norm(dW_num) + 1e-12
        )
        print(f"Layer {l} Weights Relative Gradient Error: {rel_error:.2e}")

class KNNClassifier():
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype = np.float32)
        self.y_train = np.asarray(y).ravel()

        self._classes_ , self.y_encoded = np.unique(self.y_train, return_inverse=True)
        return self

    def predict(self, X, distance_metric, batch_size = 64):
        X = np.asarray(X, dtype = np.float32)

        if distance_metric not in ['euclidean', 'manhattan']:
            raise ValueError("Invalid distance metric. Choose 'euclidean' or 'manhattan'.")

        Y_pred = np.empty(len(X), dtype=self._classes_.dtype)
        for start in range(0, len(X), batch_size):
            batch = X[start:start + batch_size]
            #shape : (batch_size, jumlah_data_train, jumlah_fitur)
            differences = batch[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]
            #cek distance metric
            if distance_metric == 'euclidean':
                distances = np.sqrt(np.sum(differences ** 2, axis=2))
            elif distance_metric == 'manhattan':
                distances = np.sum(np.abs(differences), axis=2)
            #ambil k tetangga terdekat
            nearest_indices = np.argsort(distances, axis=1)[:, :self.k]
            #ambil label dari tetangga terdekat 
            nearest_labels = self.y_encoded[nearest_indices]
            #hitung jumlah suara untuk setiap kelas
            votes = (
                nearest_labels[:, :, np.newaxis]
                == np.arange(len(self._classes_))[None, None, :]
            ).sum(axis=1)
            #ambil kelas dengan jumlah suara terbanyak
            winners = np.argmax(votes, axis=1)
            #set predicted labels untuk batch saat ini
            Y_pred[start:start + batch_size] = self._classes_[winners]

        return Y_pred

class LogisticRegression(ClassifierMixin, BaseEstimator):
    def __init__(self, learning_rate=0.01, num_iterations=1000, l2=0.01):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.l2 = l2

    def softmax(self, z):
        z -= z.max(axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y).ravel()

        self._classes_, encoded_y = np.unique(y, return_inverse=True)
        self.classes_ = self._classes_
        
        num_samples, num_features = X.shape
        num_classes = len(self._classes_)
        
        self.weights = np.zeros((num_features, num_classes))
        self.bias = np.zeros((1, num_classes))
        
        y_one_hot = np.eye(num_classes)[encoded_y]

        self.loss_history = []

        for iteration in range(self.num_iterations):
            scores = np.dot(X, self.weights) + self.bias
            probabilities = self.softmax(scores)

            loss = -np.mean(np.sum(y_one_hot * np.log(probabilities + 1e-15), axis=1))
            loss += 0.5*self.l2*np.sum(self.weights**2)

            self.loss_history.append(loss)

            error = probabilities - y_one_hot
            gradient_weights = (np.dot(X.T, error) / num_samples + self.l2*self.weights)
            gradient_bias = np.sum(error, axis=0, keepdims=True) / num_samples

            self.weights -= self.learning_rate * gradient_weights
            self.bias -= self.learning_rate * gradient_bias

        return self
    
    def predict(self, X):
        probabilities = self.predict_proba(X)
        winners = probabilities.argmax(axis=1)
        return self._classes_[winners]

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.weights.shape[0]:
            raise ValueError(f"Expected {self.weights.shape[0]} features; got {X.shape}.")
        return self.softmax(X @ self.weights + self.bias)
class Activations:
    @staticmethod
    def relu(Z):
        return np.maximum(0, Z)

    @staticmethod
    def relu_derivative(Z):
        return (Z > 0).astype(np.float64)

    @staticmethod
    def leaky_relu(Z, alpha=0.01):
        return np.where(Z > 0, Z, alpha * Z)

    @staticmethod
    def leaky_relu_derivative(Z, alpha=0.01):
        return np.where(Z > 0, 1.0, alpha)

    @staticmethod
    def sigmoid(Z):
        Z_clipped = np.clip(Z, -500, 500)
        return 1.0 / (1.0 + np.exp(-Z_clipped))

    @staticmethod
    def sigmoid_derivative(Z):
        s = Activations.sigmoid(Z)
        return s * (1.0 - s)

    @staticmethod
    def tanh(Z):
        return np.tanh(Z)

    @staticmethod
    def tanh_derivative(Z):
        return 1.0 - np.tanh(Z) ** 2

    @staticmethod
    def softmax(Z):
        Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z_shifted)
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)