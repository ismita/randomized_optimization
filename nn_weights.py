#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score
import mlrose_hiive
import time

seed = 3
# . means current folder, ~ means home folder
filename = 'default.csv' #'tmp_mmc_sc_coupon_group0927.txt'

df = pd.read_csv(filename,sep=',')

# df.info()

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, shuffle=True) # 0.25 x 0.8 = 0.2

X_train = scale(X_train)
X_test = scale(X_test)

lr_list = [0.1, 0.5, 1]
max_attp = 100
nodes_list = [[30], [30, 30]]
act="relu"
max_iters = 1000

lr_output = []
nodes_output=[]
train_accuracy_output = []
test_accuracy_output = []
time_output = []
alg_output = []

for lr in lr_list:
    for nodes in nodes_list:

        nn_model_gd = mlrose_hiive.NeuralNetwork(hidden_nodes=nodes, activation=act,
                                            algorithm='gradient_descent', max_iters=max_iters,
                                            bias=True, is_classifier=True, learning_rate=lr,
                                            early_stopping=True, clip_max=5, max_attempts=max_attp,
                                            random_state=seed)

        nn_model_rhc = mlrose_hiive.NeuralNetwork(hidden_nodes=nodes, activation=act,
                                            algorithm='random_hill_climb', max_iters=max_iters,
                                            bias=True, is_classifier=True, learning_rate=lr, restarts=2,
                                            early_stopping=True, clip_max=5, max_attempts=max_attp,
                                            random_state=seed)

        nn_model_sa = mlrose_hiive.NeuralNetwork(hidden_nodes=nodes, activation=act,
                                        algorithm='simulated_annealing', max_iters=max_iters,
                                        bias=True, is_classifier=True, learning_rate=lr,schedule=mlrose_hiive.GeomDecay(10, .95, .1),
                                        early_stopping=True, clip_max=5, max_attempts=max_attp,
                                        random_state=seed)


        nn_model_ga = mlrose_hiive.NeuralNetwork(hidden_nodes=nodes, activation=act,
                                        algorithm='genetic_alg', max_iters=max_iters, pop_size=20, mutation_prob=0.1,
                                        bias=True, is_classifier=True, learning_rate=lr,
                                        early_stopping=True, clip_max=5, max_attempts=max_attp,
                                        random_state=seed)

        alg_names = ["GD", "RHC", "SA", "GA"]
        neural_nets = {"GD": nn_model_gd, "RHC": nn_model_rhc, "SA": nn_model_sa, "GA": nn_model_ga}

        for alg in alg_names:
            nn = neural_nets[alg]
            print("current alg", alg)
            alg_output.append(alg)

            print("current lr", lr)
            lr_output.append(lr)

            print("current nodes", nodes)
            nodes_output.append(nodes)

            t = time.time()
            nn.fit(X_train, y_train)

            y_train_pred = nn.predict(X_train)
            y_train_accuracy = accuracy_score(y_train, y_train_pred)

            y_test_pred = nn.predict(X_test)
            y_test_accuracy = accuracy_score(y_test, y_test_pred)
            delta_t = time.time()-t
            print("Time needed: {}".format(delta_t))
            time_output.append(delta_t)

            print("Train accuracy for {}: {}".format(nn, y_train_accuracy))
            train_accuracy_output.append(y_train_accuracy)

            print("Test accuracy for {}: {}".format(nn, y_test_accuracy))
            test_accuracy_output.append(y_test_accuracy)


output = pd.DataFrame(list(zip(lr_output, nodes_output, train_accuracy_output, test_accuracy_output, time_output, alg_output)),
columns =['Learning rate', 'Network structure', 'Train Accuracy', 'Test Accuracy', 'Time duration(s)', 'Algorithm'])



# %%
