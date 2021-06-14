class CF_OGD:
    def __init__(self, dataset, window=5, eps=10, learning_rate=0):
        self.dataset = dataset
        self.dataset_shape = dataset.shape[1]
        self.window = window
        self.eps = eps
        self.learning_rate = learning_rate if learning_rate != 0 else 1/self.dataset_shape
        self.MA = lambda x, history:(history / x).mean(axis=0)
        # self.EMA = lambda alpha, x, history: self.algo_EMA(alpha, x, history) / x
        self.L1_median = lambda x, history:np.median(history, axis=0) / x
        self.algos_num = 6
        self.weights = self.weight_price_init()
        self.b = self.weight_b_init(dataset.shape[1])
        self.b_history = []
        self.S = []

    def weight_price_init(self):
        weight = np.ones(self.algos_num) / self.algos_num
        return weight.reshape(-1, 1) 

    def weight_b_init(self, num_columns):
        return np.ones(num_columns) / num_columns

    def algo_EMA(self, alpha, x, history):
        if len(history)==0:
          return x 
        else:
          return alpha*x + (1-alpha)*self.algo_EMA(alpha, history[-1], history[:-1])

    def predict_algos(self, x, history):
        x, history = x.values, history[-self.window:].values
        ma_pred = self.MA(x, history)
        ema1_pred = self.algo_EMA(0.1, x, history) / x
        ema2_pred = self.algo_EMA(0.3, x, history) / x
        ema3_pred = self.algo_EMA(0.6, x, history) / x
        ema4_pred = self.algo_EMA(0.8, x, history) / x
        median_pred = self.L1_median(x, history)

        preds = np.append([ma_pred * x], [ema1_pred * x], axis=0)
        preds = np.append(preds, [ema2_pred * x], axis=0)
        preds = np.append(preds, [ema3_pred * x], axis=0)
        preds = np.append(preds, [ema4_pred * x], axis=0)
        preds = np.append(preds, [median_pred * x], axis=0)
        return preds

    def predict_price_ogd(self, x, history):
        pred_algos = self.predict_algos(x, history)
        predict = pred_algos * self.weights

        return predict.sum(axis=0)

    def loss_grad(self, p_preds, p_true):
        step1 = p_preds.dot(p_preds.T)
        step2 = step1.dot(self.weights)
        step3 =  p_preds.dot(p_true.T)
        return step2 - step3

    def step_portfolio(self, x, history):
        pred_x = self.predict_price_ogd(x, history) / x.values

        x_mean = np.mean(pred_x)
        variable = min(0., np.dot(pred_x, self.b.T) - self.eps / np.linalg.norm(pred_x - x_mean)**2)

        # update portfolio
        b = self.b - variable * (pred_x - x_mean)
        self.b = tools.simplex_proj(b)
        
    def update_price_weights(self, loss):
        weight_update = self.weights - self.learning_rate * loss
        self.weights =  tools.simplex_proj(weight_update)

    def run(self):
        data = self.dataset
        for i in range(6, len(data)-1):
            x = data.iloc[i]
            history = data.iloc[(i-6):i]
            p_preds = model.predict_algos(x, history)

            if i == 6:
                self.S.append(1)
            else:
                S = self.S[-1]*np.dot(x.values/history.values[-1], self.b.T)
                self.S.append(S)

            model.step_portfolio(x, history)
            self.b_history.append(model.b)
            p_true = np.array([data.iloc[i+1].values])
            loss = model.loss_grad(p_preds, p_true)
            model.update_price_weights(loss)
