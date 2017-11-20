import numpy as np
import tensorflow as tf
import utils


class LogisticRegression:
    def __init__(self, degree, learning_rate, max_iter=np.inf, threshold=-np.inf):
        """Constructor

        Keyword arguments:
        degree -- degree of the polynomial, number of parameters
        learning_rate -- learning_rate for the gradient descent
        max_iter -- Maximal number of iterations for the minimizer. If not
                    provided, the minimizer will iterate until convergence to
                    threshold
        threshold -- Convergence threshold: if the absolute value of the difference
                     of the errors of two consecutive iterations of the minimizer
                     is less than this value, the minimizer will stop.
                     If not provided, the minimization will do max_iter iterations.
        """
        if max_iter == np.inf and threshold == -np.inf:
            raise
        self.learning_rate = learning_rate
        self.degree = degree
        self.max_iter = max_iter
        self.sess = None
        self.closed = True
        self.threshold = threshold

    def fit(self, X, y):
        """Fit the model to the data"""
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        loss = self._loss(X, y)
        train = optimizer.minimize(loss)

        init = tf.global_variables_initializer()
        self._check_session()
        self.sess.run(init)

        self.losses = []
        n_iter, prev_loss, next_loss = 0, 0, 0
        converged = False
        while n_iter < self.max_iter and not converged:
            prev_loss = next_loss
            self.sess.run(train)
            next_loss = self.sess.run(loss)
            self.losses.append(next_loss)
            if n_iter % 1000 == 0:
                print('iter', n_iter, 'loss:', next_loss)
            converged = np.abs(prev_loss - next_loss) < self.threshold
            if converged:
                print('Converged at iteration', n_iter, 'with loss difference', abs(prev_loss - next_loss))
            n_iter += 1
        self.loss = self.losses[-1]
        return self

    def predict_proba(self, X):
        """Computes the probability of each point to have label 1"""
        self._check_session()
        return self.sess.run(self._model(X))

    def predict(self, X, threshold=0.5):
        """Predicts labels"""
        probs = self.predict_proba(X)
        probs[probs < threshold] = 0
        probs[probs >= threshold] = 1
        return probs

    def close(self):
        """Clears the tensorflow session"""
        self.sess.close()
        self.closed = True

    def _check_session(self):
        if self.closed or self.sess is None:
            self.sess = tf.Session()
            self.closed = False

    def _model(self, X):
        x = tf.convert_to_tensor(utils.polynomial_enhancement(X, self.degree), dtype=tf.float32)
        expXw = tf.exp(x @ self.w)
        model = 1 / (1 + tf.pow(expXw, -1))
        return model

    def _loss(self, X, y):
        x_train = tf.convert_to_tensor(utils.polynomial_enhancement(X, self.degree),
                                       dtype=tf.float32)
        y_train = tf.convert_to_tensor(y, dtype=tf.float32)
        y_train = tf.reshape(y_train, [tf.shape(y_train)[0], 1])
        self.w = tf.Variable(tf.zeros([X.shape[1] * self.degree + 1, 1]),
                             dtype=tf.float32, name='w')

        txw = x_train @ self.w
        loss = tf.reduce_sum(tf.log(1 + tf.exp(txw)) - tf.multiply(y_train, txw))
        return loss

