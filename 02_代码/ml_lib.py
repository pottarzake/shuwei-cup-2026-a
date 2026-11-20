# -*- coding: utf-8 -*-
# The team reports DeepSeek (DeepSeek, model variant and release date unconfirmed) assisted the project code.
"""
ml_lib.py —— numpy 手写迷你机器学习库（队友负责，2026-11-20）

存在理由：本机网络对大文件下载限速到 ~160 B/s，scikit-learn 装不上。
T12 需要的四个模型全部用 numpy 实现：
  - QuadraticResponseSurface  二次响应面（最小二乘，含全部交叉项）
  - RegressionTree / RandomForest   CART 回归树 / 随机森林
  - GradientBoosting          平方损失梯度提升（限深 3，防过拟合）
  - GaussianProcessRBF        高斯过程回归（RBF 核 + 噪声，网格搜索超参，
                              带预测区间 —— T14 直接复用）
  - group_kfold               GroupKFold（按 ts 整组折叠，防泄漏）

只依赖 numpy。数据量小（72 训练组），性能无压力。
"""

import numpy as np


# ---------------- 工具 ----------------

def group_kfold(groups, n_splits):
    """groups: 数组。返回 (train_idx, test_idx) 生成器，同组必同折。"""
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rng = np.random.RandomState(42)
    rng.shuffle(uniq)
    folds = np.array_split(uniq, n_splits)
    idx = np.arange(len(groups))
    for f in folds:
        test = np.isin(groups, f)
        yield idx[~test], idx[test]


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(np.abs((y - p) / y)) * 100)


def spearman(y, p):
    """秩相关系数（排序一致性）， Pearson on ranks。"""
    def rank(a):
        a = np.asarray(a, float)
        order = np.argsort(np.argsort(a))
        return order.astype(float)
    ry, rp = rank(y), rank(p)
    if ry.std() == 0 or rp.std() == 0:
        return np.nan
    return float(np.corrcoef(ry, rp)[0, 1])


# ---------------- 二次响应面 ----------------

class QuadraticResponseSurface:
    """对前两个特征做二次展开（含交叉项），其余特征线性加入。"""

    def _expand(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        a, b = X[:, 0], X[:, 1]
        rest = X[:, 2:]
        return np.column_stack([np.ones(len(X)), a, b, a * a, a * b, b * b, rest])

    def fit(self, X, y):
        Z = self._expand(X)
        self.coef_, *_ = np.linalg.lstsq(Z, np.asarray(y, float), rcond=None)
        return self

    def predict(self, X):
        return self._expand(X) @ self.coef_


# ---------------- CART 回归树 ----------------

class RegressionTree:
    def __init__(self, max_depth=3, min_samples_leaf=3, max_features=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features  # None=全部, int=每节点随机特征数

    def fit(self, X, y):
        X = np.atleast_2d(np.asarray(X, float))
        y = np.asarray(y, float)
        self._rng = np.random.RandomState(7)
        self._nfeat = X.shape[1]
        self._tree = self._build(X, y, 0)
        return self

    def _best_split(self, X, y):
        n, p = X.shape
        mse_parent = np.var(y) * n
        best = None
        feats = np.arange(p)
        if self.max_features and self.max_features < p:
            feats = self._rng.choice(p, self.max_features, replace=False)
        for j in feats:
            xj = X[:, j]
            uniq = np.unique(xj)
            if len(uniq) < 2:
                continue
            thr = 0.5 * (uniq[:-1] + uniq[1:])
            for t in thr:
                left = xj <= t
                nl, nr = left.sum(), (~left).sum()
                if nl < self.min_samples_leaf or nr < self.min_samples_leaf:
                    continue
                mse = (np.var(y[left]) * nl + np.var(y[~left]) * nr)
                gain = mse_parent - mse
                if best is None or gain > best[0]:
                    best = (gain, j, t)
        return best

    def _build(self, X, y, depth):
        node = {"value": float(np.mean(y))}
        if depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf \
                or np.var(y) < 1e-12:
            return node
        split = self._best_split(X, y)
        if split is None or split[0] <= 1e-9:
            return node
        _, j, t = split
        left = X[:, j] <= t
        node.update({"feat": j, "thr": float(t),
                     "left": self._build(X[left], y[left], depth + 1),
                     "right": self._build(X[~left], y[~left], depth + 1)})
        return node

    def _predict_one(self, x):
        node = self._tree
        while "feat" in node:
            node = node["left"] if x[node["feat"]] <= node["thr"] else node["right"]
        return node["value"]

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        return np.array([self._predict_one(x) for x in X])


# ---------------- 随机森林 ----------------

class RandomForest:
    def __init__(self, n_estimators=200, max_depth=5, min_samples_leaf=2,
                 max_features="sqrt", seed=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.seed = seed

    def fit(self, X, y):
        X = np.atleast_2d(np.asarray(X, float))
        y = np.asarray(y, float)
        n, p = X.shape
        rng = np.random.RandomState(self.seed)
        mf = None if self.max_features is None else \
            (max(1, int(round(np.sqrt(p)))) if self.max_features == "sqrt"
             else int(self.max_features))
        self._trees = []
        for _ in range(self.n_estimators):
            idx = rng.randint(0, n, n)
            tree = RegressionTree(self.max_depth, self.min_samples_leaf, mf)
            tree.fit(X[idx], y[idx])
            self._trees.append(tree)
        return self

    def predict(self, X):
        return np.mean([t.predict(X) for t in self._trees], axis=0)


# ---------------- 梯度提升 ----------------

class GradientBoosting:
    """平方损失 GBM：F = F0 + lr * Σ tree_m。限深 3 防过拟合（任务卡口径）。"""

    def __init__(self, n_estimators=200, learning_rate=0.05, max_depth=3,
                 min_samples_leaf=3, subsample=1.0, seed=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.seed = seed

    def fit(self, X, y):
        X = np.atleast_2d(np.asarray(X, float))
        y = np.asarray(y, float)
        n = len(y)
        rng = np.random.RandomState(self.seed)
        self._F0 = float(np.mean(y))
        F = np.full(n, self._F0)
        self._trees = []
        for _ in range(self.n_estimators):
            m = max(10, int(self.subsample * n))
            idx = rng.choice(n, m, replace=False) if self.subsample < 1 \
                else np.arange(n)
            tree = RegressionTree(self.max_depth, self.min_samples_leaf)
            tree.fit(X[idx], y[idx] - F[idx])
            F = F + self.learning_rate * tree.predict(X)
            self._trees.append(tree)
        return self

    def predict(self, X):
        F = np.full(len(np.atleast_2d(X)), self._F0)
        for t in self._trees:
            F = F + self.learning_rate * t.predict(X)
        return F


# ---------------- 高斯过程（RBF + 噪声） ----------------

class GaussianProcessRBF:
    """零均值 GP，RBF 各向同性核。超参（长度尺度 l、噪声比 a）用
    对数边际似然在网格上选择。predict 返回均值与标准差（预测区间用）。"""

    def __init__(self, l_grid=(0.05, 0.1, 0.2, 0.35, 0.6, 1.0),
                 a_grid=(1e-4, 1e-3, 1e-2, 5e-2)):
        self.l_grid = l_grid
        self.a_grid = a_grid

    @staticmethod
    def _norm(X, lo, hi):
        return (np.asarray(X, float) - lo) / (hi - lo + 1e-12)

    def fit(self, X, y):
        X = np.atleast_2d(np.asarray(X, float))
        y = np.asarray(y, float)
        self._lo, self._hi = X.min(0), X.max(0)
        Xn = self._norm(X, self._lo, self._hi)
        self._ym, self._ys = float(np.mean(y)), float(np.std(y) + 1e-12)
        yn = (y - self._ym) / self._ys
        n = len(yn)
        best = None
        for l in self.l_grid:
            K = self._K(Xn, Xn, l)
            for a in self.a_grid:
                C = K + a * np.eye(n)
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    continue
                alpha = np.linalg.solve(C, yn)
                lml = -0.5 * float(yn @ alpha) - float(np.log(np.diag(L)).sum())
                if best is None or lml > best[0]:
                    best = (lml, l, a, C)
        _, self._l, self._a, self._C = best
        self._Xn = Xn
        self._alpha = np.linalg.solve(self._C, yn)
        return self

    @staticmethod
    def _K(A, B, l):
        d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return np.exp(-0.5 * d2 / (l * l))

    def predict(self, X, return_std=False):
        Xn = self._norm(np.atleast_2d(np.asarray(X, float)), self._lo, self._hi)
        Ks = self._K(Xn, self._Xn, self._l)
        mean = self._ym + self._ys * (Ks @ self._alpha)
        if not return_std:
            return mean
        Kss = np.ones(len(Xn))
        v = np.linalg.solve(self._C, Ks.T)
        var = self._ys ** 2 * np.maximum(Kss - (Ks * v.T).sum(1), 1e-12)
        return mean, np.sqrt(var)
