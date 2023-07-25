import numpy as np


# class for posterior probability and convex function
class Prob:
    def __init__(self, x):
        self.x = x

    def multinormal_pdf(self, mu, cov):
        p = np.max(mu.shape)
        return np.array(
            [
                ((2 * np.pi) ** (-p / 2))
                * (np.linalg.det(cov) ** (-1 / 2))
                * np.exp(-0.5 * (x - mu).T @ np.linalg.inv(cov) @ (x - mu))
                for x in self.x
            ]
        )

    # define the posterior probability
    def posterior_prob(self, mu1, mu2, cov1, cov2, pi1, pi2):
        f1 = self.multinormal_pdf(mu1, cov1)
        f2 = self.multinormal_pdf(mu2, cov2)
        mixture = pi1 * f1 + pi2 * f2
        return pi2 * f2 / mixture


# convex function
class ConvexFunction:
    def __init__(self, x):
        self.x = x

    # a convex distance function
    def convex_function(self):
        return np.square(1 - 2 * self.x)

    # gradient of convex function
    def grad(self):
        return np.gradient(self.convex_function(), self.x)

    # gradient of convex function at point z
    def grad_z(self, z):
        return 4 - 8 * z


# class for self-consistent convex clustering
class SelfConsistentClustering:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # convex function value at point z
    def phi(self, z):
        return np.square(1 - 2 * z)

    # compute gradient
    def grad(self, z):
        # grad = np.array([np.gradient(self.y, x) for x in self.x])
        return 8 * z - 4
        # return grad.T

    # parametrize the hyperplane at point z
    def A(self, z):
        A = np.array([self.grad(z), -1])
        return A

    # X matrix
    def X(self, z):
        return np.array([z, self.phi(z)])

    # hyperplane
    def hyperplane(self, z):
        return self.grad(z) * self.x + self.phi(z) - self.grad(z) * z

    # distance from point to hyperplane
    def distance(self, z):
        return -(
            self.A(z) @ np.array([self.x, self.y]) + self.phi(z) - self.grad(z) * z
        ) / np.linalg.norm(self.A(z))

    # define the clustering algorithm
    def clustering(self, K, max_iter=1000):
        # initial center
        posterior = self.x
        phi = self.y
        X = np.array([posterior, phi]).T
        center0 = np.random.choice(np.linspace(0, 1, phi.shape[0]), K, replace=False)
        # distance from point to center
        dist_center0 = np.array(
            [SelfConsistentClustering(posterior, phi).distance(x) for x in center0]
        ).T
        # group
        group0 = np.array([np.where(x == np.min(x)) for x in dist_center0]).squeeze()
        group1 = np.random.randint(0, K, posterior.shape[0])
        loss = []
        iteration = 0
        while group0.tolist() != group1.tolist():
            # print(iter)
            iteration += 1
            group0 = group1
            center1 = np.array([np.mean(X[group0 == i], axis=0) for i in range(K)])

            dist_exp = np.array(
                [([np.linalg.norm(x - centeri) for x in X]) for centeri in center1]
            ).T

            center1 = posterior[dist_exp.argmin(axis=0)]

            dist_center1 = np.array(
                [SelfConsistentClustering(posterior, phi).distance(x) for x in center1]
            ).T

            group1 = np.array(
                [np.where(x == np.min(x)) for x in dist_center1]
            ).squeeze()

            loss.append(dist_exp.min(axis=1).mean())

            # print(np.sum(group0 != group1))
            if iteration > max_iter:
                break
        return group1, loss
