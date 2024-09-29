from pca import pca
import numpy as np
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import matplotlib.pyplot as plt


def get_dists(X, sklearn=True):
  """
  returns dist matrix of sum(||X_i - X_j||^2)
  対角成分は0
  """
  if sklearn:
    dists = pairwise_distances(X, metric="sqeuclidean")
    return dists

  n = X.shape[0]
  dists = np.zeros([n, n])

  for i in range(n):
    for j in range(i, n):
      dists[i, j] = np.sum((X[i,:] - X[j,:])**2)
      print(f"{i},{j}")
  return dists + dists.T

def get_P(X, target_perp,lowerbound=0, upperbound=1e4, maxit=250, tol=1e-6):
  """
  This function finds stadard deviations to target perplexities with binary search. Joint probailities are returned.
  """
  D = get_dists(X)
  n = D.shape[0]
  P = np.zeros(D.shape)

  for i in tqdm(range(n)):
    lowerbound_i = lowerbound
    upperbound_i = upperbound
    d = D[i,:]

    for t in tqdm(range(maxit), leave=False):
      sigma_i = (lowerbound_i + upperbound_i) / 2
    
      #calculate p_j|i
      d_scaled = -d/(2*sigma_i**2)
      d_scaled -= np.max(d_scaled)
      exp_D = np.exp(d_scaled)
      exp_D[i] = 0
      p_ji = exp_D / np.sum(exp_D)

      #calculate entropy
      current_perp = 2 ** (np.sum(-p_ji*np.log2(p_ji+1e-10)))

      if current_perp < target_perp:
        lowerbound_i = sigma_i
      else:
        upperbound_i = sigma_i

      if np.abs(current_perp-target_perp) < tol:
        break
      #print(f"i={i}, current_perp={current_perp}, sigma={sigma_i}")
      if p_ji.max() > 100:
        print(f"{p_ji.max()=}")
        print(f"{p_ji.min()=}")
    P[i,:] = p_ji
  return (P+P.T)/(2*n)
      



def grad_py(pij, qij, Y_dists, Y, numpy=True):
  n = Y.shape[0]
  dY = np.zeros(shape = Y.shape)
  rij = pij - qij

  if numpy:
    for i in tqdm(range(n), leave=False):
      dY[i,:] = 4*np.dot(rij[i,:]*Y_dists[i,:], Y[i,:] - Y)
  else:
   for i in range(n):
     for j in range(n):
       dY[i,:] += 4 * rij[i,j] * (Y[i,:] - Y[j,:]) * Y_dists[i,j]

  return dY

def step_based_lr(t, eta, d=0.01, r=30):
  return eta*d**np.floor((1+t)/r)

def tsne(X, niter=10, n_components=2, perplexity=10, exg=4, eta=0.1, alpha=0.8, return_all_iter=False, initialize_pca=False, min_grad_diff=1e-7, exg_thr=50, n_features_in=20):
  """
  Args)
    X (np.array):
      変換したいデータ
      X.shape = (3102, 2000) = (細胞数, 遺伝子数)
  Returns)
    Y: embeded array
      Y.shape = (3102, 2) if return_all_iter = False
      Y.shape = (niter+2, 3102, 2) otherwise
  """
  X = pca(X, n_components=n_features_in).real

  #Get affinities with exaggeration
  pij = exg * get_P(X, perplexity)

  #Initialization
  size = (pij.shape[0], n_components)
  Y = np.zeros(shape = (niter+2, size[0], n_components))
  initial_vals = pca(X, n_components=n_components).real if initialize_pca else np.random.normal(0.0, np.sqrt(1e-4), size)
  Y_m1 = Y_m2 = initial_vals
  Ys = [Y_m2, Y_m1]
  error = error_prev = 0

  for i in tqdm(range(2,niter+2)):
    if i == int(exg_thr):
      pij /= exg

    #Calculate qij
    ## qij = (1 + ||yi - yj||^(-1) / sum(1 + ||yi - yj||^(-1)))
    d = get_dists(Y_m1)
    Ydist = np.power(1 + d, -1)
    np.fill_diagonal(Ydist, 0)
    #qij = Ydist / np.sum(Ydist, axis=0)
    qij = Ydist / np.sum(Ydist)
    grad = grad_py(pij, qij, Ydist, Y_m1)
    #print(f" error:{np.linalg.norm(grad)}")
    error = np.linalg.norm(grad)
    if abs(error-error_prev) < min_grad_diff:
      break

    #update learning rate
    eta = step_based_lr(i, eta)

    #Update embeddings
    Y_new = Y_m1 - eta*grad + alpha*(Y_m1 - Y_m2)
    #Y_new = Y_new - np.mean(Y_new, axis=0)
    Y_m2, Y_m1 = Y_m1, Y_new
    Ys.append(Y_new)

    error_prev = error

  if return_all_iter:
    return Ys
  else:
    return Ys[-1]
