import numpy as np

def sort_by_lambdas(lambdas, eigenvectors):
  return lambdas[idx], eigenvectors[idx]

def pca(X, n_components=2):
  """
  Args)
    X (np.array): 
      変換したい行列
      X.shape = (3102, 2000) = (細胞数,遺伝子数)
  Returns)
    transformed 
  """
  # 共分散行列計算
  ## Sigma.shape = (3102, 3102)
  N = X.shape[0]
  X_ = X - X.mean(axis=0).reshape(1,-1)
  Sigma = 1/N * X_.T @ X_

  # 共分散行列の固有値、固有ベクトル計算
  ## lambdas.shape =(2000,), eigenvectors.shape = (2000, 2000)
  ## lambdas[i]の固有ベクトルはeigenvectors[:,i]に対応する
  lambdas, eigenvectors = np.linalg.eig(Sigma)

  # 固有値が大きい順でソート＝射影先の分散が大きい順でソート
  idx = lambdas.argsort()[::-1]
  lambdas, eigenvectors = lambdas[idx], eigenvectors[idx]

  # 変換
  ## transformed.shape = (3102, 2000)
  transformed = X_ @ eigenvectors
  
  return transformed[:,:n_components]
    

def calc_factor_loadings(X, transformed, gene_labels, nth_factor=0):
  """
  nth_factorで指定された主成分に対する因子負荷量を計算し、その最大値に対応する遺伝子IDを返す
  x.shape = (3102, 2000)
  transformed = (3102, 2)
  """
  N = X.shape[1]
  factor_loadings = []
  for n in range(N):
    factor_loadings.append(np.corrcoef((transformed[:, nth_factor], X[:, n]))[0,1])
  factor_loadings = np.abs(np.array(factor_loadings))
  gene_labels = np.array(gene_labels)

  idx = factor_loadings.argsort()[::-1]
  gene_labels = gene_labels[idx]
  #print(f"{factor_loadings[idx]=}")
  print(f"{gene_labels[:10]=}")

  return gene_labels[0]
