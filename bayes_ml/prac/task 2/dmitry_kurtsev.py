import numpy as np
from scipy.signal import fftconvolve
from scipy.special import softmax
from tqdm import tqdm


def calculate_exp(X, F, B):
    XF_mul = 2*fftconvolve(X, F[::-1,::-1,None], 'valid')
    XB_sum = (B[:,:,None]*X).sum(axis=(0, 1))
    XB_face = fftconvolve(B[:,:,None]*X, np.ones(F.shape)[:,:,None], 'valid')
    BB_face = fftconvolve(B**2, np.ones(F.shape), 'valid')

    ll = (X**2).sum(axis=(0, 1)) - 2*(XB_sum[None, None, :] - XB_face) + \
         (B**2).sum() - BB_face[:,:,None] - XF_mul + (F**2).sum()
    return ll


def calculate_log_probability(X, F, B, s):
    """
    Calculates log p(X_k|d_k,F,B,s) for all images X_k in X and
    all possible displacements d_k.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.

    Returns
    -------
    ll : array, shape(H-h+1, W-w+1, K)
        ll[dh,dw,k] - log-likelihood of observing image X_k given
        that the villain's face F is located at displacement (dh, dw)
    """
    H, W = B.shape
    ll = calculate_exp(X, F, B)
    return (-H*W*np.log(2*np.pi*s**2) - ll/s**2) / 2


def run_e_step(X, F, B, s, A, use_MAP=False):
    """
    Given the current esitmate of the parameters, for each image Xk
    esitmates the probability p(d_k|X_k,F,B,s,A).

    Parameters
    ----------
    X : array, shape(H, W, K)
        K images of size H x W.
    F  : array_like, shape(h, w)
        Estimate of villain's face.
    B : array shape(H, W)
        Estimate of background.
    s : scalar, shape(1, 1)
        Eestimate of standard deviation of Gaussian noise.
    A : array, shape(H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    use_MAP : bool, optional,
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.

    Returns
    -------
    q : array
        If use_MAP = False: shape (H-h+1, W-w+1, K)
            q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
            of villain's face given image Xk
        If use_MAP = True: shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    """
    pX_dFBs = calculate_log_probability(X, F, B, s)
    log_p_d = pX_dFBs + np.log(A + 1e-50)[:,:,None]
    max_ = log_p_d.max(axis=(0, 1))
    q = softmax(log_p_d - max_, axis=(0, 1))
    if use_MAP:
        max_idx = q.reshape(-1, q.shape[2]).argmax(0)
        q = np.column_stack(np.unravel_index(max_idx, q[:,:,0].shape)).T
    return q


def run_m_step(X, q, h, w, use_MAP=False):
    """
    Estimates F,B,s,A given esitmate of posteriors defined by q.

    Parameters
    ----------
    X : array, shape(H, W, K)
        K images of size H x W.
    q  :
        if use_MAP = False: array, shape (H-h+1, W-w+1, K)
           q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
           of villain's face given image Xk
        if use_MAP = True: array, shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    h : int
        Face mask height.
    w : int
        Face mask width.
    use_MAP : bool, optional
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.

    Returns
    -------
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    """
    H, W, K = X.shape
    if use_MAP:
        z = np.zeros((H-h+1, W-w+1, K))
        z[q[0], q[1], np.arange(K)] = 1
        q = z
    
    A = q.sum(axis=2) / K
    F = fftconvolve(X, q[::-1, ::-1,:], 'valid', axes=(0, 1)).sum(axis=2) / K
    
    q_not_face = 1 - fftconvolve(q, np.ones(F.shape)[:,:,None])
    B = (q_not_face*X).sum(axis = 2) / (q_not_face.sum(axis = 2) + 1e-50)

    s = (calculate_exp(X, F, B) * q).sum() / (H*W*K)
    return F, B, np.sqrt(s), A 


def calculate_lower_bound(X, F, B, s, A, q, use_MAP=False):
    """
    Calculates the lower bound L(q,F,B,s,A) for the marginal log likelihood.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    q : array
        If use_MAP = False: shape (H-h+1, W-w+1, K)
            q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
            of villain's face given image Xk
        If use_MAP = True: shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    use_MAP : bool, optional
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.

    Returns
    -------
    L : float
        The lower bound L(q,F,B,s,A) for the marginal log likelihood.
    """
    H, W, K = X.shape
    h, w = F.shape
    if use_MAP:
        z = np.zeros((H-h+1, W-w+1, K))
        z[q[0], q[1], np.arange(K)] = 1
        q = z
    E_log_p = (calculate_log_probability(X, F, B, s) * q).sum()
    E_log_p += (np.log(A + 1e-50)[:,:,None] * q).sum()
    E_log_q = (np.log(q + 1e-50) * q).sum()
    return E_log_p - E_log_q

def run_EM(X, h, w, F=None, B=None, s=None, A=None, tolerance=0.001,
           max_iter=50, use_MAP=False):
    """
    Runs EM loop until the likelihood of observing X given current
    estimate of parameters is idempotent as defined by a fixed
    tolerance.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    h : int
        Face mask height.
    w : int
        Face mask width.
    F : array, shape (h, w), optional
        Initial estimate of villain's face.
    B : array, shape (H, W), optional
        Initial estimate of background.
    s : float, optional
        Initial estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1), optional
        Initial estimate of prior on displacement of face in any image.
    tolerance : float, optional
        Parameter for stopping criterion.
    max_iter  : int, optional
        Maximum number of iterations.
    use_MAP : bool, optional
        If true then after E-step we take only MAP estimates of displacement
        (dh,dw) of villain's face given image Xk.

    Returns
    -------
    F, B, s, A : trained parameters.
    LL : array, shape(number_of_iters,)
        L(q,F,B,s,A) after each EM iteration (1 iteration = 1 e-step + 1 m-step); 
        number_of_iters is actual number of iterations that was done.
    """
    H, W, K = X.shape
    if F is None:
        F = np.random.uniform(0, X.max(), h*w).reshape(h, w)
    if B is None:
        B = np.random.uniform(0, X.max(), H*W).reshape(H, W)
    if s is None:
        s = 1
    if A is None:
        A = np.random.uniform(0, X.max(), (H-h+1)*(W-w+1)).reshape(H-h+1, W-w+1)
        A = A / A.sum()
    
    q = run_e_step(X, F, B, s, A, use_MAP=use_MAP)
    F, B, s, A  = run_m_step(X, q, h, w, use_MAP=use_MAP)
    L_old = calculate_lower_bound(X, F, B, s, A, q, use_MAP=use_MAP)
    LL = [L_old]
    for i in tqdm(range(max_iter - 1)):
        q = run_e_step(X, F, B, s, A, use_MAP=use_MAP)
        F, B, s, A  = run_m_step(X, q, h, w, use_MAP=use_MAP)
        L_new = calculate_lower_bound(X, F, B, s, A, q, use_MAP=use_MAP)
        LL.append(L_new)
        if L_new - L_old < tolerance:
            break
        L_old = L_new
    return F, B, s, A, np.array(LL)


def run_EM_with_restarts(X, h, w, tolerance=0.001, max_iter=50, use_MAP=False,
                         n_restarts=10):
    """
    Restarts EM several times from different random initializations
    and stores the best estimate of the parameters as measured by
    the L(q,F,B,s,A).

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    h : int
        Face mask height.
    w : int
        Face mask width.
    tolerance, max_iter, use_MAP : optional parameters for EM.
    n_restarts : int
        Number of EM runs.

    Returns
    -------
    F : array,  shape (h, w)
        The best estimate of villain's face.
    B : array, shape (H, W)
        The best estimate of background.
    s : float
        The best estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        The best estimate of prior on displacement of face in any image.
    L : float
        The best L(q,F,B,s,A).
    """
    F_best, B_best, s_best, A_best, L = run_EM(X, h, w)
    L_best = L[-1]
    for i in tqdm(range(n_restarts - 1)):
        F, B, s, A, L = run_EM(X, h, w)
        if L[-1] > L_best:
            F_best, B_best, s_best, A_best, L_best = F, B, s, A, L[-1]
    return F_best, B_best, s_best, A_best, L_best
