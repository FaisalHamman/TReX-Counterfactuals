import foolbox as fb
import tensorflow as tf
import numpy as np
import art




def IGD_L1(model,
           x,
           y_sparse,
           num_classes=2,
           confidence=0.5,
           stepsize=1.e-3,
           steps=100,
           clamp=[0, 1],
           batch_size=32,
           return_probits=False,
           **kwargs):

    y_orig_onehot = tf.keras.utils.to_categorical(y_sparse,
                                                  num_classes=num_classes)

    return iterative_attack(model,
                            x,
                            y_orig_onehot,
                            1.0,
                            epsilon_iter=stepsize,
                            max_iter=steps,
                            clip_min=clamp[0],
                            clip_max=clamp[1],
                            decision_rule='L1',
                            batch_size=batch_size,
                            confidence=confidence,
                            return_probits=return_probits,
                            num_classes=num_classes)


def IGD_L2(model,
           x,
           y_sparse,
           num_classes=2,
           confidence=0.5,
           stepsize=1.e-3,
           steps=100,
           clamp=[0, 1],
           batch_size=32,
           return_probits=False,
           **kwargs):

    y_orig_onehot = tf.keras.utils.to_categorical(y_sparse,
                                                  num_classes=num_classes,
                                                  dtype='float32')

    return iterative_attack(model,
                            x,
                            y_orig_onehot,
                            0.0,
                            epsilon_iter=stepsize,
                            max_iter=steps,
                            clip_min=clamp[0],
                            clip_max=clamp[1],
                            decision_rule='L2',
                            batch_size=batch_size,
                            confidence=confidence,
                            return_probits=return_probits,
                            num_classes=num_classes)


def iterative_attack(model,
                     x,
                     y_orig_onehot,
                     beta,
                     epsilon_iter=1e-1,
                     max_iter=100,
                     clip_min=-1.,
                     clip_max=1.,
                     decision_rule='EN',
                     batch_size=128,
                     confidence=0.5,
                     return_probits=True,
                     num_classes=2):

    classifier = art.estimators.classification.tensorflow.TensorFlowV2Classifier(
        model=model,
        clip_values=[np.min(clip_min), np.max(clip_max)],
        nb_classes=num_classes,
        input_shape=x.shape[1:])

    attack = art.attacks.evasion.ElasticNet(classifier=classifier,
                                            confidence=confidence,
                                            learning_rate=epsilon_iter,
                                            beta=beta,
                                            max_iter=max_iter,
                                            batch_size=batch_size,
                                            decision_rule=decision_rule,
                                            verbose=False)

    boundaries = attack.generate(x, y=y_orig_onehot)
    y_sparse = np.argmax(y_orig_onehot, axis=-1)

    y_pred_adv_prob = model.predict(boundaries, batch_size=batch_size)
    y_pred_adv = np.argmax(y_pred_adv_prob, -1)
    is_adv = y_pred_adv == y_sparse
    is_adv = [not b for b in is_adv]
    is_adv = np.array(is_adv)

    if return_probits:
        y_pred_adv = y_pred_adv_prob

    return boundaries, y_pred_adv, is_adv


def batch_flatten(x):
    return np.reshape(x, (x.shape[0], -1))


def linear_integral(model, x, y, res=50, num_class=2):
    baseline = tf.zeros_like(x)
    assert baseline.shape == x.shape

    output_scores = []
    for i in range(1, res + 1):
        t = tf.ones((x.shape[0], 1), dtype=np.float32) * i
        x_in = baseline + (x - baseline) * t / res
        per_output = model(x_in, training=False) * tf.one_hot(
            tf.cast(y, tf.int32), num_class)
        output_of_interest = tf.reduce_sum(per_output, axis=-1)
        output_scores.append(tf.expand_dims(output_of_interest, 0))
    output_scores = tf.concat(output_scores, axis=0)
    return output_scores



def gaussian_volume(model, x, y, K=1000, sigma=0.1):
    output_scores = []
    baseline = tf.random.normal([K, x.shape[1]], mean=0, stddev=sigma, dtype=x.dtype)
    x_len = x.shape[0]
    for i in range(x_len):
        x_in1 = tf.repeat(tf.expand_dims(x[i],0),K,axis=0) 
        x_in2 = x_in1 + baseline
        per_output = model(x_in2, training=False)[:,1] 
        output_of_interest = tf.reduce_mean(per_output, axis=0)-tf.math.reduce_mean(tf.abs(per_output-model(x_in1, training=False)[:,1]), axis=0)
        
        ################Estimating Lipschitz constant method##################
        # m_values = model(x_in2, training=False)[:,1]
        # m_diff = tf.expand_dims(m_values, axis=1) - tf.expand_dims(m_values, axis=0)
        # x_diff = tf.expand_dims(x_in2, axis=1) - tf.expand_dims(x_in2, axis=0)

        # # Calculate the absolute differences in model outputs and norms of input differences
        # abs_m_diff = tf.abs(m_diff)
        # x_norm = tf.norm(x_diff, axis=-1)

        # x_norm = tf.cast(x_norm, dtype=tf.float32)

        # # Calculate the Lipschitz constants for each pair of points
        # gamma = tf.divide(abs_m_diff, x_norm + 1e-8)

        # # Find the maximum Lipschitz constant
        # gamma_hat = tf.reduce_max(gamma)
        # x_norm=tf.norm(x_in1-x_in2, axis=-1)
        # x_norm = tf.cast(x_norm, dtype=tf.float32)
        # # print(m_values.shape, x_norm.shape, gamma_hat.shape)
        # output_of_interest= tf.reduce_mean(m_values - gamma_hat*x_norm, axis=0)
        ######################################################################

        output_scores.append(tf.expand_dims(output_of_interest,0))     

        
    output_scores = tf.concat(output_scores, axis=0)
        
    return output_scores


def saliency(model, x, y, num_class=2):
    x = tf.constant(x)
    with tf.GradientTape() as tape:
        tape.watch(x)
        qoi = model(x, training=False) * tf.one_hot(tf.cast(y, tf.int32),
                                                    num_class)
        qoi = tf.reduce_sum(qoi, axis=-1)
    input_grad = tape.gradient(qoi, x)
    return input_grad


def normalize_attr(attr):
    if len(attr.shape) == 4:
        attr = tf.reduce_mean(attr, axis=-1)
        attr = tf.abs(attr)
        attr /= tf.reduce_max(attr, axis=(1, 2), keepdims=True)
    return attr


def get_pdf(cdf):
    pdf = cdf[:, 1:] - cdf[:, :-1]
    pdf = tf.abs(pdf)
    pdf = pdf / tf.reduce_sum(pdf, -1, keepdims=True)
    return pdf


def unit_vector(x):
    x_flatten = tf.keras.backend.batch_flatten(x)
    return x / l2_norm(x_flatten)


def update_delta(x, delta, grad, step_size, epsilon, p, clamp):
    if p == np.inf:
        delta = delta + step_size * tf.sign(grad)
    elif p == 2:
        delta = delta + step_size * unit_vector(grad)
    delta = delta - x
    delta_flatten = tf.keras.backend.batch_flatten(delta)
    if p == 2:
        norm = l2_norm(delta_flatten)
    elif p == np.inf:
        norm = tf.reduce_max(tf.abs(delta_flatten), axis=-1, keepdims=True)
    elif p == 1:
        norm = tf.reduce_sum(tf.abs(delta_flatten), axis=-1, keepdims=True)
    coeff = tf.clip_by_value(epsilon / norm, 0.0, 1.0)
    delta = tf.reshape(delta_flatten * coeff, [-1] + delta.shape.as_list()[1:])
    delta = delta + x
    delta = clip_with_clamp(delta, clamp)

    return delta


def clip_with_clamp(delta, clamp):
    if isinstance(clamp[0], float) and isinstance(clamp[1], float):
        delta = tf.clip_by_value(delta, clamp[0], clamp[1])
    else:
        delta = tf.minimum(delta, tf.ones_like(delta) * clamp[1])
        delta = tf.maximum(delta, tf.ones_like(delta) * clamp[0])

    return delta


def l2_norm(x):
    return (tf.sqrt(tf.reduce_sum(x**2., axis=-1, keepdims=True)) +
            tf.keras.backend.epsilon())


def batch_sns_search(model,
                      x,
                      y,
                      n_steps=20,
                      max_steps=100,
                      adv_step_size=1e-3,
                      clamp=[0, 1],
                      adv_epsilon=0.1,
                      p=2,
                      num_class=1000,
                      stddev=1e-3,
                      return_probits=False):

    x = tf.constant(x)
    delta = tf.identity(x)

    best_diff = [0] * x.shape[0]
    optimal_adv = x.numpy()


    delta += tf.random.normal(delta.shape, stddev=stddev, dtype=delta.dtype)
    delta = tf.Variable(delta)

    for step in range(max_steps):
        with tf.GradientTape() as tape:
            tape.watch(delta)

            output_scores = linear_integral(model,
                                            delta,
                                            y,
                                            res=n_steps,
                                            num_class=num_class)
            objective = tf.reduce_mean(tf.reduce_sum(output_scores, -1))
        grad = tape.gradient(objective, delta)
        
        
        delta = update_delta(x, delta, grad, adv_step_size, adv_epsilon, p,
                             clamp)

        keep_atk = tf.constant(y) == tf.argmax(model(delta), -1)
        for i in range(len(optimal_adv)):
            if keep_atk[i]:
                optimal_adv[i] = delta[i]

    y_pred = model.predict(optimal_adv)
    if not return_probits:
        y_pred = np.argmax(y_pred, axis=-1)
    return optimal_adv, y_pred, best_diff


def SNS(model, X, Y, batch_size=256,  **kwargs):
    pb = tf.keras.utils.Progbar(target=X.shape[0])
    optimal_adv, y_pred, best_diff = [], [], []
    for i in range(0, X.shape[0], batch_size):
        x = X[i:i + batch_size]
        y = Y[i:i + batch_size]
        opt, y_p, diff = batch_sns_search(model, x, y, **kwargs)
        optimal_adv.append(opt)
        y_pred.append(y_p)
        best_diff.append(diff)

        pb.add(x.shape[0])

    optimal_adv = np.concatenate(optimal_adv, 0)
    y_pred = np.concatenate(y_pred, 0)
    best_diff = np.concatenate(best_diff, 0)

    return optimal_adv, y_pred, best_diff



def batch_trex_search(model,
                      x,
                      y,
                      tau=0.9,
                      n_steps=20,
                      max_steps=100,
                      K=1000,
                      sigma=0.1,
                      adv_step_size=1e-3,
                      clamp=[0, 1],
                      adv_epsilon=0.1,
                      p=2,
                      num_class=1000,
                      stddev=1e-3,
                      return_probits=False):

    x = tf.constant(x)
    delta = tf.identity(x)

    best_diff = [0] * x.shape[0]
    optimal_adv = x.numpy()


    delta = tf.Variable(delta)
    objective = 0
    step=0
    while objective < tau and step < max_steps:
        with tf.GradientTape() as tape:
            tape.watch(delta)
            output_scores = gaussian_volume(model,
                                            delta,
                                            y,
                                            K=K,
                                            sigma=sigma,
                                            num_class=num_class)


            objective = tf.reduce_sum(output_scores, -1)

        
        grad = tape.gradient(objective, delta)       
        delta = update_delta(x, delta, grad, adv_step_size, adv_epsilon, p,
                             clamp)

        keep_atk = tf.constant(y) == tf.argmax(model(delta), -1)
        for i in range(len(optimal_adv)):
            if keep_atk[i]:
                optimal_adv[i] = delta[i]
        objective=objective.numpy()
        step =step+1
    
    y_pred = model.predict(optimal_adv)
    if not return_probits:
        y_pred = np.argmax(y_pred, axis=-1)
    return optimal_adv, y_pred, best_diff





def TreX(model, X, Y,tau, batch_size=256,  **kwargs):
    pb = tf.keras.utils.Progbar(target=X.shape[0])
    optimal_adv, y_pred, best_diff = [], [], []
    for i in range(0, X.shape[0], batch_size):
        x = X[i:i + batch_size]
        y = Y[i:i + batch_size]
        opt, y_p, diff = batch_trex_search(model, x, y,tau, **kwargs)
        optimal_adv.append(opt)
        y_pred.append(y_p)
        best_diff.append(diff)

        pb.add(x.shape[0])

    optimal_adv = np.concatenate(optimal_adv, 0)
    y_pred = np.concatenate(y_pred, 0)
    best_diff = np.concatenate(best_diff, 0)
    return optimal_adv, y_pred, best_diff


def get_boundary_points(model,
                        x,
                        y_sparse,
                        batch_size=64,
                        pipeline=['pgd'],
                        search_range=['local', 'l2', 0.3, None, 100],
                        clamp=[0, 1],
                        backend='pytorch',
                        confidence=1.0,
                        **kwargs):
    """Find nearby boundary points by running adversarial attacks

    Reference: Boundary Attributions for Normal (Vector) Explanations https://arxiv.org/pdf/2103.11257.pdf
    
    https://github.com/zifanw/boundary

    Args:
        model (tf.models.Model or torch.nn.Module): tf.keras model or pytorch model
        x (np.ndarray): Benigh inputs
        y_onehot (np.ndarray): One-hot labels for the benign inputs
        batch_size (int, optional): Batch size. Defaults to 64.
        pipeline (list, optional): A list of adversarial attacks used to find nearby boundaries. Defaults to ['pgd'].
        search_range (list, optional): Parameters shared by all adversarial attacks. Defaults to ['local', 'l2', 0.3, None, 100].
        clamp (list, optional): Data range. Defaults to [0, 1].
        backend (str, optional): Deep learning frame work. It is either 'tf.keras' or 'pytorch'. Defaults to 'pytorch'.
        device (str, optional): GPU device to run the attack. This only matters if the backend is 'pytorch'. Defaults to 'cuda:0'.
    Returns:
        (np.ndarray, np.ndarray): Points on the closest boundary and distances
    """

    if not isinstance(clamp[0], float):
        clamp = [np.min(clamp[0]), np.max(clamp[1])]

    if backend == 'tf.keras':
        fmodel = fb.TensorFlowModel(model, bounds=(clamp[0], clamp[1]))
        x = tf.constant(x, dtype=tf.float32)
        y_sparse = tf.constant(y_sparse, dtype=tf.int32)
        if isinstance(search_range[2], float):
            if search_range[1] == 'l2':
                attack = fb.attacks.L2PGD(
                    rel_stepsize=search_range[3] if search_range[3]
                    is not None else 2 * search_range[2] / search_range[4],
                    steps=search_range[4])
            else:
                attack = fb.attacks.LinfPGD(
                    rel_stepsize=search_range[3] if search_range[3]
                    is not None else 2 * search_range[2] / search_range[4],
                    steps=search_range[4])

            boundary_points = []
            success = 0
            for i in range(0, x.shape[0], batch_size):
                batch_x = x[i:i + batch_size]
                batch_y = y_sparse[i:i + batch_size]

                _, batch_boundary_points, batch_success = attack(
                    fmodel, batch_x, batch_y, epsilons=[search_range[2]])

                boundary_points.append(
                    batch_boundary_points[0].unsqueeze(0))
                success += np.sum(batch_success)

            boundary_points = tf.concat(boundary_points, axis=0)
            success /= x.shape[0]


        elif isinstance(search_range[2], (list, np.ndarray)):
            boundary_points = []
            success = 0.
            for i in range(0, x.shape[0], batch_size):

                batch_x = x[i:i + batch_size]
                batch_y = y_sparse[i:i + batch_size]

                batch_boundary_points = None
                batch_success = None

                for eps in search_range[2]:
                    if search_range[1] == 'l2':
                        attack = fb.attacks.L2PGD(
                            rel_stepsize=search_range[3] if search_range[3]
                            is not None else 2 * eps / search_range[4],
                            steps=search_range[4])
                    else:
                        attack = fb.attacks.LinfPGD(
                            rel_stepsize=search_range[3] if search_range[3]
                            is not None else 2 * eps / search_range[4],
                            steps=search_range[4])

                    _, c_boundary_points, c_success = attack(
                        fmodel, batch_x, batch_y, epsilons=[eps])
                    c_boundary_points = c_boundary_points[0].numpy()
                    c_success = tf.cast(c_success[0], tf.int32).numpy()


                    if batch_boundary_points is None:
                        batch_boundary_points = c_boundary_points
                        batch_success = c_success
                    else:
                        for i in range(batch_boundary_points.shape[0]):
                            if batch_success[i] == 0 and c_success[i] == 1:
                                batch_boundary_points[
                                    i] = c_boundary_points[i]
                                batch_success[i] = c_success[i]

                boundary_points.append(batch_boundary_points)
                success += np.sum(batch_success)

            boundary_points = tf.concat(boundary_points, axis=0)
            success /= x.shape[0]

        else:
            raise TypeError(
                f"Expecting eps as float or list, but got {type(search_range[3])}"
            )

        x = x.numpy()
        y_sparse = y_sparse.numpy()
        boundary_points = boundary_points.numpy()


    return convert_to_numpy(boundary_points)


def convert_to_numpy(x):
    """
    Reference: Boundary Attributions for Normal (Vector) Explanations https://arxiv.org/pdf/2103.11257.pdf
    
    https://github.com/zifanw/boundary
    """

    if not isinstance(x, np.ndarray):
        return x.numpy()
    else:
        return x


def remove_None(L):
    m = []
    for l in L:
        if l is not None:
            m.append(l)
    return m


def search_z_adv(vae,
                 classifier,
                 x,
                 latent_representations,
                 y_sparse,
                 epsilon=0.141,
                 steps=100,
                 num_samples=100,
                 direction='random',
                 p=2,
                 batch_size=128,
                 transform=None,
                 return_probits=True,
                 **kwargs):
    def get_random_delta(d, z_0, num_samples, epsilon, seed=None):
        if d == 'random':
            # create an intial direction
            delta = tf.random.normal((num_samples, ) + z_0.shape, seed=seed)
            delta /= tf.norm(tf.keras.backend.batch_flatten(delta),
                             ord=p)  # unit vector with length = 1
            random_norms = tf.random.uniform((num_samples, 1),
                                             minval=0,
                                             maxval=epsilon,
                                             seed=seed)
            delta *= random_norms
        else:
            raise NotImplementedError(
                f"The {d} direction has not been impletmented yet.")

        return tf.ones([num_samples, 1]) * z + delta


    reconst_x = vae.decode(latent_representations, batch_size=batch_size)
    reconst_x_pred = np.argmax(
        classifier.predict(reconst_x, batch_size=batch_size), -1)

    pb = tf.keras.utils.Progbar(target=latent_representations.shape[0],
                                stateful_metrics=['ave_iter', 'success_rate'])
    counterfactuals = []
    for i, z in enumerate(latent_representations):

        if isinstance(z, np.ndarray):
            z = tf.constant(z)

        # If the prediction is already different, dont bother searching
        if reconst_x_pred[i] != y_sparse[i]:
            counterfactuals.append(reconst_x[i][None, :])
            j = 0

        else:
            reconst = reconst_x[:1].copy()
            for j in range(steps):
                batch_z_candidates = get_random_delta(direction,
                                                      z,
                                                      num_samples,
                                                      epsilon)
                reconst_x_from_z_tilde = vae.decode(batch_z_candidates,
                                                    batch_size=batch_size)

                if transform is not None:

                    reconst_x_from_z_tilde = transform(reconst_x_from_z_tilde)

                new_pred = np.argmax(
                    classifier.predict(reconst_x_from_z_tilde,
                                       batch_size=batch_size), -1)
                valid_idx = new_pred != y_sparse[i]
                num_valid = np.sum(valid_idx)
                if num_valid > 0:
                    valid_reconst = reconst_x_from_z_tilde[valid_idx]

                    diff = valid_reconst - x[i]
                    flatten_diff = tf.keras.backend.batch_flatten(diff)
                    flatten_diff_norm = tf.norm(flatten_diff, axis=-1)
                    closest_id = tf.argmin(flatten_diff_norm)
                    reconst = valid_reconst[closest_id][None, :]
                    break

            counterfactuals.append(reconst)

        success = (np.sum(counterfactuals[-1]) > 0) * 1.0
        pb.add(1, [('avg_iter', j), ('success_rate', success)])

    counterfactuals = np.vstack(counterfactuals)
    counterfactuals_pred_logits = classifier.predict(counterfactuals,
                                                     batch_size=batch_size)
    is_adv = y_sparse != np.argmax(counterfactuals_pred_logits, axis=-1)
    adv_x = counterfactuals[is_adv]
    y_pred_adv = counterfactuals_pred_logits[is_adv]
    if not return_probits:
        y_pred_adv = np.argmax(y_pred_adv, axis=-1)

    return adv_x, y_pred_adv, is_adv
