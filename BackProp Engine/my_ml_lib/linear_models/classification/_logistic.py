# my_ml_lib/linear_models/classification/_logistic.py
import numpy as np

class LogisticRegression:
    """
    L2-Regularized Logistic Regression classifier using IRLS (Newton-Raphson).
    """
    def __init__(self, alpha=0.0, max_iter=100, tol=1e-10, fit_intercept=True):
        """
        Args:
            alpha (float): L2 regularization strength.
            max_iter (int): Maximum number of iterations for IRLS.
            tol (float): Tolerance for stopping criterion (change in weights).
            fit_intercept (bool): Whether to add a bias term.
        """
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.w_ = None # Learned weights (includes intercept if fit_intercept is True)

    def _sigmoid(self, z):
        """Numerically stable sigmoid function."""
        # Clip z to avoid overflow/underflow in exp
        z_clipped = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def fit(self, X , y):
        """
        Fit the L2-regularized logistic regression model using IRLS.

        Args:
            X (np.ndarray): Training data, shape (n_samples, n_features).
            y (np.ndarray): Target values (0 or 1), shape (n_samples,).
        """
        n_samples, n_features = X.shape
        w_old = None # Keep track of previous weights for convergence check

        # --- TODO: Step 1 - Add intercept term (bias) ---
        # If self.fit_intercept is True:
        #   - Augment X with a column of ones (e.g., using np.hstack). Assign to X_aug.
        #   - Initialize self.w_ as a zero vector of size (n_features + 1).
        # Else (if self.fit_intercept is False):
        #   - Use X directly as X_aug.
        #   - Initialize self.w_ as a zero vector of size n_features.
        if self.fit_intercept is True:
            X_aug = np.hstack(( np.ones(len(X)).reshape(-1,1) , X))
        else:
            X_aug = X
        self.w_ = np.zeros((X_aug.shape[1] , 1))
    
        #print("TODO: Implement intercept handling and weight initialization in fit()")
        # Placeholder initialization

        # --- TODO: Step 2 - Regularization setup for IRLS ---
        # Create the regularization matrix: alpha * Identity
        # Hint: Use np.eye(size).
        # If fitting an intercept, make sure the first element (corresponding to bias) is 0.
        #print("TODO: Implement regularization matrix setup in fit()")
        reg_matrix = np.zeros((self.w_.shape[0], self.w_.shape[0])) # Placeholder
        if self.fit_intercept is True:   
            reg_matrix[1:,1:] += np.eye(reg_matrix[1:,1:].shape[0]) * self.alpha
        else:
            reg_matrix += np.eye(reg_matrix.shape[0]) * self.alpha


        # --- TODO: Step 3 - IRLS Iterations ---
        #print("TODO: Implement IRLS loop in fit()")
        for i in range(self.max_iter):
            w_old = self.w_.copy() # Store weights from previous iteration

            # --- TODO: Step 3a - Calculate predictions (h) ---
            # Calculate the linear combination: 
            # Calculate the predicted probabilities: 
            h = np.zeros((n_samples,1)) # Placeholder

            h = self._sigmoid(X_aug@self.w_).reshape(-1,1)

            # --- TODO: Step 3b - Calculate gradient (∇L) ---
            gradient = np.zeros_like(self.w_) # Placeholder
            gradient = X_aug.T @ (h - y) + self.alpha * self.w_

            # --- TODO: Step 3c - Calculate weight matrix R (diagonal) ---
            # Ensure diagonal elements are not too close to zero (e.g., np.maximum(r_diag, 1e-10))
            # R = np.diag(r_diag)
            R = np.eye(n_samples) # Placeholder
            r_diag = h * (1-h)
            R = np.diag(np.maximum(r_diag.flatten() , 1e-10))


            # --- TODO: Step 3d - Calculate Hessian (H) ---
            # H = X_aug^T @ R @ X_aug + reg_matrix
            hessian = np.eye(self.w_.shape[0]) # Placeholder
            hessian = X_aug.T @ R @ X_aug + reg_matrix

            # --- TODO: Step 3e - Update weights ---
            # Solve the linear system H @ delta_w = gradient for delta_w
            # Hint: Use np.linalg.solve(hessian, gradient). Handle potential errors (np.linalg.LinAlgError)
            #       by possibly using np.linalg.pinv(hessian) @ gradient as a fallback, but print a warning.
            # Update weights: self.w_ = w_old - delta_w


            delta_w = np.zeros_like(self.w_) # Placeholder
            try:
                delta_w = np.linalg.solve(hessian , gradient)
            except:
                delta_w = np.linalg.pinv(hessian) @ gradient
                
            self.w_ = w_old - delta_w # Placeholder update

            # --- TODO: Step 3f - Check for convergence ---
            # Calculate the norm of the change in weights: weight_change = np.linalg.norm(self.w_ - w_old)
            # If weight_change < self.tol: break the loop
            # Placeholder
            weight_change = np.linalg.norm(delta_w)
            if weight_change < self.tol:
                print(f"Converged after {i+1} iterations.") # Optional convergence message
                break


        # Optional: Add a warning if the loop finished without converging
        # else: # Runs if loop completes without break
        #     print(f"Warning: IRLS did not converge within {self.max_iter} iterations.")

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.

        Args:
            X (np.ndarray): Samples, shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted probabilities [P(y=0|X), P(y=1|X)], shape (n_samples, 2).
        """
        if self.w_ is None:
            raise RuntimeError("Model is not fitted yet.")

        # --- TODO: Step 1 - Augment X if fitting intercept ---
        # If self.fit_intercept was True during fit, augment X here too.
        #print("TODO: Implement intercept handling in predict_proba()")
        X_aug = X # Placeholder
        if self.fit_intercept is True:
            X_aug = np.hstack(( np.ones(len(X)).reshape(-1,1) , X))


        # --- TODO: Step 2 - Calculate P(y=1 | X) ---
        # Calculate linear combination: 
        # Calculate probability: 
        prob_y1 = np.zeros(X.shape[0]) # Placeholder
        prob_y1 = X_aug @ self.w_
        prob_y1 = self._sigmoid(prob_y1) 


        # --- TODO: Step 3 - Calculate P(y=0 | X) ---
       
        prob_y0 = np.ones((X.shape[0],1)) # Placeholder
        prob_y0 = 1 - prob_y1


        # --- TODO: Step 4 - Stack probabilities ---
        # Return shape should be (n_samples, 2)
        # Hint: Use np.vstack([prob_y0, prob_y1]).T or np.column_stack([prob_y0, prob_y1])
        #print("TODO: Implement stacking probabilities in predict_proba()")
        
        return np.hstack((prob_y0 , prob_y1)) # Placeholder

    def predict(self, X):
        """
        Predict class labels (0 or 1) for samples in X.

        Args:
            X (np.ndarray): Samples, shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels (0 or 1), shape (n_samples,).
        """
        # --- TODO: Step 1 - Get P(y=1 | X) ---
        # Call self.predict_proba(X) and select the column corresponding to class 1.
        #print("TODO: Implement predict using predict_proba")
        probabilities_y1 = np.zeros(X.shape[0]) # Placeholder
        probabilities_y1 = self.predict_proba(X)[:,1]


        # --- TODO: Step 2 - Apply threshold ---
        # Return 1 if probability >= 0.5, else 0.
        # Hint: Use boolean comparison and .astype(int)

        return (probabilities_y1 >= 0.5).astype(int) # Placeholder

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels."""
        # This method should work once predict() is implemented correctly.

        return np.mean(self.predict(X) == y.flatten())
    
# X = np.array([[0,0,0],[0,0,0],[1,1,1],[1,1,1]])
# y = np.array([0,0,1,1]).reshape(-1,1)

# clf = LogisticRegression(alpha=0.1, max_iter=10, fit_intercept=True)
# clf.fit(X, y)
# print("Weights:\n", clf.w_)
# print("Predictions:", clf.predict(X))
# print("Accuracy:", clf.score(X,y))



