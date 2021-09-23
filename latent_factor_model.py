import numpy as np

class lf_model:

    def __init__(self, dataset):
        n_latent_factors = 2
        self.ratings = dataset.copy()
        # Initialise as random values
        self.latent_user_preferences = np.random.random((self.ratings.shape[0], n_latent_factors))
        self.latent_item_features = np.random.random((self.ratings.shape[1], n_latent_factors))

    def predict_rating(self, user_id, item_id):
        """ Predict a rating given a user_id and an item_id.
        """
        user_preference = self.latent_user_preferences[user_id]
        item_feature = self.latent_item_features[item_id]
        return user_preference.dot(item_feature)


    def sgd_optimizer(self, user_id, item_id, err, alpha=0.0001):
        #print(err)
        self.latent_user_preferences[user_id] -= alpha * err * self.latent_item_features[item_id]
        self.latent_item_features[item_id] -= alpha * err * self.latent_user_preferences[user_id]


    def train(self, epochs):
        """ Iterate over all users and all items and train for 
          a certain number of iterations
        """
        mse_history = []
        for iteration in range(epochs):
            error = []
            for user_id in range(self.latent_user_preferences.shape[0]):
                for item_id in range(self.latent_item_features.shape[0]):
                    rating = self.ratings[user_id, item_id]
                    if not np.isnan(rating):
                        predicted_rating = self.predict_rating(user_id, item_id)
                        err =  predicted_rating - rating
                        error.append(err)
                        self.sgd_optimizer(user_id, item_id, err)
            mse = (np.array(error) ** 2).mean()   
            if (iteration % 2) == 0:
                print('Iteration %d/%d:\tMSE=%.6f' % (iteration, epochs, mse))
                mse_history.append(mse)
        return mse_history