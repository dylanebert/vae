class Params:
    def __init__(self):
        self.image_size = 32
        self.num_channels = 3
        self.hidden_size = self.image_size * self.image_size * self.num_channels
        self.latent_size = self.image_size * self.image_size * self.num_channels
        self.learning_rate = .001
        self.batch_size = 100
