import mlx.nn as nn
import mlx.core as mx


class Net(nn.Module):
    def __init__(self, channels, dim, classes):
        super(Net, self).__init__()

        fully_connected_input_size = self.calculate_l_input_size(dim, 100, 3)

        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=20, kernel_size=(5, 5), padding=2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5), padding=2)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv3 = nn.Conv2d(in_channels=50, out_channels=100, kernel_size=(5, 5), padding=2)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
        self.l1 = nn.Linear(input_dims=fully_connected_input_size, output_dims=fully_connected_input_size // 2)
        self.dropout1 = nn.Dropout(0.25)
        self.relu3 = nn.ReLU()
        self.l2 = nn.Linear(input_dims=fully_connected_input_size // 2, output_dims=classes)
        self.dropout2 = nn.Dropout(0.5)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        x = mx.flatten(x, 1)

        x = self.l1(x)
        x = self.dropout1(x)
        x = self.relu3(x)

        x = self.l2(x)
        x = self.dropout2(x)

        output = nn.log_softmax(x, axis=1)
        return output
    
    def calculate_l_input_size(self, square_dim: int, final_out_channels: int, num_pools: int = 2) -> int:
        """
        Calculate the input feature dimension for the first fully-connected layer
        based on the input image dimensionality.

        Args:
            square_dim (int): The width and height of the square input images.
            final_out_channels (int): The number of output channels from the last convolutional layer.
            num_pools (int, optional): The number of pooling layers. Defaults to 3.

        Returns:
            int: The number of input features for the first fully-connected layer.
        """
        final_size = square_dim // (2 ** num_pools)
        fc_input_features = final_size ** 2 * final_out_channels
        return fc_input_features
