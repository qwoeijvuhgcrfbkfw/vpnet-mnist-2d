import torch

from char_recognition.data_loader import load_image_data, ImageDataType
from char_recognition.model_training import model_train_best_test_accuracy
from vpnet import Hermite2DXMSystem, VPNet, VPTypes, VPLoss, FCNN

model_learning_rate = 0.001
op_batch_size = 64
n_epochs = 50
loss_vp_penalty = 0.001

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = load_image_data(ImageDataType.MNIST_TRAIN)
    valid_data, test_data = load_image_data(ImageDataType.MNIST_TEST, part=0.5)

    num_hermite_directional_coeffs = 6
    hidden_layer_neuron_count = 30

    fn_system = Hermite2DXMSystem(28, 28, num_hermite_directional_coeffs)
    initial_params = [0.3, 0.2, 0.4, 0.05]

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=op_batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=op_batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=op_batch_size, shuffle=False)

    vpnet_model = VPNet(
        fn_system.num_samples,
        1,

        fn_system.num_coeffs,
        VPTypes.FEATURES,
        initial_params,
        fn_system,
        [hidden_layer_neuron_count],
        10,
        device=device
    )

    loss_criterion = VPLoss(torch.nn.CrossEntropyLoss(), vp_penalty=loss_vp_penalty)
    optimizer = torch.optim.Adam(vpnet_model.parameters(), lr=model_learning_rate, weight_decay=1e-4)

    model_train_best_test_accuracy(vpnet_model, train_loader, valid_loader, test_loader, n_epochs, optimizer, loss_criterion)

if __name__ == '__main__':
    main()