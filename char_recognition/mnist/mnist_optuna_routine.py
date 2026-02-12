import optuna
import torch
from pandas import DataFrame
from torch.utils.data import TensorDataset

from char_recognition.data_loader import load_image_data, ImageDataType
from char_recognition.model_training import count_parameters, model_train_best_test_accuracy
from vpnet import Hermite2DXMSystem, VPNet, VPTypes, VPLoss

model_learning_rate = 0.001
op_batch_size = 64
n_epochs = 50
loss_vp_penalty = 0.001

train_data: TensorDataset = load_image_data(data_type = ImageDataType.MNIST_TRAIN)
valid_data, test_data = load_image_data(data_type = ImageDataType.MNIST_TEST, part=0.5)

def optuna_mnist_objective(trial: optuna.Trial) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_hermite_directional_coeffs = trial.suggest_int("num_hermite_directional_coeffs", 1, 100)
    hidden_layer_neuron_count = trial.suggest_int("hidden_layer_neuron_count", 1, 100)


    fn_system = Hermite2DXMSystem(28, 28, num_hermite_directional_coeffs)
    initial_params = [0.3, 0.2, 0.4, 0.05]


    curr_mnist_vpnet_model = VPNet(
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


    if count_parameters(curr_mnist_vpnet_model) > 2000:
        raise optuna.exceptions.TrialPruned()

    print(f"Trial started, hyperparameters: num_hermite_directional_coeffs={num_hermite_directional_coeffs}, hidden_layer_neuron_count={hidden_layer_neuron_count}, loss_vp_penalty={loss_vp_penalty}")

    loss_criterion = VPLoss(torch.nn.CrossEntropyLoss(), vp_penalty=loss_vp_penalty)
    optimizer = torch.optim.Adam(curr_mnist_vpnet_model.parameters(), lr=model_learning_rate)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=op_batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=op_batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=op_batch_size, shuffle=False)

    total_acc = model_train_best_test_accuracy(curr_mnist_vpnet_model, train_loader, valid_loader, test_loader, n_epochs, optimizer, loss_criterion)

    return total_acc


def main():
    print("Starting optuna HPO study:")

    search_space = {
        "num_hermite_directional_coeffs": list(range(3, 10 + 1)),
        "hidden_layer_neuron_count": list(range(5, 100 + 1, 5)),
    }

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.GridSampler(search_space))
    study.optimize(optuna_mnist_objective, n_trials=1000, catch=[Exception])

    print("Finished optuna HPO study, printing results:")
    print(f"Best found parameters: {study.best_params}")
    print(f"Best value: {study.best_value}")

    df: DataFrame = study.trials_dataframe()
    df = df.drop(["datetime_start", "datetime_complete", "duration"])

    print("Study dataframe, saved to optuna_results.csv:")
    print(df)

    df.to_csv("../optuna_results.csv")

if __name__ == "__main__":
    main()
