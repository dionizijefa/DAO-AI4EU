from pathlib import Path
import grpc
import model_pb2
import model_pb2_grpc
import click


@click.command()
@click.option('--port', help='Enter a port')
def main(port):
    port_addr = port
    root = Path(__file__).resolve().parents[0].absolute()

    # open a gRPC channel
    channel = grpc.insecure_channel(port_addr)

    # create a stub (client)
    stub = model_pb2_grpc.PredictStub(channel)

    # input for the model
    smiles = "O=C3CCC(N2C(=O)c1ccccc1C2=O)C(=O)N3"

    # create a valid request message
    request_prediction = model_pb2.Input(input_smiles=smiles)
    response_prediction = stub.predict(request_prediction)
    complementary_model = stub.complementary_model(request_prediction)
    explanation = stub.explain(request_prediction)

    print('\n')
    label = 'WITHDRAWN' if response_prediction.predicted_class == 1 else 'APPROVED'
    print('Predicted class is: {}'.format(label))
    print('Predicted probability is: {}'.format(response_prediction.withdrawn_probability))
    print('Approved p-value is: {}'.format(response_prediction.pvalue_approved))
    print('Withdrawn p-value is: {}'.format(response_prediction.pvalue_withdrawn))
    print('')
    print('\n')
    print('Prediction explanation using GNN explainer')
    print('Feature importance - Higher values are better: ')
    for keys, values in explanation.feature_importance.items():
        print(keys, values)
    print('\n')
    print('Complementary model prediction')
    complementary_label = 'WITHDRAWN' if complementary_model.complementary_prediction[
                                             'predicted_class'] == 1 else 'APPROVED'
    print('Predicted class is: {}'.format(complementary_label))
    print('Predicted probability is: {}'.format(complementary_model.complementary_prediction['withdrawn_probability']))
    print('Drug properties for complementary model')
    for keys, values in complementary_model.complementary_tasks.items():
        print(keys, values)
    print('SHAP values for each drug property:')
    print('SHAP value describes how the variable affects the log-odds')
    for keys, values in complementary_model.shap_values.items():
        print(keys, values)

    # Ouput the SVG of the molecules with colored bonds according to the edge importance in GNN explainer
    vis_path = Path(root / 'visualizations')
    vis_path.mkdir(parents=True, exist_ok=True)
    with open(vis_path / '{}.svg'.format(smiles), 'w') as f:
        f.write(explanation.bond_importance)


if __name__ == "__main__":
    main()
