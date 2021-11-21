import grpc
from concurrent import futures
import time
import model_pb2 as model_pb2
import model_pb2_grpc as model_pb2_grpc
from withdrawn.dao_service import predict, complementary_model, explain

port = 8061


# create a class to define the server functions, derived from
class PredictServicer(model_pb2_grpc.PredictServicer):
    def predict(self, request, context):
        # define the buffer of the response :
        response = model_pb2.Prediction()
        # get the value of the response by calling the desired function :
        response.predicted_class, response.withdrawn_probability, response.pvalue_approved, response.pvalue_withdrawn \
            = predict(request.input_smiles)
        return response

    def complementary_model(self, request, context):
        # define the buffer of the response :
        response = model_pb2.ComplementaryModel()
        # get the value of the response by calling the desired function :
        complementary_prediction, complementary_tasks, shap_values = complementary_model(request.input_smiles)
        response.complementary_prediction.update(complementary_prediction)
        response.complementary_tasks.update(complementary_tasks)
        response.shap_values.update(shap_values)
        return response

    def explain(self, request, context):
        # define the buffer of the response :
        response = model_pb2.GNNExplanation()
        # get the value of the response by calling the desired function :
        response.bond_importance, feature_importance = explain(request.input_smiles)
        response.feature_importance.update(feature_importance)
        return response


# create a grpc server :
server = grpc.server(
    futures.ThreadPoolExecutor(max_workers=10)
)
model_pb2_grpc.add_PredictServicer_to_server(PredictServicer(), server)
print("Starting server. Listening on port : " + str(port))
server.add_insecure_port("[::]:{}".format(port))
server.start()
try:
    while True:
        time.sleep(86400)
except KeyboardInterrupt:
    server.stop(0)
