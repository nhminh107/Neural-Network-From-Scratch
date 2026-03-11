from Data import DataMNIST
from Model import Model
from Pipeline import Pipeline

def main():

    # load data
    data = DataMNIST()
    data.normalize_data()

    # create model
    model = Model()
    model.init_params()

    # pipeline
    pipeline = Pipeline(data, model)

    # train
    pipeline.train(epoch=15)

    # evaluate
    acc = pipeline.accuracy()
    print("Test Accuracy:", acc)


if __name__ == "__main__":
    main()