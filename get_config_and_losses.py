import pickle
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-number', type=int, nargs='+',
                    help='an integer for the accumulator')

args = parser.parse_args()

# parameters which can be adjusted
version_number = args.number[0]
bert_filename = ""

# just print the losses from training
losses = pickle.load(open("model_saves/v" + str(version_number) + "/losses.pkl", 'rb'))
for loss in losses:
    #print(loss)
    pass

# begin loading parameters
print("Loading parameters at: " + "model_saves/v" + str(version_number) + "/parameters.pkl")
parameters = pickle.load(open("model_saves/v" + str(version_number) + "/parameters.pkl", 'rb'))
print(parameters)

