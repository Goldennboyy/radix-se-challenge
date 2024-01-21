# Radix - SE Hiring Challenge

Hiring challenge for the Software Engineer applicants at Radix.


## Challenge description 🔖

The goal of this challenge is to build and an AI model, found in this repository.

This repository implements a simple [scikit-learn](https://scikit-learn.org/stable/) model that predicts movie genres based on a given synopsis. In the model folder, you can find the saved model (pickle) as also the python functions created to save, load and make predictions from the model. The data used can be downloaded via the code in the data folder.

To succeed the hiring challenge, you must implement the following:
* Create fastAPI endpoints to make predictions with the model (`/genres/predict`).
* Create a Docker image to create a container that runs this API locally

Any other best practice implementations are optional, but could be assessed during the technical interview that follows after this hiring challenge. We will also ask you which service you'd use on Azure to deploy it and the steps you'd undertake to deploy. This challenge leaves you the freedom to create a solution that you think would suit our needs, so make sure you can motivate your design decisions!


## Getting started 🚀

### GitHub setup

1. Clone this repository
2. Ensure that your code is private
3. Invite `RadixChallenge` (`challenge@radix.ai`) as a collaborator to your repository

### Local setup

In an environment of choice, install the dependencies using `requirements.txt` or by running `pip install .`. Once done, you can run our custom `invoke` commands, found in the `tasks/` folder:
- `invoke --list` to list all the invoke commands (also specified in `tasks/tasks.py`)
- `invoke lint` to lint this package
- `invoke test` to test this package
- `invoke serve` to serve the API
- `invoke run` to train and evaluate a model

If all four of the invoke commands run, then you're good to go!


## Questions? 🤨

We would love to help you with the challenge, but unfortunately we can't. 😉 That being said, if you find a bug or have troubles setting up your environment, we're happy to help you at [challenge@radix.ai](mailto:challenge@radix.ai)! 


## Good luck! 🚀

![Radix](https://media-exp1.licdn.com/dms/image/C4D0BAQH7Tz0v-I5l1g/company-logo_200_200/0/1569474265425?e=2159024400&v=beta&t=TiThE4sfIokh1WfVN04aM7qgcI__285c2xkE0bEzCtA)
