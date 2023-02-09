import pathlib
from ds.datasets import create_trainingset
from model_training import BinaryClassClassifier


def main():
    
    training_df = create_trainingset()
    print(f'Number of customers in the trainingset is: {len(training_df)}')

    # Train & Finalize & Save The Model
    classifier = BinaryClassClassifier()
    X, y = classifier.separate_dependent_column(training_df)
    X_train, X_dev, y_train, y_dev = classifier.split_train_dev_sets(X, y)
    X_train = classifier.convert_categorical_to_numerica(X_train)
    X_train = classifier.select_features(X_train)
    X_train = classifier.scale_data(X_train)
    selected_model = classifier.select_model(X_train, y_train)
    tunned_hparams = classifier.tune_hyperparameters(X_train, y_train, selected_model)
    the_model = classifier.finalize_model(selected_model, tunned_hparams, X_train, y_train)


    # Test The Model
    X_dev = classifier.convert_categorical_to_numerica(X_dev)
    X_dev = classifier.select_features(X_dev)
    X_dev = classifier.scale_data(X_dev)
    classifier.apply_model(model=the_model, X_dev=X_dev, y_dev=y_dev)
    

if __name__ == "__main__":
    main() 