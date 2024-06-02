import sys
import time
import argparse

from stable_baselines3.common.env_checker import check_env

from environments.pogodude import PogoEnv
from utils import BASE_MODEL_NAME, IntRef, TrainingCallback


def train_with_sb3_agent(
    version: int,
    total_timesteps: int,
    learning_rate: float,
):
    """
    Trains the provided saved agent (or initializes a new agent if the provided model name doesn't exist)
    within the pogodude environment for `total_timesteps` time steps. Saves the trained model to a test
    directory once it finishes, or if it receives a KeyboardInterrupt.

    Parameters
    ----------
    - `version`: int - the version of the environment and model to train
    - `total_timesteps`: int - the total number of timesteps to train the model for
    - `learning_rate`: float - the learning rate for training the model
    """
    
    retries = 0
    max_retries = 10

    steptracker = IntRef()
    callback = TrainingCallback(
        model_name=f"{BASE_MODEL_NAME}_{version}",
        steps_per_checkpoint=10000,
        steptracker=steptracker,
    )

    while retries < max_retries:

        env = PogoEnv(version, render_mode=("rgb_array"))
        check_env(env)

        model = callback.load_model(env, learning_rate, retries > 0)

        try:
            print("Training model")
            model.learn(
                total_timesteps=total_timesteps - steptracker(),
                callback=callback,
                progress_bar=True
            )
            break

        except KeyboardInterrupt:
            print("Interrupted by user, saving model")
            callback.save_model()
            print("Done.")
            break

        except Exception as e:
            print(f"Error during training: {e}")
            print("Reloading model and continuing training")
            retries += 1
            time.sleep(1)


        finally:
            env.close()
        
    if retries >= max_retries:
        print("Too many retries, giving up")
        callback.save_model(give_up=True)


def run_simulation_with_sb3_agent(
    version: int,
    model_dir: str,
):
    """
    Runs pogodude environment using a provided saved agent, and applies the agent's actions
    to the environment without having the agent learn. For demonstration/testing purposes.

    Parameters
    ----------
    `version`: int - the version of the environment and model to run
    `model_dir`: str - the directory where the saved model is located
    """

    env = PogoEnv(version, render_mode="human")
    check_env(env)

    steptracker = IntRef()
    callback = TrainingCallback(
        model_name=f"{BASE_MODEL_NAME}_{version}",
        training_mode=False,
        model_dir=model_dir,
        steptracker=steptracker,
    )

    try:
        model = callback.load_model(env)
        print("Using specified model")
    except FileNotFoundError:
        print("Specified model not found")
        sys.exit(1)

    vec_env = model.get_env()
    assert vec_env is not None
    obs = vec_env.reset()

    while True:
        try:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, _, _ = vec_env.step(action)
            vec_env.render("human")
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Run or train an agent to control an pogo robot"""
    )
    group1 = parser.add_argument_group("Functional arguments (mutually exclusive)")
    group1e = group1.add_mutually_exclusive_group(required=True)
    group1e.add_argument(
        "-t",
        "--train",
        action="store_true",
        help="train a new/existing model in test_models/",
    )
    group1e.add_argument(
        "-r",
        "--run",
        action="store_true",
        help="run a model",
    )
    group2 = parser.add_argument_group("Training and running arguments")
    group2.add_argument(
        "-v",
        "--version",
        type=int,
        help="version of the model to run (e.g. '1' or '2')",
    )
    group3 = parser.add_argument_group("Running arguments")
    group3.add_argument(
        "-s",
        "--saved-dir",
        action="store_true",
        help="whether the model will be/is in the saved_models/ directory (otherwise test_models/)",
    )
    group4 = parser.add_argument_group("Training arguments")
    group4.add_argument(
        "-T",
        "--total-timesteps",
        type=int,
        default=1_000_000,
        help="total number of timesteps to train the model for (default: 1,000,000)",
    )
    group4.add_argument(
        "-l",
        "--learning-rate",
        type=float,
        default=0.0003,
        help="learning rate for training the model (default: 0.0003)",
    )
    args = parser.parse_args()

    if args.train:
        if args.version is None:
            parser.error("argument -t/--train requires -v/--version")
        if args.saved_dir:
            parser.error(
                "argument -t/--train cannot be used with -s/--saved-dir (cannot train a model in the saved_models/ directory)"
            )
        train_with_sb3_agent(
            version=args.version,
            total_timesteps=args.total_timesteps,
            learning_rate=args.learning_rate,
        )
    elif args.run:
        if args.version is None:
            parser.error("argument -r/--run requires -v/--version")
        run_simulation_with_sb3_agent(
            version=args.version,
            model_dir="saved_models" if args.saved_dir else "test_models",
        )
    else:
        parser.print_help()
        exit(0)
