import os
import time

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import GymEnv

MODEL_DIRS = {"test": "test_models", "save": "saved_models"}
BASE_MODEL_NAME = "pogodude"

class IntRef:
    def __init__(self):
        self.timesteps = 0

    def __call__(self, timesteps: int | None = None):
        self.timesteps = timesteps or self.timesteps
        return self.timesteps

class TrainingCallback(BaseCallback):
    """Callback for training the model, saving checkpoints, etc."""


    def __init__(
        self,
        model_name: str,
        training_mode: bool = True,
        model_dir: str | None = None,
        steps_per_checkpoint=10000,
        steptracker: IntRef | None = None
    ):
        super().__init__(verbose=0)
        model_dir = (model_dir or MODEL_DIRS["save"]) if not training_mode else MODEL_DIRS["test"]

        self.training_mode = training_mode
        self.model_path = os.path.join(model_dir, f"{model_name}.zip")
        self.replay_buffer_path = os.path.join(model_dir, f"{model_name}_replay_buffer.pkl")
        self.model_checkpoint_path = os.path.join(model_dir, ".checkpoint_model.zip")
        self.replay_buffer_checkpoint_path = os.path.join(model_dir, ".checkpoint_replay_buffer.pkl")
        self.steptracker = steptracker
        self.steps_per_checkpoint = steps_per_checkpoint

    def _on_training_start(self):
        self.__update_steptracker()

    def _on_step(self) -> bool:

        if self.steptracker is not None:
            self.steptracker(self.num_timesteps)

        if self.num_timesteps % self.steps_per_checkpoint != 0:
            return True

        os.makedirs(os.path.dirname(self.model_checkpoint_path), exist_ok=True)
        self.model.save(self.model_checkpoint_path)
        assert isinstance(self.model, SAC)
        self.model.save_replay_buffer(self.replay_buffer_checkpoint_path)

        return True

    def _on_training_end(self) -> None:
        self.save_model()

    def __update_steptracker(self):
        if self.steptracker is not None:
            self.steptracker(self.num_timesteps)

    def save_model(self):
        """Save the model and replay buffer to disk"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save(self.model_path)
        assert isinstance(self.model, SAC)
        self.model.save_replay_buffer(self.replay_buffer_path)

        # Clean up checkpoint files
        if os.path.exists(self.model_checkpoint_path):
            os.remove(self.model_checkpoint_path)
        if os.path.exists(self.replay_buffer_checkpoint_path):
            os.remove(self.replay_buffer_checkpoint_path)

    def load_model(
        self, env: GymEnv,
        learning_rate: float | None = None,
        from_checkpoint=False
    ) -> SAC:
        """
        Load the model and replay buffer from disk, or initialize
        a new model if no saved model is found and in training mode
        """
        try:
            model = SAC.load(
                self.model_checkpoint_path
                    if from_checkpoint
                    else self.model_path,
                env
            )
            print(f"Successfully loaded {'checkpoint' if from_checkpoint else 'saved model'}")

            if self.training_mode:
                try:
                    model.load_replay_buffer(
                        self.replay_buffer_checkpoint_path
                            if from_checkpoint
                            else self.replay_buffer_path
                    )
                    print("Successfully loaded replay buffer; context has been preserved")
                except FileNotFoundError:
                    print("No saved replay buffer; context has been reset")

        except FileNotFoundError:
            if self.training_mode:
                print("No saved model found, training new model")
                assert learning_rate is not None, "Must provide a learning rate when training a new model"
                model = SAC("MlpPolicy", env, verbose=1, learning_rate=learning_rate)
            else:
                raise FileNotFoundError("No saved model found")

        model.set_random_seed(time.time_ns() % 2**32)
        
        return model
