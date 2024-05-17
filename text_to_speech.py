from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
import IPython.display as ipd


models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
    "facebook/fastspeech2-en-ljspeech",
    arg_overrides={"vocoder": "hifigan", "fp16": False}
)

model = models
TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
generator = task.build_generator(model, cfg)
text = "بچے ہمیشہ اپنے والدین سے ایک ہی کہانی بار بار پوچھتے ہیں۔ کہانی کو دہرانے سے بچے تھوڑی دیر بعد کہانی کے ماہر بن جاتے ہیں۔"
sample = TTSHubInterface.get_model_input(task, text)
wav, rate = TTSHubInterface.get_prediction(task, model[0], generator, sample)
ipd.Audio(wav, rate=rate)