import os
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter

ckpt_converter = 'checkpoints/converter'
device="cuda:0" if torch.cuda.is_available() else "cpu"
output_dir = 'outputs'

tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

os.makedirs(output_dir, exist_ok=True)

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv() 


client = OpenAI(api_key="OPENAI_API_KEY")

response = client.audio.speech.create(
    model="tts-1",
    voice="nova",
    input="This audio will be used to extract the base speaker tone color embedding. " + \
        "Typically a very short audio should be sufficient, but increasing the audio " + \
        "length will also improve the output audio quality."
)

response.stream_to_file(f"{output_dir}/openai_source_output.mp3")

base_speaker = f"{output_dir}/openai_source_output.mp3"
source_se, audio_name = se_extractor.get_se(base_speaker, tone_color_converter, vad=True)

reference_speaker = 'resources/example_reference.mp3' # This is the voice you want to clone
target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, vad=True)

text = [
    "بچے ہمیشہ اپنے والدین سے ایک ہی کہانی بار بار پوچھتے ہیں۔ کہانی کو دہرانے سے بچے تھوڑی دیر بعد کہانی کے ماہر بن جاتے ہیں۔"
]
src_path = f'{output_dir}/tmp.wav'

for i, t in enumerate(text):

    response = client.audio.speech.create(
        model="tts-1-hd",
        voice="nova",
        input=t,
    )

    response.stream_to_file(src_path)

    save_path = f'{output_dir}/output_crosslingual_{i}.wav'

    # Run the tone color converter
    encode_message = "@MyShell"
    tone_color_converter.convert(
        audio_src_path=src_path, 
        src_se=source_se, 
        tgt_se=target_se, 
        output_path=save_path,
        message=encode_message)
