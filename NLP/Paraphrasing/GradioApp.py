import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


tokenizer  = AutoTokenizer.from_pretrained('humarin/chatgpt_paraphraser_on_T5_base')
model = AutoModelForSeq2SeqLM.from_pretrained('humarin/chatgpt_paraphraser_on_T5_base')

def paraphrase(
    text,
    num_beams=5,
    num_beam_groups=5,
    num_return_sequences=5,
    repetition_penalty=10.0,
    diversity_penalty=3.0,
    no_repeat_ngram_size=2,
    temperature=0.7,
    max_length=128
):
    input_ids = tokenizer(
        f'paraphrase: {text}',
        return_tensors="pt", padding="longest",
        max_length=max_length,
        truncation=True,
    ).input_ids
    
    outputs = model.generate(
        input_ids, temperature=temperature, repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams, num_beam_groups=num_beam_groups,
        max_length=max_length, diversity_penalty=diversity_penalty
    )

    res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return res

def fn(
    text,
    num_beams=5,
    num_beam_groups=5,
    num_return_sequences=5,
    repetition_penalty=10.0,
    diversity_penalty=3.0,
    no_repeat_ngram_size=2,
    temperature=0.7,
    max_length=128
):
    res = paraphrase(text, num_beams, num_beam_groups, num_return_sequences, repetition_penalty, diversity_penalty, no_repeat_ngram_size, temperature, max_length)
    result = ''
    for i, item in enumerate(res):
        result += f'{i+1}. {item}\n'
    return result

demo = gr.Interface(
    fn=fn,
    inputs=[
        gr.Textbox(lines=3, placeholder='Enter Text To Paraphrase'),
        gr.Slider(minimum=1, maximum=10, step=1, value=5, label='Num Beams', info='This parameter controls the number of possible next tokens that are considered at each step in the beam search algorithm. A higher value will result in more diverse paraphrases, but may also take longer to generate.'),
        gr.Slider(minimum=1, maximum=10, step=1, value=5, label='Num Beam Groups', info='This parameter controls the number of beams that are run in parallel. A higher value will result in faster generation, but may also result in less diversity.'),
        gr.Slider(minimum=1, maximum=10, step=1, value=5, label='Num Return Sequences', info='This parameter controls the number of sequences that are generated at each step in the beam search algorithm. A higher value will produce more results, but may also take longer to generate.'),
        gr.Slider(minimum=0.6, maximum=20.1, step=0.5, value=10.1, label='Repetition Penalty', info='This parameter controls how much the model penalizes itself for generating repeated words or phrases. A higher value will result in more unique paraphrases, but may also result in less accurate paraphrases.'),
        gr.Slider(minimum=0.6, maximum=20.1, step=0.5, value=3.1, label='Diversity Penalty', info='This parameter controls how much the model penalizes itself for generating paraphrases that are similar to each other. A higher value will result in more diverse paraphrases, but may also result in less accurate paraphrases.'),
        gr.Slider(minimum=1, maximum=10, step=1, value=2, label='No Repeat Ngram Size', info='This parameter controls the size of the n-grams that the model is not allowed to repeat. A higher value will result in more unique paraphrases, but may also result in less accurate paraphrases.'),
        gr.Slider(minimum=0.0, maximum=1, step=0.1, value=0.7, label='Temperature', info='This parameter controls how much the model is allowed to deviate from the original text. A higher value will result in more creative paraphrases, but may also result in less accurate paraphrases.'),
        gr.Slider(minimum=32, maximum=512, step=1, value=128, label='Max Length', info='This parameter controls the maximum length of the generated paraphrase. A higher value will result in more detailed paraphrases, but may also take longer to generate.'),
    ],
    outputs=['text'],
)

demo.launch()