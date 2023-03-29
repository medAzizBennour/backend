import os
import tempfile
import flask
from flask import request
from flask_cors import CORS
import whisper
import json
import ast
import openai
import backend.sample_config as config
app = flask.Flask(__name__)
CORS(app)
openai.api_key = config.OPENAI_API_KEY
model ="medium"
audio_model = whisper.load_model(model)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if request.method == 'POST':
        temp_dir = tempfile.mkdtemp()
        save_path = os.path.join(temp_dir, 'temp.wav')
        wav_file = request.files['audio_data']
        wav_file.save(save_path)
        result = audio_model.transcribe(save_path, fp16=False,language='english')
        command=result['text']
        prompt=f"""{command}->"""
        model="davinci:ft-personal-2023-03-27-09-45-59"
        response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=1
        )
        js=response['choices'][0]['text'].strip().split('\n',1)[0].replace("\'", "\"")
        print(js)
        js_obj=json.dumps(js, ensure_ascii=False)
        data = ast.literal_eval(js)
        json_object = json.dumps({"transcribed_text":result['text'],"response":js_obj}, indent=6)
        return json_object
    else:
        return "This endpoint only processes POST wav blob"

