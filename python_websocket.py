import asyncio
import websockets
import openai
import backend.sample_config as config
import whisper
import tempfile
import os
import json
import ast

openai.api_key = config.OPENAI_API_KEY
model ="medium"
audio_model = whisper.load_model(model)

async def server(websocket):
    async for message in websocket:
        print(message)
        transcribed_message=transcribe(message)
        intent =getIntent(transcribed_message)
        await websocket.send({intent})

async def main():
    async with websockets.serve(server, "localhost", 8765):
        await asyncio.Future() 

asyncio.run(main())

def transcribe(audio):
        temp_dir = tempfile.mkdtemp()
        save_path = os.path.join(temp_dir, 'temp.wav')
        wav_file = audio.files['audio_data']
        wav_file.save(save_path)
        result = audio_model.transcribe(save_path, fp16=False,language='english')
        return result

def getIntent(message):  
        prompt=f"""{message}->"""
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
        js=response['choices'][0]['text'].strip().split('\n',1)[0]
        print(js)
        data = ast.literal_eval(js)
        json_string = json.dumps(data)
        print(json_string)
        json_object = {"transcribed_text":message}
        json_object.update(data)
        print(json_object)
        return json_object


