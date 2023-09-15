import speech_recognition as sr
from gtts import gTTS
import playsound
import time

while True:
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print('음성 인식하세여')
        audio = r.listen(source)

        res = r.recognize_google(audio, language='ko')
        print('음성 : '+res)

        if '카카오' in res:
            tts = gTTS(text='반갑습니다', lang='ko')
            tts.save('test.mp3')
            time.sleep(3)
            playsound.playsound('test.mp3', True)  # True는 퍼미션
