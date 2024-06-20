import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="9,11,13"

import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
from time import time
import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
from time import time, sleep

def read_text_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content
    
def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

class Args:
    def __init__(self):
        self.model_path = "checkpoints/llava-v1.5-13b"
        self.model_base = None
        # self.image_file = "/workspace/help_image.jpg"
        self.image_file = "/workspace/help_image.jpg"
        # self.image_file = "/workspace/opencv_frame_1.png"
        self.device = "cuda"
        self.conv_mode = None
        self.temperature = 0.2
        self.max_new_tokens = 512
        self.load_8bit = False
        self.load_4bit = False
        self.debug = False
        self.image_aspect_ratio = 'pad'

args = Args()


disable_torch_init()
model_name = get_model_name_from_path(args.model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, 
                                                                       args.model_base, 
                                                                       model_name, 
                                                                       args.load_8bit, 
                                                                       args.load_4bit, 
                                                                       device=args.device)
if 'llama-2' in model_name.lower():
    conv_mode = "llava_llama_2"
elif "v1" in model_name.lower():
    conv_mode = "llava_v1"
elif "mpt" in model_name.lower():
    conv_mode = "mpt"
else:
    conv_mode = "llava_v0"

if args.conv_mode is not None and conv_mode != args.conv_mode:
    print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
else:
    args.conv_mode = conv_mode

conv = conv_templates[args.conv_mode].copy()
if "mpt" in model_name.lower():
    roles = ('user', 'assistant')
else:
    roles = conv.roles
    
# image = load_image(args.image_file)
# # Similar operation in model_worker.py
# image_tensor = process_images([image], image_processor, args)
# if type(image_tensor) is list:
#     image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
# else:
#     image_tensor = image_tensor.to(model.device, dtype=torch.float16)

# del image
# del image_tensor
def mode0():
    # if(os.path.isfile("/workspace/vqa.txt")):
    #     os.remove("/workspace/vqa.txt")
    # if(os.path.isfile("/workspace/vqa_tmp.txt")):
    #     os.remove("/workspace/vqa_tmp.txt")
    
    
    outstr = ''
    while not os.path.isfile(args.image_file):
        continue
    # text=["Are there any visible signs of bleeding, wounds, or deformities, choose one option: A) Yes, initiate a calm and open-ended dialogue to gather more information and assess the situation. B) No , proceed to the next question, as the current information doesn't indicate immediate concern",
    #       "Does the person's posture or position suggest loss of consciousness,choose one option: A) Yes, initiate a calm and open-ended dialogue to gather more information and assess the situation. B) No , proceed to the next question, as the current information doesn't indicate immediate concern",
    #       "Does the person's position indicate a recent fall or difficulty staying upright,choose one option: A) Yes, initiate a calm and open-ended dialogue to gather more information and assess the situation. B) No , proceed to the next question, as the current information doesn't indicate immediate concern",
    #       "Are there any objects or surroundings suggesting an accident or dangerous situation,choose one option: A) Yes, initiate a calm and open-ended dialogue to gather more information and assess the situation. B) No , proceed to the next question, as the current information doesn't indicate immediate concern", 
    #       "Does the person's facial expression or body language suggest any pain or discomfort,choose one option: A) Yes, initiate a calm and open-ended dialogue to gather more information and assess the situation. B) No , proceed to the next question, as the current information doesn't indicate immediate concern",
    #       "Does the person's overall appearance or posture indicate any immediate distress or need for help,choose one option:A) Yes, initiate a calm and open-ended dialogue to gather more information and assess the situation. B) No , proceed to the next question, as the current information doesn't indicate immediate concern"]
    
    # text=['Can you see if the person has any injuries, like bleeding, bruises, or swelling: A) No; B) Yes',
    #         'Does the person seem unconscious? Look for signs like closed eyes, lack of movement: A) Yes; B) No ',
    #         'Is there any evidence that the person might have fallen, like disarranged furniture or objects: A) No; B) Yes',
    #         'Is there any indication the person is experiencing discomfort, such as grimacing, holding a body part, or positioning themselves to alleviate stress: A) Yes; B) No', 
    #         'Are there any signs that suggest the person might be in pain, like facial expressions of distress, protective postures, or vocalizations: A) Yes; B) No',
    #         ]
    text=['Are there any visible injuries on the person, such as bleeding, bruises, or swelling: A) Yes; B) No',
            'Is there any evidence that the person might have fallen, like disarranged furniture or objects: A) Yes; B) No',
            # ' Is the person pale and sweating: A) Yes; B) No',
            'Does the person lie on the ground: A) Yes; B) No',
            #'Is the person holding any part of the body and grimacing: A) No; B) Yes', 
            #'Is the person grimacing and showing any signs of discomfort: A) No; B) Yes',
            'Are there any signs that suggest the person might be in pain, like facial expressions of distress, protective postures, or vocalizations: A) Yes; B) No',
            #'Does the person seem unwell or ill:  A) No; B) Yes',
            #'Does the person seem crying on the picture: A) Yes; B) No',
            # 'Is the person holding his hands together near chest: A) Yes; B) No ',
            # 'Does the person appear to be leaning forward and resting their head on their hands: A) Yes; B) No ',
            'Does the person hold their chest and showing any signs of discomfort on the face or chest: A) Yes; B) No',
            'Does the person hold their head and showing any signs of discomfort on the face or head: A) Yes; B) No',
            'Does the person hold their arm and showing any signs of discomfort on the face or arm: A) Yes; B) No',
            #'Is the person holding their leg and showing any signs of discomfort: A) Yes; B) No'
            #'Is there any indication the person is experiencing discomfort, such as grimacing, holding a body part, or positioning themselves to alleviate stress: A) Yes; B) No'
             #'Does the person hold their chest?: A) No; B) Yes'
            # 'What class would you give based on the picture and tell why? A) Heart Attack; B) Normal Activity ',
            # 'Tell me why you think so?',
            # 'It is a heart attack scenario on the picture, what questions are better for you to ask in order to detect this heart attack scenario, having only picture as input?'
            
            ]
    for p in range(len(text)):
        print(f"Question: {text[p]}")
    
        if 'llama-2' in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"
    
        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
        else:
            args.conv_mode = conv_mode
    
        conv = conv_templates[args.conv_mode].copy()
        if "mpt" in model_name.lower():
            roles = ('user', 'assistant doctor')
        else:
            roles = conv.roles
    
        try:
            #inp = input(f"{roles[0]}: ")
            inp = ' '
            # sleep(2)
            image = load_image(args.image_file)
            # Similar operation in model_worker.py
            image_tensor = process_images([image], image_processor, args)
            if type(image_tensor) is list:
                image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
            else:
                image_tensor = image_tensor.to(model.device, dtype=torch.float16)


        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"{roles[1]}: ", end="")

        if image is not None:
            #first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                inp += '\n' +'I need your help assessing a possible medical situation. Please carefully examine the person and answer the following questions briefly, choosing either "Yes" or "No":' + text[p]
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
                # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        start = time()
        with torch.inference_mode():
            output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])
        end = time()
        print ("Time for model to asnwer VQA: ",end - start)
        del image
        del image_tensor

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        conv.messages[-1][-1] = outputs
        outstr += outputs + ' '
        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
    with open('/workspace/vqa_tmp.txt', 'w') as f: #open the file
        f.write(outstr)
    sleep(1)
    with open('/workspace/vqa.txt', 'w') as f: #open the file
        f.write(outstr)
        print("recorded: " + outstr)

    
        #print("recorded: " + outstr)
    
    os.remove('/workspace/mode.txt')
    return outstr

def mode1():
    prompts = []
    silence=0
    # dimentia=""
    silence_out=""
    while True:
        while not os.path.isfile(args.image_file):
            sleep(1)
        #text=['Please come up with a question directly to the person to specify the state of the person health', 'Please, identify whether the person needs some help according to his reply: ', 'Please come up with another question directly to the person to specify whether the person needs help']
        text=['You are a virtual doctor that identifies anomalous or emergency situations. I am an old person who lives alone in apartment. I will describe my symptoms, and you will provide a diagnosis and treatment plan. Your responses should only be in the form of questions to gather more information about my symptoms. Avoid repetitions in your questions. Determine if I need an ambulance based on the provided information.']
        #answers = ["I have broken my arm. I need some help. Please, answer shortly", "I have hurt my arm and cannot move or do anything myself. Please, answer shortly"]
        p = 0
        i = 0
        answers = []
        print(f"Question: {text[p]}")
        #for p in range(len(text)):
        while True:
            #print(f"Question: {text[p]}")
        
            if 'llama-2' in model_name.lower():
                conv_mode = "llava_llama_2"
            elif "v1" in model_name.lower():
                conv_mode = "llava_v1"
            elif "mpt" in model_name.lower():
                conv_mode = "mpt"
            else:
                conv_mode = "llava_v0"
        
            if args.conv_mode is not None and conv_mode != args.conv_mode:
                print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
            else:
                args.conv_mode = conv_mode
        
            conv = conv_templates[args.conv_mode].copy()
            if "mpt" in model_name.lower():
                roles = ('user', 'assistant doctor')
            else:
                roles = conv.roles
        
            try:
                #inp = input(f"{roles[0]}: ")
                inp = ' '
                image = load_image(args.image_file)
                # Similar operation in model_worker.py
                image_tensor = process_images([image], image_processor, args)
                if type(image_tensor) is list:
                    image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
                else:
                    image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    
    
            except EOFError:
                inp = ""
            if not inp:
                print("exit...")
                break
    
        
            
            if image is not None:
                #first message
                if model.config.mm_use_im_start_end:
                    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
                else:
                    if p == 0:
                        inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                        #inp += '\n' +"Answer on the questions: " + text[p]
                        inp += '\n' + text[p]
                    elif i < 5 and p > 0:
                        print(i)
                        #questname = "question" + str(i)
                        while not os.path.isfile("/workspace/answer.txt"):
                            continue
                        sleep(1)
                        # sleep(1)
                        with open('/workspace/answer.txt') as f:
                            answ = f.read()
                            if 'Timeout:' in answ or answ == '':
                                silence+=1
                        print(answ)
                        answers.append(answ)
                        
                        seq='Please note that you have access to the previous answers to ensure the questions generated are valid and contribute to obtaining useful information about my health.  \n'
                        
                        seq+= 'Previous questions and answers are here: \n'
                        # seq+= answers[len(answers) - 1]
                        for pr, ans in zip(prompts[:], answers[:]):
                            seq += f" {pr}\n User's anwers: {ans}\n"
                        seq+='Generate a new question based on the provided answers and questions to further assess my health condition, ensuring no repetition of previous questions.'
                        #seq += 'Based on the previous answers create new question to identify the health state of the person, please do not repeat the questions:'
                        inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                        inp += '\n' + seq
                        # if os.path.isfile("/workspace/question_tmp.txt"):
                        #     os.remove("/workspace/question_tmp.txt")
                        #os.remove("/workspace/answer.txt")
                        #os.remove("/workspace/" + str(i) + ".txt")
                        
                        if i==4:
                            i+=1
                            continue
                        
                    elif i == 5:
                        while not os.path.isfile("/workspace/answer.txt"):
                            continue
                        sleep(1)
                        # sleep(1)
                        with open('/workspace/answer.txt') as f:
                            answ = f.read()
                            if 'Timeout:' in answ or answ == '':
                                silence+=1
                        answers.append(answ)
                        print("I am here")
                        inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                        seq = 'You are a virtual doctor that identifies anomalous or emergency situations. I am an old person who lives alone in apartment. \n'
                        seq += 'Please be aware that you have access to the previous interaction, which will assist in making the decision to call an ambulance or not. Consider the provided information for context and coherence. \n'
                        seq += 'Here is your previous interaction: \n'
                        for pr, ans in zip(prompts, answers):
                            seq += f"Bot: {pr}\nUser: {ans}\n"
                        print(silence)
                        if silence>=3:
                            seq+='As there was no response within the previous interaction, it is mandatory to call the ambulance immediately. Please, answer "Yes, we need to call the ambulance, the person is unresponsive and maybe faint".'
                        else:
                            seq += 'Do we need to call an ambulance? Answer shortly:Yes or No.'
                        inp += '\n' + seq
                        
                        
                        # i += 1
                    elif i==6:
                        
                        print("Last here")
                        
                        inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                        seq='You are a virtual doctor that identifies anomalous or emergency situations. I am an old person who lives alone in apartment. \n'
                        seq='Please be aware that you have access to the previous interaction, which will assist in making the decision to call an ambulance or not. Consider the provided information for context and coherence. \n'
                        seq+= 'Here is your previous interaction: \n'
                        for pr, ans in zip(prompts[:], answers[:]):
                            seq += f"Bot: {pr}\n User: {ans}\n"
                        seq+=silence_out
                        #seq+=dimentia
                        
                        # Add a prompt asking for concise medical suggestions
                        seq += 'Please provide short and concise medical suggestions based on the user\'s situation in one paragraph:'
                        print(seq)
                        inp += '\n' + seq
                    # elif i==6:
                    #     print("Dimentia")
                    #     seq=''
                    #     inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                    #     seq+= 'Here is your previous interaction: \n'
                    #     for pr, ans in zip(prompts[:], answers[:]):
                    #         seq += f"nBot: {pr}\nUser: {ans}\n"
                    #     #seq+='Evaluate the coherence of the interaction so far in the context of dementia symptoms. \n '
                    #     #seq+= "Choose the most appropriate option, choose one option: A) The conversation is highly coherent, and there are no signs of confusion or memory loss; B) There are some minor disruptions in coherence, but they are not alarming considering the nature of the conversation about dementia.; C) the conversation lacks coherence, and there are clear signs of confusion or memory loss.\n"
                    #     if silence>2:
                    #         seq+='Please, answer "The person is unresponsive and maybe faint".'
                    #     else:
                    #         seq += "Carefully analyze the conversation so far, focusing on how ideas connect, memories are retained, and language is used. \n"
                    #         seq += "Consider these specific aspects:\n"
                    #         seq += "- Topic consistency: Does the conversation stay on track, or are there sudden shifts or unrelated topics?\n"
                    #         seq += "- Logical flow: Do the ideas build upon each other, or are there gaps or inconsistencies in reasoning?\n"
                    #         seq += "- Memory retention: Are details repeated, or are previously mentioned points forgotten?\n"
                    #         seq += "- Word choice and sentence structure: Is the language appropriate, or are there difficulties expressing thoughts?\n"
                    #         seq += "Based on these factors, assess the coherence of the interaction in relation to dementia symptoms and choose the most fitting option, choose only one option:\n"
                    #         seq += "A) Conversation progresses naturally and coherently, with clear connections between ideas and consistent information recall.\n"
                    #         seq += "B) Occasional topic jumps or difficulty recalling precise details, but the conversation remains generally focused and understandable.\n"
                    #         seq += "C) Frequent shifts in topic, notable gaps in information recall, or challenges in piecing together the conversation's overall logic become evident.\n"
                    #         seq+=  "D) Unresponsive (most of the time 'Timeout: No speech detected within the specified time').\n"
                    #     inp += '\n' + seq
        
                        

                conv.append_message(conv.roles[0], inp)
                

                #image = None
            else:
                    # later messages
                image = None
                conv.append_message(conv.roles[0], inp)
            
            print(f"{roles[1]}: ", end="")
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
        
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            start = time()
            with torch.inference_mode():
                output_ids = model.generate(
                        input_ids,
                        images=image_tensor,
                        do_sample=True,
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_tokens,
                        streamer=streamer,
                        use_cache=True,
                        stopping_criteria=[stopping_criteria])
            end = time()
            print ("Time: ", end - start)
            
            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            if silence>=3 and i==5:
                silence_out+=outputs
            # if i==6:
            #     dimentia+=outputs
            if args.debug:
                print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
            if i == 5:
                yes = "Yes"
                if yes in outputs:
                    with open('/workspace/yes.txt', 'w') as f:
                        f.write(outputs)
                        print('Yes created and recorded: ' + outputs)
            if i<6:
                with open('/workspace/question' + str(i+1) + "_tmp.txt", 'w') as f: #open the file
                    f.write(outputs)
                    print("question number" + str(i+1))
                if os.path.isfile('/workspace/answer.txt'):
                    os.remove('/workspace/answer.txt')
                #sleep(1)
                # line_check = '' 
                # while line_check != outputs: 
                #     with open('/workspace/question_tmp.txt', 'r') as fr: 
                #         line_check = fr.read() 
                with open('/workspace/question' + str(i+1) + ".txt", 'w') as f2: #open the file
                    f2.write(outputs)
                    print("recorded: " + outputs)
                    print("question number" + str(i+1))
                
                    #print("recorded: " + outputs)
            else:
                
                with open('/workspace/suggestion_tmp.txt', 'w') as f: #open the file
                    f.write(outputs)
                    #print("recorded: " + outputs)
                with open('/workspace/suggestion.txt', 'w') as f: #open the file
                    f.write(outputs)
                    print("recorded: " + outputs)
            conv.messages[-1][-1] = outputs
            del image
            del image_tensor
        
            prompts.append(outputs)
            if args.debug:
                print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
            p += 1
            i += 1
            
            if i == 7:
                break
        
        
        break

def main():
    try:
        while True:
            while not os.path.isfile("/workspace/mode.txt"):
                continue
            sleep(1)
            
            with open('/workspace/mode.txt', 'r') as f: #open the file
                mode = f.read()
            print(mode)
            if int(mode) == 0:
                outstr = mode0()
                print(outstr)
                if os.path.isfile("/workspace/mode.txt"):
                    os.remove("/workspace/mode.txt")    
                # with open("/workspace/vqa.txt") as f:
                #     line = f.read()
                #     print(line)            
            elif int(mode) == 1:
                mode1()
                sleep(2)
                for i in range(6):
                    if os.path.exists("/workspace/question" + str(i+1) + ".txt"):
                        os.remove("/workspace/question" + str(i+1) + ".txt")
                    if os.path.exists("/workspace/question" + str(i+1) + "_tmp.txt"):
                        os.remove("/workspace/question" + str(i+1) + "_tmp.txt")
                if os.path.exists("/workspace/question.txt"):
                    os.remove("/workspace/question.txt")
                if os.path.exists("/workspace/help_image.jpg"):
                    os.remove("/workspace/help_image.jpg")
                if os.path.exists("/workspace/test1.wav"):
                    os.remove("/workspace/test1.wav")
                if os.path.exists("/workspace/vqa.txt"):
                    os.remove("/workspace/vqa.txt")
                # if os.path.exists("/workspace/vqa_tmp.txt"):
                #     os.remove("/workspace/vqa_tmp.txt")
                if os.path.exists("/workspace/answer.txt"):
                    os.remove("/workspace/answer.txt")
                if os.path.exists("/workspace/suggestion.txt"):
                    os.remove("/workspace/suggestion.txt")
                if os.path.exists("/workspace/suggestion_tmp.txt"):
                    os.remove("/workspace/suggestion_tmp.txt")
                if os.path.exists("/workspace/mode.txt"):
                    os.remove("/workspace/mode.txt")
                if os.path.exists("/workspace/yes.txt"):
                    os.remove("/workspace/yes.txt")
                #os.remove(args.image_file)

    except KeyboardInterrupt:
        print("KeyboardInterrupt: Cleaning up...")
        for i in range(6):
            if os.path.exists("/workspace/question" + str(i+1) + ".txt"):
                os.remove("/workspace/question" + str(i+1) + ".txt")
            if os.path.exists("/workspace/question" + str(i+1) + "_tmp.txt"):
                os.remove("/workspace/question" + str(i+1) + "_tmp.txt")
        # Delete the file if it exists
        if os.path.exists("/workspace/question.txt"):
            os.remove("/workspace/question.txt")
        if os.path.exists("/workspace/help_image.jpg"):
            os.remove("/workspace/help_image.jpg")
        if os.path.exists("/workspace/test1.wav"):
            os.remove("/workspace/test1.wav")
        if os.path.exists("/workspace/vqa.txt"):
            os.remove("/workspace/vqa.txt")
        # if os.path.exists("/workspace/vqa_tmp.txt"):
        #     os.remove("/workspace/vqa_tmp.txt")
        if os.path.exists("/workspace/answer.txt"):
            os.remove("/workspace/answer.txt")
        if os.path.exists("/workspace/suggestion.txt"):
            os.remove("/workspace/suggestion.txt")
        if os.path.exists("/workspace/suggestion_tmp.txt"):
            os.remove("/workspace/suggestion_tmp.txt")
        if os.path.exists("/workspace/mode.txt"):
            os.remove("/workspace/mode.txt")
        if os.path.exists("/workspace/yes.txt"):
            os.remove("/workspace/yes.txt")

if __name__ == "__main__":
    main()


    